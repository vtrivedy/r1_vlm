import re


def print_reward(
    *,
    function_name,
    prompts,
    completions,
    target,
    rewards,
    additional_fields,
    reward_function_kwargs,
):
    """
    Prints reward function results with arbitrary additional fields.

    Args:
        function_name (str): Name of the reward function being used
        prompts (list): List of input prompts
        completions (list): List of model completions
        target (list): List of target values
        rewards (list): List of reward values
        additional_fields (list[str]): List of additional field names to print from kwargs
        reward_function_kwargs (dict): Kwargs containing additional data fields
    """
    print(f"\nExecuting {function_name}")
    print("=" * 100)

    for idx, (prompt, completion, gt, reward) in enumerate(
        zip(prompts, completions, target, rewards)
    ):
        # Clean up image padding tokens
        prompt_cleaned = re.sub(
            r"(<\|image_pad\|>)+", "{... many image pad tokens ...}", prompt
        )
        print(f"Function name: {function_name}")
        print(f"Sample {idx + 1}:")
        print(f"Prompt:\n{prompt_cleaned}")
        print(f"Completion:\n{completion}")
        print(f"Target: {gt}")
        print(f"Reward: {reward}")

        # Print any additional fields requested
        for field in additional_fields:
            if field in reward_function_kwargs:
                value = reward_function_kwargs[field][idx]
                print(f"{field}: {value}")
            else:
                print(f"Can't find {field} in reward_function_kwargs")

        print("-" * 100)


def extract_completion_texts(completions):
    """
    Extracts cleaned completion text as strings from model conversation outputs.
    Removes bootstrap prompt and keeps only content starting at the first <think> tag.

    Args:
        completions (list): List of completion conversation objects

    Returns:
        list[str]: List of cleaned completion texts
    """
    completion_texts = []
    for completion_conv in completions:
        try:
            completion = completion_conv[0]["content"]
            # remove anything before the first <think> tag (bootstrap prompt)
            completion = re.search(r".*?(<think>[\s\S]*)", completion).group(1)
            completion_texts.append(completion)
        except Exception as e:
            print(f"Error processing completion: {e}")
            completion_texts.append("")
    return completion_texts


def format_reward_func(completions, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers

      Returns:
          list[float]: Reward scores
    """
    PRINT_RESULTS = False

    rewards = []

    target = kwargs["decoded_message"]

    completion_texts = extract_completion_texts(completions)

    for completion, gt in zip(completion_texts, target):
        try:
            # Check if the format is correct
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

            match = re.search(regex, completion, re.DOTALL)

            # if the format is not correct, reward is 0
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception as e:
            print(f"Error in format_reward_func: {e}")
            rewards.append(0.0)

    if PRINT_RESULTS:
        print_reward(
            function_name="format_reward_func",
            prompts=kwargs["prompts_text"],
            completions=completion_texts,
            target=target,
            rewards=rewards,
            additional_fields=["coded_message", "decoded_message", "mapping"],
            reward_function_kwargs=kwargs,
        )

    return rewards


def answer_reward_func(completions, **kwargs):
    """
    Rewards for the exact match of the decoded message
    """
    PRINT_RESULTS = True

    rewards = []
    targets = kwargs["decoded_message"]

    completion_texts = extract_completion_texts(completions)

    for completion, target in zip(completion_texts, targets):
        try:
            # Extract answer: requires the predicted decoded message is within <answer>...</answer>
            match = re.search(r"<answer>(.*?)<\/answer>", completion)

            if match is None:
                rewards.append(0.0)
                continue
            predicted = match.group(1).strip()
            # Remove any non-letter/non-space characters from prediction
            predicted = re.sub(r"[^A-Za-z\s]", "", predicted)

            if predicted == target:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception as e:
            print(f"Error in answer_reward_func: {e}")
            rewards.append(0.0)

    if PRINT_RESULTS:
        print_reward(
            function_name="answer_reward_func",
            prompts=kwargs["prompts_text"],
            completions=completion_texts,
            target=targets,
            rewards=rewards,
            additional_fields=["coded_message", "decoded_message", "mapping"],
            reward_function_kwargs=kwargs,
        )

    return rewards
