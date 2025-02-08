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

    for idx, (prompt, completion_conv, gt, reward) in enumerate(
        zip(prompts, completions, target, rewards)
    ):
        # Clean up image padding tokens
        prompt_cleaned = re.sub(
            r"(<\|image_pad\|>)+", "{... many image pad tokens ...}", prompt
        )
        print(f"Function name: {function_name}")
        print(f"Sample {idx + 1}:")
        print(f"Prompt:\n{prompt_cleaned}")
        print(f"Completion:\n{completion_conv[0]['content']}")
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

    for completion_conv, gt in zip(completions, target):
        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion_conv[0]["content"]

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
            prompts=kwargs["prompts"],
            completions=completions,
            target=target,
            rewards=rewards,
            additional_fields=["coded_message", "decoded_message", "mapping"],
            reward_function_kwargs=kwargs,
        )

    return rewards


def answer_reward_func(completions, **kwargs):
    """
    Returns edit distance
    """


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Returns the edit distance between two strings.
    """
    if not isinstance(s1, str) or not isinstance(s2, str):
        raise ValueError("Input must be strings")

    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1
    dp = list(range(len2 + 1))
    for i in range(1, len1 + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, len2 + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j], dp[j - 1], prev)
            prev = temp
    return dp[len2]


def soft_edit_distance_reward(completions, **kwargs):
    PRINT_RESULTS = True

    rewards = []
    targets = kwargs["decoded_message"]

    for completion_conv, target in zip(completions, targets):
        try:
            # Extract answer: requires the predicted decoded message is within <answer>...</answer>
            completion = "<think>" + completion_conv[0]["content"]
            match = re.search(r"<answer>(.*?)<\/answer>", completion)

            if match is None:
                rewards.append(0.0)
                continue
            predicted = match.group(1).strip()

            # the maximum possible edit distance is the length of the longer string
            max_len = max(len(target), len(predicted))

            if max_len == 0:
                raise ValueError(
                    "Max length is 0. Soemthing is not right because the target is always non-empty"
                )
            else:
                distance = levenshtein_distance(predicted, target)
                reward = 1.0 - (distance / max_len)
                reward = max(0.0, min(1.0, reward))
                rewards.append(reward)
        except Exception as e:
            print(f"Error in soft_edit_distance_reward: {e}")
            rewards.append(0.0)

    if PRINT_RESULTS:
        print_reward(
            function_name="soft_edit_distance_reward",
            prompts=kwargs["prompts"],
            completions=completions,
            target=targets,
            rewards=rewards,
            additional_fields=["coded_message", "decoded_message", "mapping"],
            reward_function_kwargs=kwargs,
        )

    return rewards
