import re 


def print_reward(*,
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
    target = kwargs["label"] if kwargs['task'] == 'recognition' else kwargs['total']

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
            additional_fields=["task", 'label', 'total'],
            reward_function_kwargs=kwargs,
        )

    return rewards

def answer_reward_func(completions, **kwargs):
    """
    Evaluates if the answer is exactly correct
    """
    
    PRINT_RESULTS = True
    
    rewards = []
    
    tasks = kwargs['task']
    
    # verify that all the tasks are the same 
    if not all(task == tasks[0] for task in tasks):
        raise ValueError(f"Tasks are not all the same: {tasks}, invalidates an assumption of the code")
    
    task = tasks[0]

    completion_texts = extract_completion_texts(completions)

    if task == "recognition":
        rewards = _recognition_answer_reward_func(completions=completions, completion_texts=completion_texts, **kwargs)
    elif task == "addition":
        rewards = _addition_answer_reward_func(completions=completions, completion_texts=completion_texts, **kwargs)
    else:
        raise ValueError(f"Invalid task: {task}")
    
    if PRINT_RESULTS:
        print_reward(
            function_name="answer_reward_func",
            prompts=kwargs["prompts_text"],
            completions=completion_texts,
            target=kwargs["label"] if task == "recognition" else kwargs["total"],
            rewards=rewards,
            additional_fields=["task", 'label', 'total'],
            reward_function_kwargs=kwargs,
        )
    
    return rewards

def _recognition_answer_reward_func(completions, completion_texts, **kwargs):
    """
    Evaluates if the answer is exactly correct for the recognition task
    """
    
    rewards = []
    # in this case the target is the list of a list of digits
    target = kwargs['label']
    
    for completion, gt in zip(completion_texts, target):
        if type(gt) != list:
            raise ValueError(f"Target is not a list: {gt}! This is unexpected.")
        try:
            # extract the answer
            match = re.search(r"<answer>(.*?)<\/answer>", completion)
            
            if match is None:
                rewards.append(0.0)
                continue

            answer = match.group(1).strip()
            
            allowed_pattern = r"^\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\]\s*$"
            
            if not re.match(allowed_pattern, answer):
                rewards.append(0.0)
                continue
            
            # convert the answer from str(list) to list
            answer = eval(answer, {"__builtins__": None}, {})
            
            if not isinstance(answer, list):
                rewards.append(0.0)
                continue
            
            
            if answer == gt:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        
        except Exception as e:
            print(f"Error in _recognition_answer_reward_func: {e}")
            rewards.append(0.0)
    
    return rewards

def _addition_answer_reward_func(completions, completion_texts, **kwargs):
    """
    Evaluates if the answer is exactly correct for the addition task
    """
    
    rewards = []

    # in this case the target is the sum of the digits
    target = kwargs['total']
    for completion, gt in zip(completion_texts, target):
        if type(gt) != int:
            raise ValueError(f"Target is not an int: {gt}! This is unexpected.")
        
        try:
            # extract the answer
            match = re.search(r"<answer>(.*?)<\/answer>", completion)

            if match is None:
                rewards.append(0.0)
                continue

            answer = match.group(1).strip()
            
            # regex pattern that only allows numbers and whitespace
            allowed_pattern = r"^[\d\s]+$"
            
            if not re.match(allowed_pattern, answer):
                rewards.append(0.0)
                continue
            
            answer = int(answer)
            
            if answer == gt:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        
        except Exception as e:
            print(f"Error in _addition_answer_reward_func: {e}")
            rewards.append(0.0)
    
    return rewards
            