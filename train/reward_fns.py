import re

# https://www.philschmid.de/mini-deepseek-r1#3-train-the-model-using-grpo-educational-part was the best reference for this


def format_reward_func(completions, target, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers

      Returns:
          list[float]: Reward scores
    """
    rewards = []

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

    return rewards


def answer_reward_func(completions, target, **kwargs):
    """
    Evaluates completions based on mathematical correctness of the answer

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        nums (list[str]): Available numbers

    Returns:
        list[float]: Reward scores
    """

    rewards = []

    for completion_conv, gt in zip(completions, target):
        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion_conv[0]["content"]

            # Check if the format is correct
            match = re.search(r"<answer>(.*?)<\/answer>", completion)

            if match is None:
                rewards.append(0.0)
                continue

            answer = match.group(1).strip()

            # Define a regex pattern that only allows numbers, period, and whitespace
            allowed_pattern = r"^[\d\s.]+$"
            if not re.match(allowed_pattern, answer):
                rewards.append(0.0)
                continue

            # check if float(answer) == gt (target is already a float)
            assert isinstance(gt, float), "target is not a float but a %s" % type(gt)

            # convert answer to float
            answer = float(answer)

            # check if the two are within e-5 of each other
            if abs(answer - gt) < 1e-5:
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        except Exception as e:
            print(f"Error in answer_reward_func: {e}")
            rewards.append(0.0)

    for completion_conv, target, reward in zip(completions, target, rewards):
        print(f"Completion: \n {completion_conv[0]['content']}")
        print(f"Target: {target}")
        print(f"Reward: {reward}")
        print("-" * 100)
    return rewards


def test_reward_funcs():
    # simple correct example, int answer
    correct_sample_1 = (
        "We need to count the number of people in the image. There are 2 people in the image. "
        "We need to count the number of bananas in the image. There are 4 bananas in the image. "
        "We need to multiply the number of people by the number of bananas. 2 * 4 = 8. </think>\n"
        "<answer> 8 </answer>"
    )

    # correct example, float answer
    correct_sample_2 = (
        "We need to count the number of people in the image. There are 2 people in the image. "
        "We need to count the number of bananas in the image. There are 4 bananas in the image. "
        "We need to multiply the number of people by the number of bananas. 2 * 4 = 8.0 </think>\n"
        "<answer> 8.0 </answer>"
    )

    # correct example, division needed, truncation needed
    correct_sample_3 = (
        "We need to count the number of people in the image. There are 1 people in the image. "
        "We need to count the number of bananas in the image. There are 3 bananas in the image. "
        "We need to divide the number of people by the number of bananas. 1 / 3 = 0.3333333333333333.</think>\n"
        "<answer> 0.33 </answer>"
    )

    # correct example, no thinking
    correct_sample_4 = "</think>\n<answer> 8 </answer>"

    # incorrect format, no end of thinking, no start of answer.
    wrong_format_1 = "The answer is 8"

    # multiple thinking sections
    wrong_format_2 = (
        "We need to count the number of people in the image. There are 1 people in the image. </think> "
        "<think>We need to count the number of bananas in the image. There are 3 bananas in the image. </think> "
        "<think>We need to divide the number of people by the number of bananas. 1 / 3 = 0.3333333333333333.</think> "
        "<answer> 0.33 </answer>"
    )

    # correct format (no thinking), wrong result
    wrong_result_1 = "</think><answer> 6 </answer>"

    test_rewards_format = format_reward_func(
        completions=[
            correct_sample_1,
            correct_sample_2,
            correct_sample_3,
            correct_sample_4,
            wrong_format_1,
            wrong_format_2,
            wrong_result_1,
        ],
        target=[8.0, 8.0, 0.33, 8.0, 8.0, 8.0, 8.0],
    )
    assert test_rewards_format == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], (
        "format reward func not working properly"
    )

    test_rewards_answer = answer_reward_func(
        completions=[
            correct_sample_1,
            correct_sample_2,
            correct_sample_3,
            correct_sample_4,
            wrong_format_1,
            wrong_format_2,
            wrong_result_1,
        ],
        target=[8.0, 8.0, 0.33, 8.0, 8.0, 8.0, 8.0],
    )

    assert test_rewards_answer == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], (
        "answer reward func not working properly"
    )


if __name__ == "__main__":
    test_reward_funcs()
