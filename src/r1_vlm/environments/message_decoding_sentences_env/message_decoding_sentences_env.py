import random
import re
from typing import Any, List

import Levenshtein
from datasets import Dataset, concatenate_datasets, load_dataset
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.parsers import XMLParser

from r1_vlm.environments.simple_vision_env import SimpleVisionEnv


class MessageDecodingEnv(SimpleVisionEnv):
    def __init__(
        self,
        dataset: str = "sunildkumar/message-decoding-words-and-sentences-r1",
        system_prompt: str = "",
        **kwargs,  # passed to the superclass
    ):
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.dataset_name = dataset
        self.parser = XMLParser(fields=["think", "answer", "chars"])

    def get_dataset(self) -> tuple[Dataset, Dataset]:
        dataset = load_dataset(self.dataset_name)["train"]

        word_examples = dataset.filter(lambda x: x["task"] == "word")
        word_pair_examples = dataset.filter(lambda x: x["task"] == "word_pair")
        sentence_examples = dataset.filter(lambda x: x["task"] == "sentence")

        # split into train and test
        word_examples = word_examples.train_test_split(test_size=0.2, seed=42)
        word_examples_train = word_examples["train"]
        word_examples_test = word_examples["test"]

        word_pair_examples = word_pair_examples.train_test_split(test_size=0.2, seed=42)
        word_pair_examples_train = word_pair_examples["train"]
        word_pair_examples_test = word_pair_examples["test"]

        sentence_examples = sentence_examples.train_test_split(test_size=0.2, seed=42)
        sentence_examples_train = sentence_examples["train"]
        sentence_examples_test = sentence_examples["test"]

        # the test dataset is just the concatenation of the three test datasets
        test_dataset = concatenate_datasets(
            [word_examples_test, word_pair_examples_test, sentence_examples_test]
        )

        # the train dataset is interleaved from the three tasks, sampling randomly
        train_dataset_length = 100000
        train_dataset = []
        while len(train_dataset) < train_dataset_length:
            word_idx = random.randint(0, len(word_examples_train) - 1)
            word_example = word_examples_train[word_idx]
            train_dataset.append(word_example)

            word_pair_idx = random.randint(0, len(word_pair_examples_train) - 1)
            word_pair_example = word_pair_examples_train[word_pair_idx]
            train_dataset.append(word_pair_example)

            sentence_idx = random.randint(0, len(sentence_examples_train) - 1)
            sentence_example = sentence_examples_train[sentence_idx]
            train_dataset.append(sentence_example)

        train_dataset = Dataset.from_list(train_dataset)

        return train_dataset, test_dataset

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        def chars_intermediate_reward_func(completions, **kwargs) -> List[float]:
            """
            Reward function that checks if the <chars> section is correct. Reward is proportional to 1 - edit distance.
            """
            responses = [self.parser.parse(c[0]["content"]).chars for c in completions]
            true_chars = kwargs["decoded_message"]
            coded_messages = kwargs["coded_message"]

            # convert true chars to the format we expect in <chars>
            # e.g. "cat dog" -> "c a t _ d o g" or "cat" -> "c a t"
            def format_chars(text: str) -> str:
                words = text.split()
                spaced_words = [" ".join(word) for word in words]
                return " _ ".join(spaced_words)

            formatted_true_chars = [format_chars(msg) for msg in true_chars]

            def check_chars(response, answer, coded_message):
                if response is None:
                    return 0.0
                try:
                    response = response.strip()
                    answer = answer.strip()

                    edit_distance_answer = Levenshtein.distance(response, answer)
                    edit_distance_coded_message = Levenshtein.distance(
                        response, coded_message
                    )

                    # no reward if the chars data is more similar to the coded message than the answer
                    if edit_distance_coded_message < edit_distance_answer:
                        return 0.0

                    reward = 1 - edit_distance_answer / max(len(answer), len(response))

                    reward = min(max(0.0, reward), 1.0)

                    return reward

                except Exception:
                    return 0.0

            rewards = [
                check_chars(r, t, c)
                for r, t, c in zip(responses, formatted_true_chars, coded_messages)
            ]
            return rewards

        def correctness_reward_func(completions, **kwargs) -> List[float]:
            """
            1.0 if exactly correct, otherwise 0.0. Conditioned on getting the the <chars> section correct.
            """
            # parse the predicted decoded message from each completion
            responses = [self.parser.parse(c[0]["content"]).answer for c in completions]
            true_decoded_messages = kwargs["decoded_message"]

            def check_answer(response, answer):
                # the parser returns None if the answer is not found
                if response is None:
                    return 0.0

                try:
                    response = response.strip()
                    answer = answer.strip()
                except Exception as e:
                    print(f"Error in check_answer for correctness: {e}")
                    return 0.0

                if response == answer:
                    return 1.0
                else:
                    return 0.0

            # check which completions got the <chars> section correct - boolean mask
            chars_intermediate_reward = chars_intermediate_reward_func(
                completions, **kwargs
            )
            chars_correct = [r == 1.0 for r in chars_intermediate_reward]

            answers_correct = [
                check_answer(r, t) for r, t in zip(responses, true_decoded_messages)
            ]

            # achieves the answer reward IFF <chars> correct as well.
            rewards = []
            for char_correct, answer_correct in zip(chars_correct, answers_correct):
                if char_correct:
                    rewards.append(answer_correct)
                else:
                    rewards.append(0.0)

            return rewards

        def check_format(text: str) -> float:
            """
            Helper function to check if the format is correct

            Desired format:
            <think>
            Some initial thinking
            <chars> c a t </chars>
            More thinking after
            </think>
            <answer> cat </answer>

            # TODO: This isn't exactly correct as we allow more than one think section.
            """
            # remove the bootstrap prompt from the text if it appears at the start
            text = text.removeprefix("Let me solve this step by step.\n")

            try:
                # Check if the format is correct
                regex = r"^<think>([\s\S]*?)<chars>([\s\S]*?)<\/chars>[\s\S]*?<\/think>\n<answer>([\s\S]*?)<\/answer>$"

                regex_match = re.search(regex, text, re.DOTALL)

                if regex_match is None or len(regex_match.groups()) != 3:
                    return 0.0
                else:
                    return 1.0
            except Exception:
                return 0.0

        def format_reward_func(completions, **kwargs) -> List[float]:
            """
            Reward function that checks for proper XML formatting

            Must have think and answer fields
            """

            rewards = [check_format(c[0]["content"]) for c in completions]
            return rewards

        # removed thinking_reward as it was actively hurting performance
        return [
            chars_intermediate_reward_func,
            correctness_reward_func,
            format_reward_func,
        ]


if __name__ == "__main__":
    # Test showing we allow more than one think section. Example 1 should fail and 2 should pass, but both pass right now.
    env = MessageDecodingEnv()
    functions = env.get_rubric()
    format_fn = functions[2]

    example1 = """<think>I will go through each character in the coded message "j n v t l y" one by one and use the decoder table to find the corresponding decoded character.
j → p
n → l
v → e
t → n
l → t
y → y
<chars> p l e n t y </chars>
</think>
<think>
I have now decoded the entire coded message.
</think>
<answer> pylent </answer>"""

    example1 = [{"content": example1}]

    print(f"Example 1: {format_fn([example1])}")

    example2 = """<think>I will go through each character in the coded message "j n v t l y" one by one and use the decoder table to find the corresponding decoded character.
j → p
n → l
v → e
t → n
l → t
y → y
<chars> p l e n t y </chars>
</think>
<answer> pylent </answer>"""

    example2 = [{"content": example2}]

    print(f"Example 2: {format_fn([example2])}")
