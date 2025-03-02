import random
import re
from typing import Any, List

from datasets import Dataset, concatenate_datasets, load_dataset, train_test_split
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
        self.parser = XMLParser(fields=["think", "answer"])

    def get_dataset(self) -> Dataset:
        dataset = load_dataset(self.dataset_name)["train"]

        word_examples = dataset.filter(lambda x: x["task"] == "word")
        word_pair_examples = dataset.filter(lambda x: x["task"] == "word_pair")
        sentence_examples = dataset.filter(lambda x: x["task"] == "sentence")

        # split into train and test
        word_examples, word_examples_test = train_test_split(
            word_examples, test_size=0.2, seed=42
        )
        word_pair_examples, word_pair_examples_test = train_test_split(
            word_pair_examples, test_size=0.2, seed=42
        )
        sentence_examples, sentence_examples_test = train_test_split(
            sentence_examples, test_size=0.2, seed=42
        )

        # the test dataset is just the concatenation of the three test datasets
        test_dataset = concatenate_datasets(
            [word_examples_test, word_pair_examples_test, sentence_examples_test]
        )

        # the train dataset is interleaved from the three tasks, sampling randomly
        train_dataset_length = 100000
        train_dataset = []
        while len(train_dataset) < train_dataset_length:
            word_idx = random.randint(0, len(word_examples) - 1)
            word_example = word_examples[word_idx]
            train_dataset.append(word_example)

            word_pair_idx = random.randint(0, len(word_pair_examples) - 1)
            word_pair_example = word_pair_examples[word_pair_idx]
            train_dataset.append(word_pair_example)

            sentence_idx = random.randint(0, len(sentence_examples) - 1)
            sentence_example = sentence_examples[sentence_idx]
            train_dataset.append(sentence_example)

        train_dataset = Dataset.from_list(train_dataset)

        return train_dataset, test_dataset

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        def correctness_reward_func(completions, **kwargs) -> List[float]:
            """
            1.0 if exactly correct, otherwise 0.0
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

            rewards = [
                check_answer(r, t) for r, t in zip(responses, true_decoded_messages)
            ]
            return rewards

        def check_format(text: str) -> float:
            """
            Helper function to check if the format is correct
            """
            # remove the bootstrap prompt from the text if it appears at the start
            text = text.removeprefix("Let me solve this step by step.\n")

            try:
                # Check if the format is correct
                regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

                match = re.search(regex, text, re.DOTALL)

                if match is None or len(match.groups()) != 2:
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
            correctness_reward_func,
            format_reward_func,
        ]
