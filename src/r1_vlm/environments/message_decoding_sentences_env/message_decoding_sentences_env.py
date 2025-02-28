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
        self.parser = XMLParser(fields=["think", "answer"])

    def get_dataset(self) -> Dataset:
        dataset = load_dataset(self.dataset_name)["train"]

        # Curriculm learning
        # select 500 "word" examples to start with
        # then select all "sentence" examples, sorted from shortest to longest by "decoded_message" length

        word_examples = dataset.filter(lambda x: x["task"] == "word")
        sentence_examples = dataset.filter(lambda x: x["task"] == "sentence")

        # choose 500 random "word" examples
        word_examples = word_examples.shuffle(seed=42).select(range(500))

        # Add a length column
        def add_length(example):
            example["length"] = len(example["decoded_message"])
            return example

        sentence_examples = sentence_examples.map(add_length)

        # Sort by length
        sentence_examples = sentence_examples.sort("length")
        sentence_examples = sentence_examples.remove_columns("length")

        # concatenate the "word" and "sentence" examples
        dataset = concatenate_datasets([word_examples, sentence_examples])

        return dataset

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

        def edit_distance_reward_func(completions, **kwargs) -> List[float]:
            """
            Reward proportional to the edit distance normalized by the length of the GT string and clamped between 0 and 1.

            Given GT and proposed string (PS):
                e_d = edit_distance(GT, PS)
                reward = 1 - e_d / max(len(GT), len(PS))
                reward = min(max(0.0, reward), 1.0)
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

                    edit_distance = Levenshtein.distance(response, answer)
                    reward = 1 - edit_distance / max(len(answer), len(response))
                    reward = min(max(0.0, reward), 1.0)
                    return reward
                except Exception as e:
                    print(f"Error in check_answer for edit distance: {e}")
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

        def thinking_reward(completions, **kwargs):
            """
            Rewards the model for thinking longer about the problem. Requires the model to properly format its response
            to achieve this reward.
            """

            def check_format_bool(text: str) -> bool:
                """
                Returns True if the model properly formatted
                """
                return check_format(text) == 1.0

            # find the completions that achieve the format reward
            formatted_properly = [
                check_format_bool(c[0]["content"]) for c in completions
            ]

            # if nothing is formatted properly, no thinking reward
            if not any(formatted_properly):
                return [0.0] * len(completions)

            # find how long each completion "thinks" for
            thinking_texts = [
                self.parser.parse(c[0]["content"]).think for c in completions
            ]
            thinking_texts = ["" if t is None else t for t in thinking_texts]
            thinking_lengths = [len(t) for t in thinking_texts]

            # find the max length that was formatted properly
            max_len = 0
            for length, formatted_properly in zip(thinking_lengths, formatted_properly):
                if formatted_properly:
                    max_len = max(max_len, length)

            # if no one thought, no thinking reward - e.g. <think></think>
            if max_len == 0:
                return [0.0] * len(completions)

            # normalize the thinking lengths by the max length
            normalized_thinking_lengths = [
                length / max_len for length in thinking_lengths
            ]

            rewards = []
            for normalized_length, formatted_properly in zip(
                normalized_thinking_lengths, formatted_properly
            ):
                if formatted_properly:
                    rewards.append(normalized_length)
                else:
                    rewards.append(0.0)

            return rewards

        return [
            correctness_reward_func,
            edit_distance_reward_func,
            format_reward_func,
            thinking_reward,
        ]


if __name__ == "__main__":
    env = MessageDecodingEnv()
    dataset = env.get_dataset()
    import ipdb

    ipdb.set_trace()
