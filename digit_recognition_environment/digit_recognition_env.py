from typing import Any, List

from datasets import Dataset, load_dataset
from simple_vision_env import SimpleVisionEnv
from transformers import AutoProcessor
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.parsers import XMLParser


class DigitRecognitionEnv(SimpleVisionEnv):
    def __init__(
        self,
        dataset: str = "sunildkumar/digit-recognition-r1",
        system_prompt: str = "",
        processor_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        **kwargs,  # passed to the superclass
    ):
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.dataset_name = dataset
        self.parser = XMLParser(fields=["think", "answer"])
        self.processor = AutoProcessor.from_pretrained(processor_name)

    def get_dataset(self, **kwargs: Any) -> Dataset:
        dataset = load_dataset(self.dataset_name)
        # Filter for recognition task only
        dataset = dataset.filter(lambda x: x["task"] == "recognition")
        return dataset

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        def correctness_reward_func(completions, answer, **kwargs) -> List[float]:
            """Reward function that checks if the predicted digits match the true labels"""
            responses = [self.parser.parse(c[0]["content"]).answer for c in completions]

            def parse_list(r: str) -> List[int]:
                try:
                    # Remove brackets and split by commas
                    nums = [int(x.strip()) for x in r.strip("[]").split(",")]
                    return nums
                except:
                    return []

            parsed_responses = [parse_list(r) for r in responses]
            return [
                1.0 if r == sorted(a) else 0.0 for r, a in zip(parsed_responses, answer)
            ]

        def format_reward_func(completions, **kwargs) -> List[float]:
            """
            Reward function that checks for proper XML formatting

            Must have think and answer fields
            """

            def check_format(text: str) -> float:
                try:
                    parsed = self.parser.parse(text)
                    return (
                        1.0
                        if hasattr(parsed, "think") and hasattr(parsed, "answer")
                        else 0.0
                    )
                except:
                    return 0.0

            return [check_format(c[0]["content"]) for c in completions]

        return [correctness_reward_func, format_reward_func]
