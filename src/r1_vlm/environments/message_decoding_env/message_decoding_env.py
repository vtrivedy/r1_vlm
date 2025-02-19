
import re
from typing import Any, List

from datasets import Dataset, load_dataset
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.parsers import XMLParser

from r1_vlm.environments.simple_vision_env import SimpleVisionEnv


class MessageDecodingEnv(SimpleVisionEnv):
    def __init__(
        self,
        dataset: str = "sunildkumar/message-decoding-words-r1",
        system_prompt: str = "",
        **kwargs,  # passed to the superclass
    ):
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.dataset_name = dataset
        self.parser = XMLParser(fields=["think", "answer"])
    
    
    def get_dataset(self) -> Dataset:
        dataset = load_dataset(self.dataset_name)["train"]
        return dataset
    
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        def correctness_reward_func(completions, **kwargs) -> List[float]:
            pass
        
        def format_reward_func(completions, **kwargs) -> List[float]:
            """
            Reward function that checks for proper XML formatting

            Must have think and answer fields
            """

            def check_format(text: str) -> float:
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
                except Exception as e:
                    print(f"Error in format_reward_func: {e}")
                    return 0.0

            rewards = [check_format(c[0]["content"]) for c in completions]
            return rewards
        
        return [correctness_reward_func, format_reward_func]
        