from typing import Callable

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoProcessor
from trl.trainer.grpo_trainer import RewardFunc

from r1_vlm.datasets.utils import preprocess_r1_dataset
from r1_vlm.environments.tool_vision_env import ToolVisionEnv
from r1_vlm.tools.digits_answer_tool import get_answer


class DigitsToolUseBaselineEnv(ToolVisionEnv):
    '''
    This env tests the ability of the model to use tools, in this case a tool that "cheats" by getting correct answer directly from the dataset.
    This is meant as a sanity check to ensure that the tool calling works properly, not a proper experiment.
    '''
    
    def __init__(
        self,
        processing_class: AutoProcessor,
        dataset_name: str = "sunildkumar/digit-recognition-tool-use-baseline-r1",
        # tool that directly gets the answer from the dataset
        tools: list[Callable] = [get_answer],
        max_steps: int = 10,
        ):
        
        super().__init__(
            tools=tools,
            processing_class=processing_class,
            max_steps=max_steps,
        )
        
        self.dataset_name = dataset_name
    
    def get_dataset(self) -> Dataset:
        
        dataset = load_dataset(self.dataset_name)

        # select all three splits
        digits_1 = dataset["digits_1"]
        digits_2 = dataset["digits_2"]
        digits_3 = dataset["digits_3"]

        # concatenate the three splits
        dataset = concatenate_datasets([digits_1, digits_2, digits_3])
        
        # handle system prompt injection
        dataset = self.inject_system_prompt(dataset)

        # handle image injection
        dataset = preprocess_r1_dataset(dataset)
        
        return dataset
    
    def get_rubric(self) -> list[RewardFunc]:
        
        
            
        
        def placeholder_reward_func(prompts, completions, completions_messages, **kwargs) -> list[float]:
            '''
            Placeholder reward function that returns 1.0 for all completions. Currently testing things upstream. 
            '''
            
            return [1.0] * len(prompts)

        return [placeholder_reward_func]
    