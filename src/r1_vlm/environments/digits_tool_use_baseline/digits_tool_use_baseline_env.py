import re
from statistics import mean
from typing import Any, Callable

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
    
    def get_assistant_messages(self, conversation: list[dict[str, Any]]) -> list[str]:
        '''
        Returns the assistant messages from the completion messages as a list of strings.
        '''
        assistant_messages = [message["content"][0]["text"] for message in conversation if message["role"] == "assistant"]
        return assistant_messages
    
    def get_rubric(self) -> list[RewardFunc]:
        
        
        def format_reward_func(prompts, completions, completions_messages, **kwargs) -> list[float]:
            '''
            Returns the average compliance over all model messages in the completion.
            
            prompts: list of messages that make up the original prompt
            completions: list of completion strings (not used, but required by the interface)
            completions_messages: list of messages in the completion
            '''
            
            merged_completion_conversations = self.preprocess_messages(prompts_messages=prompts, completions_messages=completions_messages)
            
            rewards = []
            for conversation in merged_completion_conversations:
                assistant_messages = self.get_assistant_messages(conversation)
                
                format_correct = [check_format(message) for message in assistant_messages]
                format_correct = mean(format_correct)
                rewards.append(format_correct)
                
            return rewards
        
        
        def check_format(text: str) -> float:
            '''
            Checks if the format is correct for a single message.
            '''
            # Find and start from the first <think> tag (removes the bootstrap prompt, if it exists)
            think_start = text.find("<think>")
            if think_start != -1:
                text = text[think_start:]

            try:
                # Check if the format is correct
                answer_regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
                tool_regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<tool>([\s\S]*?)<\/tool>$"

                answer_match = re.search(answer_regex, text, re.DOTALL)
                tool_match = re.search(tool_regex, text, re.DOTALL)

                if (answer_match is not None and len(answer_match.groups()) == 2) or \
                   (tool_match is not None and len(tool_match.groups()) == 2):
                    return 1.0
                return 0.0
            except Exception as e:
                print(f"Error in check_format: {e}")
                return 0.0
                

    

        return [format_reward_func]
    