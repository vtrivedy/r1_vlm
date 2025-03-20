from typing import Any, Dict, List

from datasets import Dataset
from transformers import AutoProcessor
from trl.trainer.grpo_trainer import RewardFunc

from .multistep_vision_env import MultistepVisionEnv

ENV_MESSAGE = {'role': 'user', 'content': [{"text": "Are you sure? Please double check your work.", "type": "text"}]}

class DoubleCheckVisionEnv(MultistepVisionEnv):
    def __init__(self, 
                 sampling_args: dict[str, Any] = {},
                 mask_env_response: bool = True,
                 max_workers: int = 10,
                 processing_class: AutoProcessor = None):
        
        """
        A simple multistep vision environment that asks the model if it's sure of its answer.
        
        Args:
            dataset_name: The name of the dataset to use. To be used in `get_dataset`.
            sampling_args: args to be applied as updates to the SamplingParams object provided to .generate
            mask_env_response: If True, the environment response will be masked when computing loss. Essentially, the environment response will not be considered part of the completion. 
            max_workers: The max number of workers used for the `update_state` step.
        """
        
        super().__init__(sampling_args=sampling_args, mask_env_response=mask_env_response, max_workers=max_workers, processing_class=processing_class)


    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        raise NotImplementedError("DoubleCheckVisionEnv requires a rubric for your task. Expected to be implemented by subclass.")
    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        raise NotImplementedError("DoubleCheckVisionEnv requires a dataset for your task. Expected to be implemented by subclass.")
    
    def is_completed(self, messages: List[Dict[str, Any]], **kwargs: Any) -> bool:
        """
        Checks if the conversation is completed. In this case, it's completed once 
        the user asks "Please restate your answer again." and then model responds.
        """
        # Need to check if we've gone through the pattern: ENV_MESSAGE followed by two assistant responses - the bootstrap and the actual response. 
        return (len(messages) > 2 and 
                messages[-3] == ENV_MESSAGE and 
                messages[-2]["role"] == "assistant")
        
    def env_response(self, messages: List[Dict[str, Any]], **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Returns the environment response and a bootstrap assistant message.
        """
        return [
            ENV_MESSAGE,
            {'role': 'assistant', 'content': [{"text": "<think> ", "type": "text"}]}
        ]