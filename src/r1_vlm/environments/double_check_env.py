from typing import Any, Dict, List

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from .multistep_vision_env import MultistepVisionEnv


class DoubleCheckVisionEnv(MultistepVisionEnv):
    def __init__(self, 
                 dataset_name: str,
                 sampling_args: dict[str, Any] = {},
                 mask_env_response: bool = True,
                 max_workers: int = 10):
        """
        A simple multistep vision environment that asks the model if it's sure of its answer.
        
        Args:
            dataset_name: The name of the dataset to use. To be used in `get_dataset`.
            sampling_args: args to be applied as updates to the SamplingParams object provided to .generate
            mask_env_response: If True, the environment response will be masked when computing loss. Essentially, the environment response will not be considered part of the completion. 
            max_workers: The max number of workers used for the `update_state` step.
        """

        super().__init__(sampling_args=sampling_args, mask_env_response=mask_env_response, max_workers=max_workers)
        
        self.dataset_name = dataset_name

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        raise NotImplementedError("DoubleCheckVisionEnv requires a rubric for your task.")
    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        raise NotImplementedError("DoubleCheckVisionEnv requires a dataset for your task.")
    
    def is_completed(self, messages: List[Dict[str, Any]], **kwargs: Any) -> bool:
        """
        Checks if the conversation is completed. In this case, it's completed once the user asks "Are you sure?" and then model responds. 
        """
       
    
    def env_response(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        """
        Returns the environment response, which is simply asking the model
        if it's sure of its answer.
        """
        # For a text-only response
        return {'role': 'user', 'content': 'Are you sure?'}
