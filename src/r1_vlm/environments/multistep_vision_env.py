from abc import abstractmethod
from typing import Any

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.envs.environment import Environment


class MultistepVisionEnv(Environment):
    def __init__(self,
                 #system_prompt:str = None,
                 sampling_args: dict[str, Any] = {},
                 mask_env_response: bool = True,
                 max_workers: int = 10,
                 **kwargs):
        '''
        Sampling args: Args will be applied as updates to the SamplingParams object provided to .generate
        mask_env_response: If True, the environment response will be masked when computing loss. Essentially, the environment response will not be considered part of the completion. 
        max_workers: The max number of workers used for the `update_state` step.
        '''
        super().__init__(**kwargs)
        self.sampling_args = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "n": 1
        }
        self.sampling_args.update(sampling_args)
        self.env_mask = 0 if mask_env_response else 1
        self.max_workers = max_workers
    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    @abstractmethod
    def get_rubric(self, **kwargs: Any) -> list[RewardFunc]:
        pass
    
    @abstractmethod
    def is_completed(self, messages: list[dict[str, str]], **kwargs: Any) -> bool:
        pass
   
    @abstractmethod
    def env_response(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, str]:
        pass
   