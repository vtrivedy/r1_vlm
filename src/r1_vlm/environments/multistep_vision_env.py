from abc import abstractmethod
from typing import Any

from datasets import Dataset
from simple_vision_env import prepare_inputs_for_env
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.envs.environment import Environment
from vllm import LLM, SamplingParams  # type: ignore


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
    
    def prepare_data(self, *, inputs, processing_class):
        """
        prepares the data to be used for forward pass with VLLM and logprobs calculations with hf
        """
        conversations, texts, batch, vllm_inputs = prepare_inputs_for_env(
            inputs=inputs, processing_class=processing_class
        )

        return conversations, texts, batch, vllm_inputs
    
    def generate(self, conversations, vlm_inputs, vlm: LLM, sampling_params: SamplingParams) ->  list[list[dict[str, Any]]]:
        custom_sp = sampling_params.clone()
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)

        # initialize state variables
        all_completed = False
        states = [{
            "messages": conversation,
            # the number of messages in the conversation before generation
            "prompt_messages": len(conversation),
            "prompt_ids": [],
            "completed": False,
            "completion_ids": [],
            "completion_mask": []
        } for conversation in conversations]

        # main loop
        while not all_completed:
            states = self.step(states, vlm, custom_sp)
            all_completed = all(state["completed"] for state in states)

        completion_messages = [s["messages"][s["prompt_messages"]:] for s in states]
        completion_ids = [s["completion_ids"] for s in states]
        completion_mask = [s["completion_mask"] for s in states]
        output = {
            "ids": completion_ids,
            "messages": completion_messages,
            "mask": completion_mask
        }
        return output

    
    def step(self):
        pass 
