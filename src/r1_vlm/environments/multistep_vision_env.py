from abc import abstractmethod
from typing import Any

from datasets import Dataset
from transformers import AutoProcessor
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.envs.environment import Environment
from vllm import LLM, SamplingParams  # type: ignore

from r1_vlm.environments.simple_vision_env import prepare_inputs_for_env


class MultistepVisionEnv(Environment):
    def __init__(self,
                 #system_prompt:str = None,
                 processing_class: AutoProcessor,
                 sampling_args: dict[str, Any] = {},
                 mask_env_response: bool = True,
                 max_workers: int = 10,
 
                 **kwargs):
        '''
        Sampling args: Args will be applied as updates to the SamplingParams object provided to .generate
        mask_env_response: If True, the environment response will be masked when computing loss. Essentially, the environment response will not be considered part of the completion. 
        max_workers: The max number of workers used for the `update_state` step.
        processing_class: The processing class to use to process the inputs. This is a VLM processor object from the transformers library.
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
        self.processing_class = processing_class
        
    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        pass
    
    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
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
    
    def step(self, states: list[dict[str, Any]], vlm: LLM, sampling_params: SamplingParams) -> list[dict[str, Any]]:
        
        # indicies we need to step
        live_indices = [i for i, s in enumerate(states) if not s["completed"]]
        
        # list of conversations to step
        messages_to_step = [states[i]["messages"] for i in live_indices]
        
        # convert conversations to vlm inputs
        vlm_inputs = []
        for conversation in messages_to_step:
            conversations, texts, batch, vllm_inputs = self.prepare_data(conversation, self.processing_class)
            vlm_inputs.append(vllm_inputs)
        
        vlm_responses = vlm.chat(vlm_inputs, sampling_params=sampling_params, use_tqdm=False)
        
        def update_state(j, vlm_response):
            
            # get the state prior to the step
            state = states[j].copy()
            
            # populate the prompt ids if we are on the first step
            if len(state["prompt_ids"]) == 0:
                state["prompt_ids"] = vlm_response.prompt_token_ids
            
            # update the conversation with the model's response
            state["messages"].append({"role": "assistant", "content": vlm_response.outputs[0].text})
            
            # get token lengths of env response and new completion
            total_prev_len = len(state["prompt_ids"]) + len(state["completion_ids"])
            env_response_len  = len(list(vlm_response.prompt_token_ids)) - total_prev_len # type: ignore
            new_completion_len = len(vlm_response.outputs[0].token_ids)
            
            # update completion masks
            state["completion_mask"].extend([self.env_mask] * env_response_len)
            state["completion_mask"].extend([1] * new_completion_len)
            
            # update completion ids
            state["completion_ids"] = list(vlm_response.prompt_token_ids) # type: ignore
            state["completion_ids"].extend(list(vlm_response.outputs[0].token_ids))
            state["completion_ids"] = state["completion_ids"][len(state["prompt_ids"]):]
            
            # if we are done, we truncate if necessary
            if self.is_completed(state["messages"]) or len(state["completion_ids"]) > sampling_params.max_tokens: # type: ignore
                state["completed"] = True
                state["completion_ids"] = state["completion_ids"][:sampling_params.max_tokens]
                state["completion_mask"] = state["completion_mask"][:len(state["completion_ids"])]
                
            # otherwise, we get the env response
            else:
                state["messages"].append(self.env_response(state["messages"]))
            
            
            if not len(state["completion_mask"]) == len(state["completion_ids"]):
                print(state["messages"])
                print(state["completion_mask"])
                print(state["completion_ids"])
                raise ValueError(f"Completion mask and completion ids are not the same length for state {j}")
            
            return j, state
            
            
        
        
    
    
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
