import re
from statistics import mean
from typing import Any, List

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoProcessor
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.parsers import XMLParser

from r1_vlm.datasets.utils import preprocess_r1_dataset
from r1_vlm.environments.double_check_env import DoubleCheckVisionEnv


class DigitsDoubleCheckEnv(DoubleCheckVisionEnv):
    def __init__(
        self,
        dataset_name: str = "sunildkumar/digits-doublecheck-r1",
        mask_env_response: bool = True,
        processing_class: AutoProcessor = None,
    ):
        super().__init__(
            mask_env_response=mask_env_response,
            processing_class=processing_class,
        )
        self.dataset_name = dataset_name
        self.parser = XMLParser(fields=["think", "answer"])
        
    def get_dataset(self) -> Dataset:
        dataset = load_dataset(self.dataset_name)

        # select all three splits
        digits_1 = dataset["digits_1"]
        digits_2 = dataset["digits_2"]
        digits_3 = dataset["digits_3"]

        # concatenate the three splits
        dataset = concatenate_datasets([digits_1, digits_2, digits_3])

        # handle image injection
        dataset = preprocess_r1_dataset(dataset)

        return dataset
    
    def get_rubric(self, **kwargs: Any) -> list[RewardFunc]:
        '''
        The correctness rewards only apply to the last message in the completion, not the intermediate completion message from the model.
        '''
        
        def _recognition_correctness_reward_func(completions, **kwargs) -> list[float]:
                """Reward function for recognition task"""

                # verify that the task is recognition for all instances. If it isn't assumptions downstream fail
                tasks = kwargs["task"]
                if not all(t == "recognition" for t in tasks):
                    raise ValueError("All instances must be recognition tasks")

                # the answer in this case is the label, which is a list of list of ints, e.g. [[0, 5], [0, 5], [2, 8], [2, 8]]
                answers = kwargs["label"]

                # parse the last message in each completion (completions are conversations) for data between <answer> and </answer> tags
                # only selects the data between the first answer tags in the message if there are multiple valid answer sets in it.
                # TODO: verify this is actually only looking at the last message in the completion. 
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
                    1.0 if r == sorted(a) else 0.0
                    for r, a in zip(parsed_responses, answers)
                ]
    
        def _addition_correctness_reward_func(completions, **kwargs) -> List[float]:
            """
            Reward function for addition task
            """
            
            # verify that the task is recognition for all instances. If it isn't assumptions downstream fail
            tasks = kwargs["task"]
            if not all(t == "addition" for t in tasks):
                raise ValueError("All instances must be addition tasks")
            
            # the answer is a list of ints, e.g. [1, 2, 3, 4]
            answers = kwargs["total"]
            
            # parse the last message in each completion (completions are conversations) for data between <answer> and </answer> tags
            # only selects the data between the first answer tags in the message if there are multiple valid answer sets in it.
            # TODO: verify this is actually only looking at the last message in the completion. 
            responses = [self.parser.parse(c[0]["content"]).answer for c in completions]

            def check_answer(response, answer):
                try:
                    response = int(response)
                    answer = int(answer)

                    if response == answer:
                        return 1.0
                    else:
                        return 0.0

                except Exception as e:
                    print(f"Error in _addition_correctness_reward_func: {e}")
                    return 0.0

            rewards = [check_answer(r, a) for r, a in zip(responses, answers)]
            return rewards

        def correctness_reward_func(completions, **kwargs) -> List[float]:
            """Reward function that checks if the predicted digits match the true labels"""

            tasks = kwargs["task"]
            if not (task == tasks[0] for task in tasks):
                raise ValueError(
                    "All tasks must be same type, invalidates an assumption of the code"
                )

            task = tasks[0]

            if task == "recognition":
                return _recognition_correctness_reward_func(completions, **kwargs)
            elif task == "addition":
                return _addition_correctness_reward_func(completions, **kwargs)
            else:
                raise ValueError(f"Invalid task: {task}")
            
        
        def format_reward_func(prompts, completions, completions_messages, **kwargs) -> List[float]:
            '''
            Returns the average compliance over all model messages in the completion.
            '''
            print("PROMPTS[0]")
            print(prompts[0])
            print("COMPLETIONS[0]")
            print(completions[0])
            print("COMPLETIONS_MESSAGES[0]")
            print(completions_messages[0])
            
            raise ValueError("Not implemented")
            completion_rewards = []
            for completion in completions:
                # select all messages from the model
                model_messages = [m for m in completion if m["role"] == "assistant"]
                
                assert len(model_messages) == 2, f"There should be two model messages in the completion, before and after the environment response. Found {len(model_messages)}."
                
                completion_rewards = [check_format(m["content"]) for m in model_messages]
                completion_reward = mean(completion_rewards)
                completion_rewards.append(completion_reward)
                
            return completion_rewards
                
            
        def check_format(text: str) -> float:
            '''
            Checks if the format is correct for a single message.
            '''
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
        
        return [correctness_reward_func, format_reward_func]
        