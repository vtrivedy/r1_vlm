import json
from typing import Any, Dict, List, Sequence, Union

from verifiers import SimpleEnv
from vllm import LLM, SamplingParams  # type: ignore


class SimpleVisionEnv(SimpleEnv):
    def generate(
        self,
        prompts: List[List[Dict[str, Any]]],
        llm: LLM,
        sampling_params: SamplingParams,
        output_type: str = "ids",
        **kwargs: Any,
    ) -> Union[List[Sequence[int]], List[str], List[List[Dict[str, Any]]]]:
        print("You've reached generate")
        import ipdb

        ipdb.set_trace()

        custom_sp = sampling_params.clone()
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)

        states = [
            {"messages": m, "prompt_ids": [], "completion_ids": []} for m in prompts
        ]

        # get completions
        completions = llm.chat(prompts, sampling_params=custom_sp, use_tqdm=False)  # type: ignore
        for i, completion in enumerate(completions):
            states[i]["messages"].append(
                {"role": "assistant", "content": completion.outputs[0].text}
            )
            states[i]["prompt_ids"] = list(completion.prompt_token_ids)
            states[i]["completion_ids"] = list(completion.outputs[0].token_ids)

        self.logger.debug(
            f"Prompt 0 IDs: {states[0]['prompt_ids']} \nlen: {len(states[0]['prompt_ids'])}"
        )
        self.logger.debug(
            f"Completion 0 IDs: {states[0]['completion_ids']} \nlen: {len(states[0]['completion_ids'])}"
        )
        self.logger.info(
            "Prompt 0:\n"
            + json.dumps(states[0]["messages"][:-1], indent=4)
            + "\n\nCompletion 0:\n"
            + json.dumps(states[0]["messages"][-1], indent=4)
        )

        if output_type == "ids":
            return [states[i]["completion_ids"] for i in range(len(states))]
        elif output_type == "messages":
            return [states[i]["messages"] for i in range(len(states))]
        else:
            raise ValueError(f"Invalid output type: {output_type}")
