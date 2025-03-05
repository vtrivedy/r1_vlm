from unittest.mock import patch

from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import ModelConfig
from vllm import LLM, SamplingParams

from r1_vlm.environments.message_decoding_words_and_sequences_env.message_decoding_sequences_env import (
    MessageDecodingEnv,
)

vf_env = MessageDecodingEnv()
train_dataset, test_dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()


checkpoint = "/millcreek/home/sunil/r1_vlm/vlm-r1-message-decoding-words-and-sequences_official_demo/checkpoint-1850"


model_config = ModelConfig(
    model_name_or_path=checkpoint,
    torch_dtype="bfloat16",
    use_peft=False,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_config.model_name_or_path,
    torch_dtype=model_config.torch_dtype,
    use_cache=False,
)
model.eval()

processor = AutoProcessor.from_pretrained(
    model_config.model_name_or_path, padding_side="left"
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=512,
)


world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
profiling_patch = patch(
    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
    return_value=None,
)
with world_size_patch, profiling_patch:
    vlm = LLM(
        model=model.name_or_path,
        device="cuda:0",
        gpu_memory_utilization=1.0,
        dtype="bfloat16",
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 1, "video": 0},
    )


# 50 worked
batch_size = 100
# list of lists of examples
batches = []
for example in test_dataset:
    if len(batches) == 0:
        batches.append([example])
    elif len(batches[-1]) < batch_size:
        batches[-1].append(example)
    else:
        batches.append([example])


def extract_answer(generated_text):
    # capture string between <answer> and </answer>
    start = generated_text.find("<answer>") + len("<answer>")
    end = generated_text.find("</answer>")
    answer = generated_text[start:end]
    answer = answer.strip()
    return answer


results = []


for batch in tqdm(batches):
    conversations, texts, processed_batch, vllm_inputs = vf_env.prepare_data(
        inputs=batch, processing_class=processor
    )

    # use env + vllm instance in trainer for inference
    completion_ids = vf_env.generate(
        conversations=conversations,
        vlm_inputs=vllm_inputs,
        vlm=vlm,
        sampling_params=sampling_params,
    )

    # decode the ids to text
    generated_texts = processor.batch_decode(
        completion_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    answers = [extract_answer(text) for text in generated_texts]
    gts = [example["decoded_message"] for example in batch]

    for answer, gt, example in zip(answers, gts, batch):
        result = {"answer": answer, "gt": gt, "task": example["task"]}
        results.append(result)


# compute accuracy on each task
task_counts = {}

for result in results:
    task = result["task"]
    if task not in task_counts:
        task_counts[task] = {"correct": 0, "total": 0}

    is_correct = result["answer"] == result["gt"]
    task_counts[task]["correct"] += int(is_correct)
    task_counts[task]["total"] += 1

# compute accuracy for each task
for task in task_counts.keys():
    results = task_counts[task]
    accuracy = results["correct"] / results["total"]
    print(f"{task}: {accuracy:.2f}, {results}")


# CUDA_VISIBLE_DEVICES=0 uv run src/r1_vlm/environments/message_decoding_words_and_sequences_env/eval.py
