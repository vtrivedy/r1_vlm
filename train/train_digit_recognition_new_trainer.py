import os

import trl
from datasets import load_dataset
from digit_recognition_reward_fns import answer_reward_func, format_reward_func
from prepare_inputs import tokenize_and_inject_images
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from trl import GRPOConfig, ModelConfig
from trl.trainer.qwen_grpo_trainer import QwenGRPOTrainer

os.environ["WANDB_ENTITY"] = "groundlightai"
os.environ["WANDB_PROJECT"] = "digit-recognition-new-trainer"

print(trl.__file__)


dataset = load_dataset("sunildkumar/digit-recognition-r1")
digits_1 = dataset["digits_1"].shuffle(seed=42)
split_1 = digits_1.train_test_split(test_size=0.1, seed=42)
train_dataset, eval_dataset = split_1["train"], split_1["test"]

print(
    f"There are {len(train_dataset)} training examples and {len(eval_dataset)} eval examples."
)


model_config = ModelConfig(
    model_name_or_path="Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="bfloat16",
    use_peft=True,
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_config.model_name_or_path,
    torch_dtype=model_config.torch_dtype,
    # has to be set to false for gradient checkpointing to work
    use_cache=False,
    # faster generation, R1-V suggestion
    attn_implementation="flash_attention_2",
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    padding_side="left",
)

training_args = GRPOConfig(
    output_dir="vlm-r1-digit-recognition-new-trainer",
    learning_rate=1e-6,
    lr_scheduler_type="cosine",
    warmup_steps=0,
    logging_steps=1,
    save_steps=20,
    # ckpts are 51 gb each!!
    save_total_limit=50,
    num_train_epochs=1,
    # represents the number of generations per device
    per_device_train_batch_size=8,
    # number of generations total
    num_generations=8,
    gradient_accumulation_steps=4,
    # Turning this on...
    gradient_checkpointing=True,
    bf16=True,
    # GRPO specific parameters
    max_prompt_length=1024,
    max_completion_length=512,  # max length of the generated output for our solution
    beta=0.001,
    use_vllm=False,
    report_to="wandb",
    # R1-V suggestion
    temperature=1.0,
    # sync the reference model every so often
    sync_ref_model=True,
    # how often to merge the reference model with the train model, default is 64
    ref_model_sync_steps=64,
    eval_strategy="no",
    log_completions=True,
)

trainer = QwenGRPOTrainer(
    model=model,
    processing_class=processor,
    reward_funcs=[
        format_reward_func,
        answer_reward_func,
    ],
    tokenize_and_inject_images=tokenize_and_inject_images,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
# TODOS:
# [x] - the section per_device_train/eval_batch_size * num processes can be divided by the number of generations seems a bit worrying. It might limit num_generations to 4?
# [x] - inject images into the input at the top of _prepare_inputs
# [] - add logging metrics suggested by tyler
# [] -

# GOALS:
# [In progress] - Branch off of TRL main (again) with the new version of their trainer. Get 2B training off of it on digits on a single GPU.
# [TODO] - Get 2B training off of it on digits on multiple GPUs - either zero2 or zero3.
# [TODO] - Get 7B model training on digits task - maybe zero3 if necessary? Prove we can solve it to prove the code works.
# [TODO] - Get VLLM working. It should make the generation step a whole lot faster and thus training faster.
# [TODO] - Try the decoding task again.

# NOTES:
# The format of the data passed to the reward function has changed. It now includes the "bootstrap" prompt as well. 
