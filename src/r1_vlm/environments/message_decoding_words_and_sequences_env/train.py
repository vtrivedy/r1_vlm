import os

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import GRPOConfig, ModelConfig
from trl.trainer.qwen_grpo_trainer import QwenGRPOTrainer

from r1_vlm.environments.message_decoding_words_and_sequences_env.message_decoding_sequences_env import (
    MessageDecodingEnv,
)

os.environ["WANDB_ENTITY"] = "groundlightai"
os.environ["WANDB_PROJECT"] = "message-decoding-words-and-sequences"

vf_env = MessageDecodingEnv()
train_dataset, test_dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()

# Flag that determines if gradient checkpointing is used. If it is, we need to set use_cache to False.
gradient_checkpointing = False

checkpoint = None
model_config = ModelConfig(
    model_name_or_path=checkpoint if checkpoint else "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype="bfloat16",
    use_peft=False,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_config.model_name_or_path,
    torch_dtype=model_config.torch_dtype,
    use_cache=False,
)


# use cache if not gradient checkpointing
if gradient_checkpointing:
    model.config.use_cache = False
elif not gradient_checkpointing:
    model.config.use_cache = True
else:
    raise ValueError("Invalid gradient checkpointing value")


processor = AutoProcessor.from_pretrained(
    model_config.model_name_or_path, padding_side="left"
)


training_args = GRPOConfig(
    model_init_kwargs=model_config,
    output_dir="vlm-r1-message-decoding-words-and-sequences",
    learning_rate=1e-6,
    adam_beta2=0.98,
    lr_scheduler_type="cosine",
    warmup_steps=0,
    logging_steps=1,
    save_steps=50,
    save_total_limit=50,
    num_train_epochs=1,
    per_device_train_batch_size=3,
    num_generations=9,
    gradient_accumulation_steps=5,
    gradient_checkpointing=gradient_checkpointing,
    bf16=True,
    max_prompt_length=None,
    max_completion_length=400,
    # in order: chars, correctness, format, correctness_intermediate
    reward_weights=[1.0, 1.0, 1.0, 1.0],
    beta=0.001,
    temperature=1.0,
    sync_ref_model=True,
    ref_model_sync_steps=64,
    eval_strategy="no",
    log_completions=True,
    use_vllm=True,
    vllm_gpu_memory_utilization=0.5,
    vllm_max_model_len=33000,
    report_to="wandb",
    vllm_device="cuda:3",
)

trainer = QwenGRPOTrainer(
    model=model,
    processing_class=processor,
    reward_funcs=rubric,
    args=training_args,
    train_dataset=train_dataset,
    env=vf_env,
    # False as we are training on a curriculum of examples
    shuffle_dataset=False,
)

trainer.train()

# CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --config_file src/r1_vlm/deepspeed_configs/multi_gpu_3only.yaml src/r1_vlm/environments/message_decoding_words_and_sequences_env/train.py  2>&1 | tee message_decoding_logs_$(date +%Y%m%d_%H%M%S).log
