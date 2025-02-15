import os

import trl
from datasets import load_dataset
from message_decoding_reward_fns import answer_reward_func, format_reward_func
from prepare_inputs import tokenize_and_inject_images
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import GRPOConfig, ModelConfig
from trl.trainer import QwenGRPOTrainer

print(trl.__file__)


os.environ["WANDB_ENTITY"] = "groundlightai"
os.environ["WANDB_PROJECT"] = "message-decoding-words"

full_dataset = load_dataset("sunildkumar/message-decoding-words-r1")["train"]
full_dataset = full_dataset.shuffle(seed=42)

splits = full_dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = splits["train"]
eval_dataset = splits["test"]


gradient_checkpointing = False


model_config = ModelConfig(
    model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype="bfloat16",
    use_peft=False,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_config.model_name_or_path,
    torch_dtype=model_config.torch_dtype,
    use_cache=False,
    # flash attention not supported on our trainer yet
    # attn_implementation="flash_attention_2",
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
    output_dir="vlm-r1-message-decoding-words",
    learning_rate=1e-6,
    adam_beta2=0.98,
    lr_scheduler_type="cosine",
    warmup_steps=0,
    logging_steps=1,
    save_steps=20,
    # ckpts are 51 gb each!!
    save_total_limit=50,
    num_train_epochs=3,
    per_device_train_batch_size=5,
    num_generations=15,
    gradient_accumulation_steps=4,
    gradient_checkpointing=gradient_checkpointing,
    bf16=True,
    # GRPO specific parameters
    max_prompt_length=1024,
    max_completion_length=512,
    beta=0.001,
    temperature=1.0,
    sync_ref_model=True,
    ref_model_sync_steps=64,
    eval_strategy="no",
    log_completions=True,
    use_vllm=False,
    report_to="wandb",
)


trainer = QwenGRPOTrainer(
    model=model,
    reward_funcs=[
        format_reward_func,
        answer_reward_func,
    ],
    processing_class=processor,
    args=training_args,
    tokenize_and_inject_images=tokenize_and_inject_images,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
