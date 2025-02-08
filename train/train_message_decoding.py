import trl
from datasets import load_dataset
from message_decoding_reward_fns import format_reward_func, soft_edit_distance_reward
from peft import LoraConfig
from prepare_inputs import tokenize_and_inject_images
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from trl import GRPOConfig, ModelConfig
from trl.trainer import QwenGRPOTrainer

print(trl.__file__)


dataset = load_dataset("sunildkumar/message-decoding-r1")
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

# load the model
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

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)


# NOTE: Not setting min/max pixels here unlike in digit recognition. All the images are 300x300 and should get tokenized at that resolution.
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", padding_side="left"
)

training_args = GRPOConfig(
    output_dir="vlm-r1-message-decoding",
    learning_rate=1e-6,
    lr_scheduler_type="cosine",
    warmup_steps=0,
    logging_steps=1,
    save_steps=20,
    # ckpts are 51 gb each!!
    save_total_limit=50,
    num_train_epochs=1,
    # I've heard I shouldn't increase this due to a bug.
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    bf16=True,
    # GRPO specific parameters
    max_prompt_length=1024,
    max_completion_length=512,
    num_generations=8,
    beta=0.001,
    use_vllm=False,
    report_to="wandb",
    # R1-V suggestion
    temperature=1.0,
)

trainer = QwenGRPOTrainer(
    model=model,
    reward_funcs=[
        format_reward_func,
        soft_edit_distance_reward,
    ],
    processing_class=processor,
    args=training_args,
    tokenize_and_inject_images=tokenize_and_inject_images,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
