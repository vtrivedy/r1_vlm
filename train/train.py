import trl
from datasets import load_dataset
from peft import LoraConfig
from reward_fns import answer_reward_func, format_reward_func
from trl import GRPOConfig, GRPOTrainer, ModelConfig

print(trl.__file__)


dataset = load_dataset("sunildkumar/coco-computation-r1")

model_config = ModelConfig(
    model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype="bfloat16",
    use_peft=True,
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)


# Hyperparameters
training_args = GRPOConfig(
    output_dir="vlm-r1-aha-moment",
    learning_rate=5e-7,
    lr_scheduler_type="cosine",
    logging_steps=10,
    max_steps=100,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    # GRPO specific parameters
    # TODO: Make sure these are right
    max_prompt_length=256,
    max_completion_length=1024,  # max length of the generated output for our solution
    num_generations=2,
    beta=0.001,
)


trainer = GRPOTrainer(
    model=model_config.model_name_or_path,
    reward_funcs=[format_reward_func, answer_reward_func],
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
)
