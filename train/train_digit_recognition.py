import trl
from datasets import load_dataset
from peft import LoraConfig
from prepare_inputs import tokenize_and_inject_images
from digit_recognition_reward_fns import format_reward_func, answer_reward_func
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from trl import GRPOConfig, ModelConfig
from trl.trainer import QwenGRPOTrainer
from datasets import concatenate_datasets
from torch.utils.data import SequentialSampler

print(trl.__file__)


dataset = load_dataset("sunildkumar/digit-recognition-r1")
digits_1 = dataset["digits_1"]
digits_2 = dataset["digits_2"]
digits_3 = dataset["digits_3"]

# Split each dataset
split_1 = digits_1.train_test_split(test_size=0.1, seed=42)
split_2 = digits_2.train_test_split(test_size=0.1, seed=42)
split_3 = digits_3.train_test_split(test_size=0.1, seed=42)

digits_1_train, digits_1_eval = split_1["train"], split_1["test"]
digits_2_train, digits_2_eval = split_2["train"], split_2["test"]
digits_3_train, digits_3_eval = split_3["train"], split_3["test"]

# Repeating each section of the curriculum multiple times during training.
NUM_PASSES_PER_SECTION = 3

train_dataset = concatenate_datasets(
    [digits_1_train] * NUM_PASSES_PER_SECTION +
    [digits_2_train] * NUM_PASSES_PER_SECTION +
    [digits_3_train] * NUM_PASSES_PER_SECTION
)
eval_dataset = concatenate_datasets([digits_1_eval, digits_2_eval, digits_3_eval])

print(f"There are {len(train_dataset)} training examples and {len(eval_dataset)} eval examples.")

# load the model
model_config = ModelConfig(
    model_name_or_path="Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="bfloat16",
    use_peft=True,
)
# Load the model using checkpoint weights. Or set to None to use the model from the model hub.
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

# set the minimum pixels and maximum pixels to this value == 529 image tokens per image. 
pixels = 224 * 224
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", padding_side="left", min_pixels=pixels, max_pixels=pixels
)

# Hyperparameters
training_args = GRPOConfig(
    output_dir="vlm-r1-digit-recognition",
    learning_rate=1e-6,
    lr_scheduler_type="cosine",
    warmup_steps=0,
    logging_steps=1,
    save_steps=20,
    # ckpts are 51 gb each!!
    save_total_limit= 50,
    num_train_epochs=1,
    # I've heard I shouldn't increase this due to a bug. 
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    bf16=True,
    # GRPO specific parameters
    # TOOD: Make sure these are right
    max_prompt_length=1024,
    max_completion_length=512,  # max length of the generated output for our solution
    num_generations=8,
    beta=0.001,
    use_vllm=False,
    report_to="wandb",
    # R1-V suggestion
    temperature=1.0, 
)

# turn off shuffling so the model sees the data in increasing difficulty order
class NoShuffleQwenGRPOTrainer(QwenGRPOTrainer):
    def _get_train_sampler(self):
        return SequentialSampler(self.train_dataset)
    

trainer = NoShuffleQwenGRPOTrainer(
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
    #    peft_config=peft_config,
)


trainer.train()
