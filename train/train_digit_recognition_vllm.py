import os

import torch
import trl
from curriculum_utils import (
    calculate_curriculum_steps,
    create_curriculum_lr_lambda,
    plot_lr_schedule,
)
from datasets import concatenate_datasets, load_dataset
from digit_recognition_reward_fns import answer_reward_func, format_reward_func
from prepare_inputs import tokenize_and_inject_images
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import GRPOConfig, ModelConfig
from trl.trainer.qwen_grpo_trainer import QwenGRPOTrainer

os.environ["WANDB_ENTITY"] = "groundlightai"
os.environ["WANDB_PROJECT"] = "digit-recognition-vllm"

print(trl.__file__)


dataset = load_dataset("sunildkumar/digit-recognition-r1")
digits_1 = dataset["digits_1"].shuffle(seed=42)
digits_2 = dataset["digits_2"].shuffle(seed=42)
digits_3 = dataset["digits_3"].shuffle(seed=42)

# Split each dataset
split_1 = digits_1.train_test_split(test_size=0.1, seed=42)
split_2 = digits_2.train_test_split(test_size=0.1, seed=42)
split_3 = digits_3.train_test_split(test_size=0.1, seed=42)

digits_1_train, digits_1_eval = split_1["train"], split_1["test"]
digits_2_train, digits_2_eval = split_2["train"], split_2["test"]
digits_3_train, digits_3_eval = split_3["train"], split_3["test"]


train_dataset = concatenate_datasets([digits_1_train, digits_2_train, digits_3_train])
eval_dataset = concatenate_datasets([digits_1_eval, digits_2_eval, digits_3_eval])

print(
    f"There are {len(train_dataset)} training examples and {len(eval_dataset)} eval examples."
)

# Flag that determines if gradient checkpointing is used. If it is, we need to set use_cache to False.
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
    output_dir="vlm-r1-digit-recognition-vllm",
    learning_rate=1e-6,
    adam_beta2=0.98,
    lr_scheduler_type="cosine",
    warmup_steps=0,
    logging_steps=1,
    save_steps=20,
    # ckpts are 51 gb each!!
    save_total_limit=50,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    num_generations=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=gradient_checkpointing,
    bf16=True,
    # GRPO specific parameters
    max_prompt_length=None,  # must be None for vllm
    max_completion_length=512,
    beta=0.001,
    temperature=1.0,
    sync_ref_model=True,
    ref_model_sync_steps=64,
    eval_strategy="no",
    log_completions=True,
    use_vllm=True,
    report_to="wandb",
)


# Setup curriculum learning
dataset_sizes = [len(digits_1_train), len(digits_2_train), len(digits_3_train)]
num_gpus = torch.cuda.device_count()
transition_steps = calculate_curriculum_steps(
    dataset_sizes,
    1,  # the per device batch size - manually setting this here because the value in args is not what it seems
    training_args.gradient_accumulation_steps,
    1,  # the number of GPUs isn't relevant here (I think??) because of the change to generation setup.
)
curriculum_lr_lambda = create_curriculum_lr_lambda(transition_steps)
plot_lr_schedule(transition_steps, curriculum_lr_lambda)


# Customize the trainer to use our curriculum learning lambda for the lr scheduler
class LRLambdaQwenGRPOTrainer(QwenGRPOTrainer):
    def __init__(self, *args, lr_lambda=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_lambda = lr_lambda

    def create_scheduler(self, num_training_steps, optimizer=None):
        optimizer = self.optimizer if optimizer is None else optimizer
        self.lr_scheduler = LambdaLR(optimizer, self.lr_lambda)
        return self.lr_scheduler


trainer = LRLambdaQwenGRPOTrainer(
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
    # Don't shuffle the dataset so we train on the curriculm in order - 1 digit, then 2 digits, then 3 digits.
    shuffle_dataset=False,
    # use our curriculum learning lambda for the lr scheduler
    lr_lambda=curriculum_lr_lambda,
)

trainer.train()


# TODOs:
# [] - Is training_args.vllm_dtype appropriate? Defaults to `auto`
# [] - Is training_args.vllm_max_model_len appropriate? Defaults to None
# [] - Verify all self.llm are changed to self.vlm

# NOTES:
# self._last_loaded_step is keeps track of the last global training step where weights were sent to vllm
# We only need to update the weights on vllm after a change to the training model, which doesn't happen every step IF you're using gradient accumulation

# changed self.llm -> self.vlm to make it obvious that we are using a vlm not llm
