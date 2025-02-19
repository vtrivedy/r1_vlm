# r1_vlm
Trying GRPO on a toy task on a small VLM.


# Idea
This [blog post](https://www.philschmid.de/mini-deepseek-r1) shows how GRPO an LLM to do r1 style reasoning
on a toy problem. As far as I know, no one has tried this on a VLM. My original idea was to prove one can use GRPO on a VLM as well and show it can improve performance on a toy task. 
Now that I've achieved this, next I'm trying to extend this to more complex tasks. Currently, I'm working on integrating the `verifiers` library, which will unlock standard patterns for more complex model
interactions, like multi-step reasoning, and tool use.


# Task 1: Digit Recognition
As proof that my code works, I trained Qwen2.5VL 3B on a digit recognition task derived from MNIST. In each image, there are one, two or three digits. For each image, the model is either
asked to return the list of digits in ascending order, or the sum of the digits.

You can see the "raw" dataset [here](https://huggingface.co/datasets/sunildkumar/digit-recognition) and then the R1 setup on top [here](https://huggingface.co/datasets/sunildkumar/digit-recognition-r1).

You can run training on 4 GPUs, 3 for training, one for completion generation with `vllm` using the following command. I've tested it on 4x A100 80GB GPUs. You can also get it running on two GPUs as well by tuning down the number of generations.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --config_file train/multi_gpu_3only.yaml digit_recognition_environment/train.py 
```

## Training:
```
# run from root of repo
uv run accelerate launch --config_file train/multi_gpu.yaml  train/train.py

uv run accelerate launch --config_file train/multi_gpu.yaml  train/train_counting.py

CUDA_VISIBLE_DEVICES=1,2,3 uv run accelerate launch --config_file train/multi_gpu_3only.yaml  train/train_counting.py

CUDA_VISIBLE_DEVICES=1 uv run train/train_counting.py


CUDA_VISIBLE_DEVICES=1 uv run train/train_message_decoding.py

CUDA_VISIBLE_DEVICES=1,2,3 uv run accelerate launch --config_file train/multi_gpu_3only.yaml train/train_message_decoding.py 2>&1 | tee message_decoding_logs_$(date +%Y%m%d_%H%M%S).log


# 2b message decoding new trainer single gpu
CUDA_VISIBLE_DEVICES=1 uv run train/train_message_decoding_new_trainer.py

# 2b message decoding new trainer all gpu
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --config_file train/multi_gpu.yaml train/train_message_decoding_new_trainer.py 2>&1 | tee message_decoding_new_trainer_logs_$(date +%Y%m%d_%H%M%S).log

# 3b word decoding single gpu
CUDA_VISIBLE_DEVICES=1 uv run train/train_message_decoding_words.py

# 3b word decoding 3 gpu
CUDA_VISIBLE_DEVICES=1,2,3 uv run accelerate launch --config_file train/multi_gpu_3only.yaml train/train_message_decoding_words.py 2>&1 | tee message_decoding_words_logs_$(date +%Y%m%d_%H%M%S).log



# 3b message decoding words vllm 3 gpu for train, 1 for vllm
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --config_file train/multi_gpu_3only.yaml train/train_message_decoding_words_vllm.py 2>&1 | tee message_decoding_words_vllm_logs_$(date +%Y%m%d_%H%M%S).log

```