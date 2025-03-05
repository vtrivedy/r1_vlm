# r1_vlm
Making it easy to train a VLM with GRPO. 

![Message Decoding Demo](images/demo_video.mp4)

# Installation
TODO: update this section once I port over the code to gl org.
This project relies on forks of some dependencies. First clone this repo. Then clone the following repos adjaces to this one. The two forks are installed as editable dependencies into `r1_vlm`. I don't have a stable branch for which branch on these forks to use, as I'm actively changing them. You can see the latest PRs in the relevant repos or leave an issue on this repo and I'll help you out. 
```
1. git clone git@github.com:sunildkumar/r1_vlm.git
2. git clone git@github.com:sunildkumar/trl.git # this is my fork of TRL with added support for VLMs, verifiers, and vllm.
3. git clone git@github.com:sunildkumar/verifiers.git # this is my fork of the verifiers library, which updates the TRL dependency from HuggingFace's to my fork (above).
```

Afterwards, your directory structure should look like this:
```
r1_vlm/
trl/
verifiers/
```

Then install with `uv`:
```
cd r1_vlm
uv sync
```


# Task 1: Digit Recognition
As proof that my code works, I trained Qwen2.5VL 3B on a digit recognition task derived from MNIST. In each image, there are one, two or three digits. For each image, the model is either
asked to return the list of digits in ascending order, or the sum of the digits.

You can see the "raw" dataset [here](https://huggingface.co/datasets/sunildkumar/digit-recognition) and then the R1 setup on top [here](https://huggingface.co/datasets/sunildkumar/digit-recognition-r1).

![Example of digit recognition task](images/digits_example.png)

You can run training on 4 GPUs, 3 for training, one for completion generation with `vllm` using the following command. I've tested it on 4x A100 80GB GPUs. You can also get it running on two GPUs as well by tuning down the number of generations and not using deepspeed.
```bash

# 4 GPU training with deepspeed
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --config_file src/r1_vlm/deepspeed_configs/multi_gpu_3only.yaml src/r1_vlm/environments/digit_recognition_env/train.py

# 2 GPU training, you'll need to adjust the number of generations in the train.py file.
CUDA_VISIBLE_DEVICES=0,1 uv run src/r1_vlm/environments/digit_recognition_env/train.py
```

Results:
![Correctness Reward](images/digit_recognition_correctness_reward.png)

# Task 2: Message Decoding
This task is more complex. We've created a dataset of "coded messages". Each message is a common english word. We define a coding/decoding mapping that we use to scramble the word, creating a coded message. We then provide the model with the coded message and the decoder image, and ask the model to decode the message back to the original word. This task has the nice property that it is very difficult to solve without engaging with both text and image modalities (technically, for some words, there is a unique solution but it is unlikely that the model knows this). You can see the full dataset
[here](https://huggingface.co/datasets/sunildkumar/message-decoding-words).


## Example Reasoning Trace:
Here's an example where the message is "VISION". 

![Decoder Image](images/example_decoder.webp)

```
system
You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.

user
Use the decoder in the image to decode this coded message: "K X I X R S". The decoded message should be an english word. Show your work in <think> </think> tags and return the answer in <answer> </answer> tags, for example <answer> "CAT" </answer>.

assistant
Let me solve this step by step.
<think> 
1. First, I'll decode each character in the message "K X I X R S" using the provided decoder:

   K → V
   X → I
   I → S
   X → I
   R → O
   S → N

2. Now, I'll Combine the results: "V I S I O N".

Answer: VISION

Answer: VISION

</think>
<answer>VISION</answer>
```

You can train with:
```bash 
# 4 GPU training with deepspeed
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --config_file src/r1_vlm/deepspeed_configs/multi_gpu_3only.yaml src/r1_vlm/environments/message_decoding_env/train.py

# 2 GPU training, you'll need to adjust the number of generations in the train.py file.
CUDA_VISIBLE_DEVICES=0,1 uv run src/r1_vlm/environments/message_decoding_env/train.py
```

Training results:
![Correctness Reward](images/message_decoding_correctness_reward.png)



## Training:
```
# run from root of repo
uv run accelerate launch --config_file train/multi_gpu.yaml  train/train.py

uv run accelerate launch --config_file train/multi_gpu.yaml  train/train_counting.py

CUDA_VISIBLE_DEVICES=1,2,3 uv run accelerate launch --config_file train/multi_gpu_3only.yaml  train/train_counting.py

CUDA_VISIBLE_DEVICES=1 uv run train/train_counting.py


```