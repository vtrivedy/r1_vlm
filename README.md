# r1_vlm
Trying GRPO on a toy task on a small VLM.


# Idea
This [blog post](https://www.philschmid.de/mini-deepseek-r1) shows how GRPO an LLM to do r1 style reasoning
on a toy problem. As far as I know, no one has tried this on a VLM. My idea is to generate a simple visual reasoning
dataset (not unlike the counting game in the blog post) and see if a VLM can do it.

## Dataset
Using the COCO dataset, I've generate a dataset of computation problems. For each image, I ask it to {add, subtract, multiply, divide}
the counts of two classes that are present in the image. For example, "Multiply the number of dogs by the number of cats in the image".

See the dataset [here](https://huggingface.co/datasets/sunildkumar/coco-computation-r1).


https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl is super helpful for sample code on how to FT a Qwen2VL generally. 

Custom fork of TRL for GRPO on VLMs: https://github.com/sunildkumar/trl. As of the time of writing, the latest version of GRPOTrainer does not support VLMs. 





