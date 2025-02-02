# r1_vlm
Trying GRPO on a toy task on a small VLM.


# Idea
This [blog post](https://www.philschmid.de/mini-deepseek-r1) shows how GRPO an LLM to do r1 style reasoning
on a toy problem. As far as I know, no one has tried this on a VLM. My idea is to generate a simple visual reasoning
dataset (not unlike the counting game in the blog post) and see if a VLM can do it.

## Dataset
Using the COCO dataset, I've generate a dataset of computation problems. For each image, I ask it to {add, subtract, multiply, divide}
the counts of two classes that are present in the image. For example, "Multiply the number of dogs by the number of cats in the image".


