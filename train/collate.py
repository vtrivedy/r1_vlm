import torch
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

# https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl was the best reference for this, with some modifications

# Load processor and set paths
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")


def collate_fn(examples):
    conversations = [ex["messages"] for ex in examples]

    # Clean up None values from the messages
    for conv in conversations:
        for message in conv:
            content = message["content"]
            message["content"] = [
                {k: v for k, v in item.items() if v is not None} for item in content
            ]

    # NOTE: to self - The model never sees the target! So we shouldn't tokenize it!!
    targets = [ex["target"] for ex in examples]

    # apply the chat template to the messages and add image tokens
    texts = processor.apply_chat_template(
        conversations, continue_final_message=True, tokenize=False
    )

    image_inputs = []
    for conv in conversations:
        image_input, _ = process_vision_info(conv)
        image_inputs.append(image_input)

    batch = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    labels = batch["input_ids"].clone()
    labels[
        labels == processor.tokenizer.pad_token_id
    ] = -100  # Mask padding tokens in labels

    # these are the token ids for images, which should be ignored for loss (Not sure this matters for GRPO?)
    image_token_ids = [151652, 151653, 151655]

    # Mask image token IDs in the labels
    for image_token_id in image_token_ids:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels

    # todo: is this right?
    batch["targets"] = torch.tensor(targets)

    return batch


def test_collate():
    dataset = load_dataset("sunildkumar/coco-computation-r1", split="train")

    bs = 5
    examples = dataset.select(range(bs))

    collated = collate_fn(examples)

    assert collated["input_ids"].shape == (bs, 405)
    assert collated["labels"].shape == (bs, 405)
    assert collated["attention_mask"].shape == (bs, 405)
    assert collated["targets"].shape == (bs,)


if __name__ == "__main__":
    test_collate()
