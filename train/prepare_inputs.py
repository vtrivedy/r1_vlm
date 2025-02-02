from qwen_vl_utils import process_vision_info


def tokenize_and_inject_images(*, inputs, processing_class):
    """
    inputs: a list of inputs, in this case a list of examples from our dataset
    processing_class: the processing class to use to process the inputs. This is a VLM processor object from the transformers library.

    This is sorta like a collation function but for GRPO

    returns: batch: a batch of inputs ready for training, prompts: The prompts (in text) that were used to generate the batch
    """

    conversations = [ex["messages"] for ex in inputs]

    # Clean up None values from the messages
    for conv in conversations:
        for message in conv:
            content = message["content"]
            message["content"] = [
                {k: v for k, v in item.items() if v is not None} for item in content
            ]

    # apply the chat template to the messages and add image tokens
    texts = processing_class.apply_chat_template(
        conversations, continue_final_message=True, tokenize=False
    )

    image_inputs = []
    for conv in conversations:
        image_input, _ = process_vision_info(conv)
        image_inputs.append(image_input)

    batch = processing_class(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    # NOTE: There is no need to create labels as we are not autoregressively training!

    return batch, texts
