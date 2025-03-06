import os

from datasets import Dataset, DatasetDict, load_dataset
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from r1_vlm.datasets.utils import IMAGE_PLACEHOLDER
load_dotenv(dotenv_path=find_dotenv())


def generate_r1_messages(example):
    coded_message = example["coded_message"]
    decoded_message = example["decoded_message"]
    mapping = example["mapping"]
    task = example["task"]
    image = example["image"]

    # add spaces between each character to prevent tokenization issues
    coded_message = " ".join(coded_message)

    instruction = (
        "Use the decoder in the image to decode a coded message."
        "The decoded message will be one or more words. Underscore characters "
        '("_") in the coded message should be mapped to a space (" ") when decoding.'
    )

    ending = (
        "Show your work in <think> </think> tags and return the answer in <answer> </answer> tags. "
        "While thinking, you must include a section with the decoded characters using <chars></chars> tags. "
        "The <chars> section should include the decoded characters in the order they are decoded. It should include the "
        "underscore character wherever there is a space in the decoded message. For example, if the coded message is "
        "a b c _ d e f, the chars section might be <chars> c a t _ d o g </chars>. You can think about the problem for "
        "as long as you'd like. While thinking, you should robustly verify your solution. Once you are done thinking, "
        f"provide your answer in the <answer> section, e.g. <answer> cat dog </answer>. The coded message is: {coded_message}."
    )
    instruction = f"{instruction} {ending}"

    r1_messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": IMAGE_PLACEHOLDER},
                {"type": "text", "text": instruction},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me solve this step by step.\n<think>"}
            ],
        },
    ]

    return {
        "messages": r1_messages,
        "coded_message": coded_message,
        "mapping": mapping,
        "decoded_message": decoded_message,  # No spaces here because we want the model to reply with the relevant english word(s).
        "task": task,
        "image": image, 
    }


def create_r1_message_decoding_dataset():
    dataset = load_dataset(
        "sunildkumar/message-decoding-words-and-sequences", split="train"
    )

    examples = []
    for example in tqdm(dataset, desc="Processing examples"):
        processed_example = generate_r1_messages(example)
        examples.append(processed_example)

    processed_dataset = Dataset.from_list(examples)

    splits = processed_dataset.train_test_split(test_size=0.1, seed=42)

    dataset_dict = {
        "train": splits["train"],
        "test": splits["test"],
    }

    r1_dataset = DatasetDict(dataset_dict)
    r1_dataset.push_to_hub(
        "sunildkumar/message-decoding-words-and-sequences-r1-testing",
        token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
    )


if __name__ == "__main__":
    create_r1_message_decoding_dataset()