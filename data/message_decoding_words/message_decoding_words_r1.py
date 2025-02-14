import os

from datasets import Dataset, DatasetDict, load_dataset
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

load_dotenv(dotenv_path=find_dotenv())

# this holds the images/ directory created by message_decoding_words_dataset.py
base_image_path = "/millcreek/home/sunil/r1_vlm/data/message_decoding_words"


def generate_r1_messages(example):
    coded_message = example["coded_message"]
    decoded_message = example["decoded_message"]
    mapping = example["mapping"]
    file_path = os.path.join(base_image_path, example["file_path"])

    assert os.path.exists(file_path), f"File does not exist: {file_path}"

    # add spaces between each character to prevent tokenization issues
    coded_message = " ".join(coded_message)

    instruction = f'Use the decoder in the image to decode this coded message: "{coded_message}". The decoded message should be an english word.'

    ending = 'Show your work in <think> </think> tags and return the answer in <answer> </answer> tags, for example <answer> "CAT" </answer>.'

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
                {"type": "image", "image": file_path},
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
        "decoded_message": decoded_message,  # No spaces here because we want the model to reply with the relevant english word.
    }


def create_r1_message_decoding_dataset():
    dataset = load_dataset("sunildkumar/message-decoding-words")
    processed_datasets = {}
    for split in dataset.keys():
        print(f"Processing {split} split...")
        examples = []
        for example in tqdm(dataset[split], desc=f"Processing {split} examples"):
            processed_example = generate_r1_messages(example)
            examples.append(processed_example)

        processed_datasets[split] = Dataset.from_list(examples)

    return DatasetDict(processed_datasets)


if __name__ == "__main__":
    dataset = create_r1_message_decoding_dataset()
    dataset.push_to_hub(
        "sunildkumar/message-decoding-words-r1",
        token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
    )
