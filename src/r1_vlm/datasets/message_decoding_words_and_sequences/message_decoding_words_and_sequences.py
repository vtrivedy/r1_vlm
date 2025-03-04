import os
import random
from pathlib import Path

from datasets import Dataset, load_dataset
from tqdm.auto import tqdm

from r1_vlm.datasets.message_decoding_words.message_decoding_words_dataset import (
    generate_decoder_image,
    generate_mapping,
)

# setting a seed for reproducibility
random.seed(42)


def create_sample(example):
    message = example["text"]

    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    assert len(alphabet) == 26
    mapping = generate_mapping(alphabet)

    decoder_image = generate_decoder_image(mapping=mapping, image_size=400)

    # add a mapping for the underscore ("_") character. It will map to " " (space).
    # This is so we can effectively communicate the space character in the coded message.
    mapping["_"] = " "

    # reverse the mapping to encode the message
    reverse_mapping = {v: k for k, v in mapping.items()}

    # create the coded and decoded message. If we encounter a character that is not in the mapping,
    # we will map it to itself.
    coded_message = ""
    decoded_message = ""
    for char in message:
        # check if the character is in the mapping
        if char.isascii() and (char.isalpha() or char == " "):
            is_lower = char.islower()

            # pass the lowercase version of the character to the mapping
            # .lower() on the space character is a no-op
            key_char = char.lower() if not is_lower else char
            mapped_char = reverse_mapping[key_char]

            # add lowercase char to the coded message
            coded_message += mapped_char
            # use the lowercase version of the character in the decoded message too.
            decoded_message += char.lower() if not is_lower else char

        # if the character is not in the mapping, something is wrong
        else:
            raise ValueError(f"Character {char} is not in the mapping")

    return decoder_image, decoded_message, coded_message, mapping


def create_dataset():
    data = {
        "coded_message": [],
        "decoded_message": [],
        "mapping": [],
        "file_path": [],
        "image": [],
        "task": [],
    }

    image_dir = Path(__file__).parent / "images"
    image_dir.mkdir(exist_ok=True)

    # verify that the image directory is empty
    if len(list(image_dir.glob("*.png"))) > 0:
        raise ValueError("Image directory is not empty")

    # create dataset of words, word pairs, and word triples
    words_dataset = load_dataset("sunildkumar/popular_english_words", split="train")
    words_list = [example["word"] for example in words_dataset]
    examples = []

    # single word examples
    for word in words_list:
        examples.append({"text": word, "task": "word"})

    # word pair examples
    for i in range(len(words_list)):
        word1 = random.choice(words_list)
        word2 = random.choice(words_list)
        examples.append({"text": f"{word1} {word2}", "task": "word_2"})

    # word triple examples
    for i in range(len(words_list)):
        word1 = random.choice(words_list)
        word2 = random.choice(words_list)
        word3 = random.choice(words_list)
        examples.append({"text": f"{word1} {word2} {word3}", "task": "word_3"})

    data = {
        "coded_message": [],
        "decoded_message": [],
        "mapping": [],
        "file_path": [],
        "image": [],
        "task": [],
    }

    # create dataset from examples
    for i, example in tqdm(enumerate(examples), total=len(examples)):
        text = example["text"]
        task = example["task"]

        decoder_image, message, coded_message, mapping = create_sample(example)

        fpath = f"images/{i}.png"
        full_path = image_dir / f"{i}.png"

        # verify that the image doesn't already exist, if it does, something is wrong as we should error
        if full_path.exists():
            raise ValueError(f"Image {full_path} already exists")

        # Use full path for saving the image
        full_path = image_dir / f"{i}.png"

        decoder_image.save(full_path)

        data["coded_message"].append(coded_message)
        data["decoded_message"].append(message)
        data["mapping"].append(mapping)
        data["file_path"].append(fpath)
        data["image"].append(decoder_image)
        data["task"].append(task)

    decoding_dataset = Dataset.from_dict(data)

    decoding_dataset.push_to_hub(
        "sunildkumar/message-decoding-words-and-sequences",
        token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
    )


if __name__ == "__main__":
    create_dataset()
