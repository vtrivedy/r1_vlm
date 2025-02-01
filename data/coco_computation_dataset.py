import os
import random

from datasets import Dataset, DatasetDict, load_dataset
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

load_dotenv(dotenv_path=find_dotenv())


# Creates a dataset mapping an image from COCO to a computation problem
# e.g. "For this image, multiply the number of dogs by the number of cats"
# Encoded as {"image": ..., "class_1": ..., "class_2": ..., "count_1": ..., "count_2": ..., "operation": ..., "answer": ...}
# In the case where the answer is not an integer, we truncate it to 2 decimal places


def coco_computation_dataset():
    dataset = load_dataset("sunildkumar/coco-class-counts")
    computation_examples_per_image = 10

    operations = [
        ("add", lambda x, y: x + y),
        ("subtract", lambda x, y: x - y),
        ("multiply", lambda x, y: x * y),
        ("divide", lambda x, y: x / y),
    ]

    for split in dataset.keys():  # Use the splits from the base dataset
        print(f"Processing {split} split...")
        for example in tqdm(dataset[split], desc=f"Processing {split} examples"):
            file_name = example["file_name"]
            class_counts = example["class_counts"]
            class_counts = {k: v for k, v in class_counts.items() if v is not None}

            if len(class_counts.keys()) >= 2:
                for _ in range(computation_examples_per_image):
                    class_1, class_2 = random.sample(sorted(class_counts.keys()), 2)
                    count_1 = class_counts[class_1]
                    count_2 = class_counts[class_2]

                    operation, operation_fn = random.choice(operations)
                    answer = operation_fn(count_1, count_2)

                    # truncate the answer to 2 decimal places
                    answer = float(f"{answer:.2f}")

                    yield {
                        "split": split,
                        "file_name": file_name,
                        "class_1": class_1,
                        "class_2": class_2,
                        "count_1": count_1,
                        "count_2": count_2,
                        "operation": operation,
                        "answer": answer,
                    }


if __name__ == "__main__":
    examples = list(coco_computation_dataset())
    dataset = DatasetDict(
        {
            split: Dataset.from_list([ex for ex in examples if ex["split"] == split])
            for split in load_dataset("sunildkumar/coco-class-counts").keys()
        }
    )

    dataset.push_to_hub(
        "sunildkumar/coco-computation", token=os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
