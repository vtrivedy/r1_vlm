#!/usr/bin/env python3
import os

import tqdm
from datasets import Dataset, DatasetDict
from dotenv import find_dotenv, load_dotenv
from pycocotools.coco import COCO

load_dotenv(dotenv_path=find_dotenv())


def count_classes(split="train"):
    """
    Creates a HF dataset mapping each image in the COCO dataset to the
    number of instances of each class in the image.

    Args:
        split (str): Dataset split ('train' or 'val')
    """
    data_dir = "/millcreek/data/academic/coco"
    ann_file = os.path.join(data_dir, "annotations", f"instances_{split}2017.json")
    coco = COCO(ann_file)

    # Get all image IDs
    img_ids = coco.getImgIds()

    data = []
    for img_id in tqdm.tqdm(img_ids, desc=f"Processing {split} split"):
        img_info = coco.loadImgs(img_id)[0]
        # Retrieve all annotation IDs for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        class_counts = {}
        # Count each category occurrence in the current image
        for ann in anns:
            cat_id = ann["category_id"]
            # Retrieve the category name from the category id
            cat_name = coco.loadCats(cat_id)[0]["name"]
            class_counts[cat_name] = class_counts.get(cat_name, 0) + 1

        class_counts = {k: v for k, v in class_counts.items() if v > 0}
        data.append({"file_name": img_info["file_name"], "class_counts": class_counts})

    # Create a Hugging Face dataset
    dataset = Dataset.from_dict(
        {
            "file_name": [item["file_name"] for item in data],
            "class_counts": [item["class_counts"] for item in data],
        }
    )

    return dataset


if __name__ == "__main__":
    # Process both splits
    train_dataset = count_classes("train")
    val_dataset = count_classes("val")

    # Combine into a single dataset with splits
    dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

    dataset.push_to_hub(
        "sunildkumar/coco-class-counts", token=os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
