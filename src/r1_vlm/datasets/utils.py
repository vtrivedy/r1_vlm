from datasets import Dataset
# we use this placeholder to indicate where to inject images into our datasets.
IMAGE_PLACEHOLDER = "IMAGE_PLACEHOLDER"

def inject_images_into_dataset(dataset: Dataset) -> Dataset:
    """
    Creates a new dataset with images injected into messages where IMAGE_PLACEHOLDER is found.
    Uses Dataset.with_transform to avoid Arrow serialization issues.
    
    Args:
        dataset: A HuggingFace dataset containing 'messages' and 'image' columns
    
    Returns:
        A new dataset with the image injection transform applied
    """
    def _inject_images(example):
        # example["messages"] is a list of list of messages of length 1, so we get the first element to get a list of messages
        assert len(example["messages"]) == 1, "Expected a list of list of messages of length 1"
        messages = example["messages"][0]
        # example["image"] is a list of images of length 1, so we get the first element to get a single image
        assert len(example["image"]) == 1, "Expected a list of images of length 1"
        image = example["image"][0]
    
        for message in messages:
            content = message["content"]
            for item in content:
                if item["type"] == "image" and item["image"] == IMAGE_PLACEHOLDER:
                    item["image"] = image
        
        return example

    return dataset.with_transform(_inject_images)


def preprocess_r1_dataset(dataset: Dataset) -> Dataset:
    """
    Args:
        dataset: A HuggingFace dataset containing 'messages' and 'image' columns created by r1_vlm.datasets 
    Returns:
        A new dataset with the image injection transform applied. 
    
    """
    
    # thin wrapper, as I figure we'll need to do more preprocessing here eventually.
    transformed_dataset = inject_images_into_dataset(dataset)
    return transformed_dataset