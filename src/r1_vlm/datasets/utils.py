from datasets import Dataset, load_dataset
import PIL
# we use this placeholder to indicate where to inject images into our datasets.
# Currently assuming only one image per example. We might have to do something more sophisticated in the future if we want to support multiple input images per example.
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
    def _inject_images(examples):
        messages_batch = examples["messages"]
        images_batch = examples["image"]
        
        # Validate types once for the first example
        if not isinstance(messages_batch[0], list):
            raise ValueError(f"Expected a list of messages, got {type(messages_batch[0])}")
        if not isinstance(images_batch[0], PIL.Image.Image):
            raise ValueError(f"Expected a PIL image, got {type(images_batch[0])}")
        
        for messages, image in zip(messages_batch, images_batch):
            for message in messages:
                content = message["content"]
                [item.update({"image": image}) for item in content if item["type"] == "image" and item["image"] == IMAGE_PLACEHOLDER]

        print("running")
        
        return examples

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


def test():
    dataset = load_dataset("sunildkumar/message-decoding-words-and-sequences-r1-testing")["train"]
    dataset = preprocess_r1_dataset(dataset)
    
    import pdb; pdb.set_trace()
    print(dataset)

if __name__ == "__main__":
    test()