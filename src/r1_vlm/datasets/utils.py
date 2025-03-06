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

def test_image_injection():
    """
    Tests the image injection functionality using the message decoding dataset.
    """
    from datasets import load_dataset
    
    # Load a small subset of the dataset for testing
    dataset = load_dataset(
        "sunildkumar/message-decoding-words-and-sequences-r1-testing",
        split="train[:5]"
    )
    
    # Get an example before transformation
    example_before = dataset[0]
    has_placeholder = any(
        item["type"] == "image" and item["image"] == IMAGE_PLACEHOLDER
        for message in example_before["messages"]
        for item in message["content"]
    )
    assert has_placeholder, "Test data should contain IMAGE_PLACEHOLDER"
    
    # Apply the transformation
    transformed_dataset = inject_images_into_dataset(dataset)
    
    # Verify transformation
    example_after = transformed_dataset[0]
    no_placeholder = all(
        not (item["type"] == "image" and item["image"] == IMAGE_PLACEHOLDER)
        for message in example_after["messages"]
        for item in message["content"]
    )
    assert no_placeholder, "IMAGE_PLACEHOLDER should be replaced"
    
    print("Image injection test passed successfully!")
    
    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    test_image_injection()