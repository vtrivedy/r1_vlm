# This file contains tools for verifying that we can do tool calling
# 1. A tool that takes the task "recognition" or "addition" and returns the answer. It is "secretly" injected with the input data so it can do this. This proves we can do tool calling at all.
# 2. A tool that takes the task and the image as a key in a dict. Proves we can inject image data into the tool call properly.


def get_answer(task: str, **kwargs) -> str:
    '''
    Returns the answer, either listing or summing the digits in the given image.
    
    Args:
        task: str, either "recognition" or "addition"
    
    Returns:
        A string representation of the answer. e.g. "[4, 7]" for recognition, or "11" for addition
    '''
    
    valid_tasks = ["recognition", "addition"]
    if task not in valid_tasks:
        raise ValueError(f" Error: Invalid task: {task}. Valid tasks are: {valid_tasks}")
    
    # verify that the kwargs contains the input example
    if "input_example" not in kwargs:
        raise ValueError(" Error: The kwargs must contain the input example. We should be injecting this data into the tool call in the env response.")
    
    input_example = kwargs["input_example"]
    
    if task == "recognition":
        data = input_example["label"]
    elif task == "addition":
        data = input_example["total"]
        
    return str(data)

    
    
    
    
    