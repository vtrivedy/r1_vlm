# This file contains tools for verifying that we can do tool calling
# 1. A tool that takes the task "recognition" or "addition" and returns the answer. It is "secretly" injected with the input data so it can do this. This proves we can do tool calling at all.
# 2. A tool that takes the task and the image as a key in a dict. Proves we can inject image data into the tool call properly.


def get_answer(task: str, **kwargs) -> str:
    '''
    Returns the answer, either listing or summing the digits in the given image.
    
    Args:
        task: str, either "recognition" or "addition"
        kwargs: dict, should not be used. 
    
    
    Returns:
        A string representation of the answer. e.g. "[4, 7]" for recognition, or "11" for addition
        

    Examples:
        <tool>{"name": "get_answer", "args": {"task": "recognition"}}</tool> 
        <tool>{"name": "get_answer", "args": {"task": "addition"}}</tool>
    '''
    
    # NOTE: The tool cheats (purposely)! It's not a "serious" tool, but rather a proof of concept to verify tool calling works properly. 
    print("TOOL CALLED")
    valid_tasks = ["recognition", "addition"]
    if task not in valid_tasks:
        raise ValueError(f" Error: Invalid task: {task}. Valid tasks are: {valid_tasks}")
    
    # verify that the kwargs contains the input example
    if "input_example" not in kwargs:
        raise ValueError(" Error: The kwargs must contain the input example. We should be injecting this data into the tool call in the env response. This is a code error rather than a model error.")
    
    # verify no other kwargs are passed
    if len(kwargs) > 1:
        raise ValueError(" Error: kwargs were passed data from the tool call. This is not allowed. Only use the named arguments.")
    
    input_example = kwargs["input_example"]
    
    if task == "recognition":
        data = input_example["label"]
    elif task == "addition":
        data = input_example["total"]
        
    return str(data)

    
    
    
    
    