import numpy as np
import PIL
from datasets import Dataset
from tqdm import tqdm

# This file contains tools for verifying that we can do tool calling
# 1. A tool that takes the task "recognition" or "addition" and returns the answer. It is "secretly" injected with the input data so it can do this. This proves we can do tool calling at all.
# 2. A tool that takes the task and the image as a key in a dict. Proves we can inject image data into the tool call properly.


class DigitsAnswerTool:
    '''
    Hash table from image to the image's label and total. 
    Each image is hashed as a sum of elementwise produce of pixels and a hash matrix of the same shape. 
    '''
    def __init__(self, dataset: Dataset):
        # maps a float to a dict {label: list[int], total: int}
        self.hash_table = {}
        self.hash_matrix = None
    
    def build_hash_table(self, dataset: Dataset) -> None:
        for example in tqdm(dataset, desc="Building hash table"):
            image = example["image"]
            label = example["label"]
            total = example["total"]
            
            assert isinstance(image, PIL.Image.Image)
            assert isinstance(label, list)
            assert isinstance(total, int)
            
            self.add_image(image, label, total)
        
    def generate_hash_matrix(self, shape: tuple[int, int, int]) -> None:
        '''
        Generates a hash matrix for the image. Saves to self.hash_matrix.
        '''
        
        if self.hash_matrix is not None:
            raise ValueError("Error: Hash matrix already generated.")
        
        else:
            # matrix of random floats between 0 and 1
            self.hash_matrix = np.random.random(shape)
        
        
    def hash_image(self, image: PIL.Image.Image) -> float:
        '''
        Hashes the image.
        '''
        
        image = np.array(image)
        
        if self.hash_matrix is None:
            self.generate_hash_matrix(image.shape)
        
        # hash the image
        hash_value = np.sum(image * self.hash_matrix)
        
        return hash_value
        
    
    def add_image(self, image: PIL.Image.Image, label: list[int], total: int):
        '''
        Adds image to the hash table.
        '''
        hash_value = self.hash_image(image)
        
        if hash_value in self.hash_table:
            return
        
        self.hash_table[hash_value] = {"label": label, "total": total}
        
    
    def lookup_image(self, image: PIL.Image.Image) -> dict[str, int | list[int]]:
        '''
        Looks up the image in the hash table and returns the label and total as a dict. 
        '''
        hash_value = self.hash_image(image)
        
        if hash_value not in self.hash_table:
            print("Error: Image not found in the hash table.")
            raise ValueError(" Error: Image not found in the hash table.")
        
        return self.hash_table[hash_value]

# Make digits_answer_tool accessible to get_answer
_digits_answer_tool = None

def set_digits_answer_tool(tool: DigitsAnswerTool):
    global _digits_answer_tool
    _digits_answer_tool = tool

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
    if _digits_answer_tool is None:
        print("Error: DigitsAnswerTool not initialized. Call set_digits_answer_tool first.")
        raise ValueError("DigitsAnswerTool not initialized. Call set_digits_answer_tool first.")
    
    valid_tasks = ["recognition", "addition"]
    if task not in valid_tasks:
        raise ValueError(f" Error: Invalid task: {task}. Valid tasks are: {valid_tasks}")
    
    # verify that the kwargs contains the messages
    if "messages" not in kwargs:
        raise ValueError(" Error: The kwargs must contain the messages. We should be injecting this data into the tool call in the env response. This is a code error rather than a model error.")
    
    # verify no other kwargs are passed
    if len(kwargs) > 1:
        raise ValueError(" Error: kwargs were passed data from the tool call. This is not allowed. Only use the named arguments.")
    
    messages = kwargs["messages"]
    
    images = extract_images(messages)
    if len(images) != 1:
        print(f"Expected to find exactly one image in the messages. Found {len(images)} images.")
        raise ValueError(" Error: Expected exactly one image in the messages. This is a code error rather than a model error.")
    
    image = images[0]
    result = _digits_answer_tool.lookup_image(image)
    
    if task == "recognition":
        data = result["label"]
    else:  # addition
        data = result["total"]
        
    return str(data)

def extract_images(messages: list[dict[str, str]]) -> list[PIL.Image.Image]:
    '''
    Extracts images from the messages.
    
    Args:
        messages: list of messages
    '''
    
    images = []
    for element in messages:
        for message in element["content"]:
            if message["type"] == "image":
                images.append(message["image"])
                
    return images

    
    
