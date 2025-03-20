import inspect
from typing import Any, Callable, Dict, List

from transformers import AutoProcessor
from trl.trainer.grpo_trainer import RewardFunc

from r1_vlm.environments.multistep_vision_env import MultistepVisionEnv
from r1_vlm.tools.digits_answer_tool import get_answer
from r1_vlm.tools.tool_prompts import DEFAULT_TOOL_PROMPT_TEMPLATE


def infer_schema_from_function(func: Callable) -> Dict[str, Any]:
    """Infers a tool schema from a function's signature and docstring."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    # Parse docstring sections
    doc_parts = doc.split("\n\n")
    description = doc_parts[0].strip()
    
    print(f"parts: {doc_parts}")
    
    # Extract examples if present
    examples = []
    for part in doc_parts:
        if part.startswith("Examples:"):
            examples = [line.strip() for line in part.split("\n")[1:] if line.strip()]
    
    # Build args schema
    args = {}
    for name, param in sig.parameters.items():
        param_doc = ""
        for part in doc_parts:
            if part.strip().startswith("Args:"):
                for line in part.split("\n")[1:]:
                    if line.strip().startswith(f"{name}:"):
                        param_doc = line.strip()[len(name)+1:].strip()
        
        args[name] = {
            "type": str(param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "any"),
            "description": param_doc,
        }
        if param.default != inspect.Parameter.empty:
            args[name]["default"] = param.default
    
    return {
        "name": func.__name__,
        "description": description,
        "args": args,
        "returns": str(sig.return_annotation.__name__ if sig.return_annotation != inspect.Parameter.empty else "any"),
        "examples": examples
    }

def format_tool_descriptions(schemas: List[Dict[str, Any]]) -> str:
    """Formats tool schemas into a user-friendly description string."""
    descriptions = []
    for schema in schemas:
        desc = [f"{schema['name']}: {schema['description']}"]
        
        desc.append("\nArguments:")
        for arg_name, arg_info in schema['args'].items():
            default = f" (default: {arg_info['default']})" if 'default' in arg_info else ""
            desc.append(f"  - {arg_name}: {arg_info['description']}{default}")
        
        if schema['examples']:
            desc.append("\nExamples:")
            for example in schema['examples']:
                desc.append(f"  {example}")
        
        descriptions.append("\n".join(desc))
    
    return "\n\n".join(descriptions)

class ToolVisionEnv(MultistepVisionEnv):
    def __init__(self,
                 tools: List[Callable],
                 processing_class: AutoProcessor,
                 sampling_args={
                     "stop": ["</tool>", "</answer>"],
                     "include_stop_str_in_output": True
                 },
                 mask_env_response: bool = True,
                 max_steps: int = 10,
                 **kwargs):
        super().__init__(processing_class=processing_class, sampling_args=sampling_args, mask_env_response=mask_env_response, **kwargs)
        
        # Infer schemas from tool functions
        self.tool_schemas = [infer_schema_from_function(tool) for tool in tools]
        self.tools = {tool.__name__: tool for tool in tools}    
        # Format the system prompt with tool descriptions
        tool_descriptions = format_tool_descriptions(self.tool_schemas)
        formatted_prompt = DEFAULT_TOOL_PROMPT_TEMPLATE.format(tool_descriptions=tool_descriptions)
        
        import ipdb
        ipdb.set_trace()
        print('hi')
    
    
    def get_rubric(self) -> List[RewardFunc]:
        raise NotImplementedError("ToolVisionEnv requires a rubric for your task. Expected to be implemented by subclass.")
    
    def env_response(self) -> List[Dict[str, Any]]:
        raise NotImplementedError("ToolVisionEnv requires a env response for your task. Expected to be implemented by subclass.")
    
    def is_completed(self, messages: List[Dict[str, Any]], **kwargs: Any) -> bool:
        raise NotImplementedError("ToolVisionEnv requires a is_completed method for your task. Expected to be implemented by subclass.")
    
    
        

if __name__ == "__main__":
    ToolVisionEnv(tools=[get_answer], processing_class=None)
        
    
    
    