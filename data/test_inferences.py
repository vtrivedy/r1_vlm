from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import ModelConfig
import imgcat
from PIL import Image

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", padding_side="left"
)

dataset = load_dataset("sunildkumar/coco-counts-r1")
# load the model
model_config = ModelConfig(
    model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype="bfloat16",
    use_peft=True,
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_config.model_name_or_path,
    torch_dtype=model_config.torch_dtype,
    # has to be set to false for gradient checkpointing to work
    use_cache=False,
    device_map="auto",
)

for example in dataset["train"]:
      
    image_path = example["messages"][1]['content'][0]['image']
    class_1 = example['class_1']
    
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {
                "type": "text", 
                "text": f'Detect all {class_1} in the image and return their locations in the form of coordinates. The format of output should be like {{"bbox_2d": [x1, y1, x2, y2], "label": "<label>"}}'
            },
        ],
    }
    ]   

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


    image = Image.open(image_path)
    imgcat.imgcat(image)
    texts = processor.apply_chat_template(
        messages, continue_final_message=True, tokenize=False
    )

    print(texts)

    image_input, _ = process_vision_info(messages)

    batch = processor(
        text=texts,
        images=image_input,
        padding=True,
        return_tensors="pt",
    )
    
    import ipdb
    ipdb.set_trace()

    batch = batch.to("cuda")

    output = model.generate(**batch, max_length=100000)

    print(processor.decode(output[0], skip_special_tokens=False))
    print("output ^^^^")
