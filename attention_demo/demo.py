# seeing if we can visualize the attention weights during decoding
# run with CUDA_VISIBLE_DEVICES=0 uv run attention_demo/demo.py
import imageio.v3 as imageio
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

checkpoint = (
    "/millcreek/home/sunil/r1_vlm/vlm-r1-message-decoding-words/checkpoint-2300"
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=checkpoint,
    torch_dtype="bfloat16",
    device_map="auto",
    attn_implementation="eager",
)
model.eval()
print("model loaded")

processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path=checkpoint,
    padding_side="left",
)

dataset = load_dataset("sunildkumar/message-decoding-words-r1")["train"]

example = dataset[0]


messages = example["messages"]
for message in messages:
    content = message["content"]
    message["content"] = [
        {k: v for k, v in item.items() if v is not None} for item in content
    ]


text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, _ = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

print("Starting generation")
generated_output = model.generate(
    **inputs, max_new_tokens=128, output_attentions=True, return_dict_in_generate=True
)
print("Generation complete")
# Extract attention weights
attention_weights = generated_output.attentions


def visualize_attention_step(attention_weights_step, token_ids, processor, step_number):
    # attention_weights_step shape is [1, 16, seq_len, seq_len]
    # First squeeze out the batch dimension
    attention = attention_weights_step.squeeze(0)  # now [16, seq_len, seq_len]

    # Get attention weights for the last generated token (last position)
    # Average across attention heads
    last_token_attention = attention[:, -1, :].mean(dim=0).detach().cpu().float()

    # renormalize the attention weights so they sum to 1
    last_token_attention = last_token_attention / last_token_attention.sum()
    attention_weights_np = last_token_attention.numpy()

    tokens = processor.batch_decode(
        token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    # Create a white image
    img_width = 1200
    img_height = 800
    image = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(image)

    # Try to load a monospace font, fallback to default if not available
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14
        )
    except:
        font = ImageFont.load_default()

    # Calculate text positions
    x, y = 20, 20  # Starting position
    max_width = img_width - 40  # Margin on both sides
    line_height = 20

    for i, (token, weight) in enumerate(zip(tokens, attention_weights_np)):
        # Scale weight to make colors more visible
        scaled_weight = min(weight * 5, 1.0)

        # Handle newlines in token
        if "\\n" in token:
            x = 20  # Reset x to start of line
            y += line_height  # Move to next line
            token = token.replace("\\n", "")  # Remove the newline for display
            if not token:  # Skip empty tokens after removing newline
                continue

        # Get text size for this token
        bbox = draw.textbbox((x, y), token, font=font)
        text_width = bbox[2] - bbox[0]

        # Check if we need to start a new line
        if x + text_width > max_width:
            x = 20  # Reset x to start of line
            y += line_height  # Move to next line

        # For the last token (current generation), use blue
        if i == len(tokens) - 1:
            color = (0, 0, 255)  # Blue for current token
        else:
            # Create red to green gradient based on attention weight
            red = int(255 * (1 - scaled_weight))
            green = int(255 * scaled_weight)
            color = (red, green, 0)

        # Draw the token
        draw.text((x, y), token, fill=color, font=font)

        # Move x position for next token (add small space between tokens)
        x += text_width + 4

    # Save the image
    image.save(f"attention_visualization_step_{step_number}.png")


# Visualize attention for all generation steps
num_steps = len(attention_weights)
base_sequence = inputs["input_ids"].shape[1]

# Store paths of generated images
image_paths = []

for step in tqdm(range(1, num_steps)):  # start from 1 as step 0 is just the input
    # attention_weights[step][-1] has shape [1, 16, seq_len, seq_len]
    attention_weights_step = attention_weights[step][-1]  # get last layer's attention
    current_tokens = generated_output.sequences[0][: base_sequence + step]
    image_path = f"attention_visualization_step_{step}.png"
    visualize_attention_step(
        attention_weights_step, current_tokens, processor, step_number=step
    )
    image_paths.append(image_path)

# Create movie from the visualization images
frames = []
for image_path in image_paths:
    img = imageio.imread(image_path)
    frames.append(img)

# Write the movie with 0.5 second duration per frame (2 fps)
imageio.imwrite("attention_visualization.mp4", frames, fps=2, codec="libx264")
