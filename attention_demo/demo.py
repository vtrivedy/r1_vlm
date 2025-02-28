# seeing if we can visualize the attention weights during decoding
# run with CUDA_VISIBLE_DEVICES=0 uv run attention_demo/demo.py
import imageio.v3 as imageio
import numpy as np
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def visualize_attention_step(attention_weights_step, token_ids, processor):
    # attention_weights_step shape is [1, 16, seq_len, seq_len]
    # First squeeze out the batch dimension
    attention = attention_weights_step.squeeze(0)  # now [16, seq_len, seq_len]

    # Get attention weights for the last generated token (last position)
    # Average across attention heads
    last_token_attention = attention[:, -1, :].mean(dim=0).detach().cpu().float()

    # renormalize the attention weights so they sum to 1
    last_token_attention = last_token_attention / last_token_attention.sum()
    attention_weights_np = last_token_attention.numpy()

    # Apply non-linear scaling to enhance visibility of medium attention weights
    # First normalize to [0,1] range
    min_val = attention_weights_np.min()
    max_val = attention_weights_np.max()
    if max_val > min_val:
        normalized_weights = (attention_weights_np - min_val) / (max_val - min_val)
        # Apply power scaling (values less than 1 will be boosted)
        scaled_weights = np.power(
            normalized_weights, 0.4
        )  # Adjust power value to control contrast
    else:
        scaled_weights = attention_weights_np

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

    for i, (token, weight) in enumerate(zip(tokens, scaled_weights)):
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
            red = int(255 * (1 - weight))
            green = int(255 * weight)
            color = (red, green, 0)

        # Draw the token
        draw.text((x, y), token, fill=color, font=font)

        # Move x position for next token (add small space between tokens)
        x += text_width + 4

    # Convert PIL image to numpy array
    return np.array(image)


def create_attention_visualization(
    attention_weights,
    sequences,
    processor,
    layer_idx=-1,
    fps=2,
    output_path="attention_visualization.mp4",
):
    """
    Create a video visualization of attention weights during generation.

    Args:
        attention_weights: List of attention weights from model generation
        sequences: Token sequences from model generation
        processor: Tokenizer/processor for decoding tokens
        layer_idx: Index of attention layer to visualize (default: -1 for last layer)
        fps: Frames per second for output video (default: 2)
        output_path: Path to save the output video (default: "attention_visualization.mp4")
    """
    num_steps = len(attention_weights)
    base_sequence = sequences.shape[1] - num_steps

    # Store frames in memory
    frames = []

    for step in tqdm(range(1, num_steps)):  # start from 1 as step 0 is just the input
        attention_weights_step = attention_weights[step][
            layer_idx
        ]  # get specified layer's attention
        current_tokens = sequences[0][: base_sequence + step]
        frame = visualize_attention_step(
            attention_weights_step, current_tokens, processor
        )
        frames.append(frame)

    # Write the movie
    imageio.imwrite(output_path, frames, fps=fps, codec="libx264")


if __name__ == "__main__":
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
        **inputs,
        max_new_tokens=128,
        output_attentions=True,
        return_dict_in_generate=True,
    )
    print("Generation complete")

    # Get number of layers from the first attention output
    num_layers = len(generated_output.attentions[0])
    print(f"Creating visualizations for {num_layers} layers...")

    # Create visualization for each layer
    for layer_idx in tqdm(range(num_layers), desc="Processing layers"):
        output_path = f"attention_visualization_layer{layer_idx}.mp4"
        create_attention_visualization(
            generated_output.attentions,
            generated_output.sequences,
            processor,
            layer_idx=layer_idx,
            fps=2,
            output_path=output_path,
        )
        print(f"Created {output_path}")

    print("All visualizations complete!")
