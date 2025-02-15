import random
from threading import Thread

import gradio as gr
import torch  # Need this for torch.no_grad()
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TextIteratorStreamer,
)
from trl import ModelConfig


def get_eval_dataset():
    full_dataset = load_dataset("sunildkumar/message-decoding-words")["train"]
    full_dataset = full_dataset.shuffle(seed=42)

    # split the dataset with the same seed as used in the training script
    splits = full_dataset.train_test_split(test_size=0.1, seed=42)
    test_dataset = splits["test"]

    return test_dataset


def load_model_and_tokenizer():
    model_config = ModelConfig(
        model_name_or_path="/millcreek/home/sunil/r1_vlm/vlm-r1-message-decoding-words/checkpoint-340",
        torch_dtype="bfloat16",
        use_peft=False,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_config.model_name_or_path,
        torch_dtype=model_config.torch_dtype,
        use_cache=False,
        device_map="auto",  # Force CPU usage
    )

    # put model in eval mode
    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path, padding_side="left"
    )

    return model, processor


# Load resources once at startup
eval_dataset = get_eval_dataset()
model, processor = load_model_and_tokenizer()


def show_random_example():
    # Get a random example
    random_idx = random.randint(0, len(eval_dataset) - 1)
    example = eval_dataset[random_idx]

    # Return image for display, mapping for state, and image for state
    return example["image"], example["mapping"], example["image"]


def prepare_model_input(image, mapping, processor, submitted_word):
    """
    Prepare the input for the model using the mapping, processor, and submitted word.

    Args:
        image: The decoder image to use
        mapping (dict): The mapping data from the dataset
        processor: The model's processor/tokenizer
        submitted_word (str): The word submitted by the user

    Returns:
        dict: The processed inputs ready for the model
    """
    decoded_message = submitted_word.upper()
    print(f"Decoded message: {decoded_message}")

    # reverse the decoder to encode the word
    encoder = {v: k for k, v in mapping.items()}
    print(f"Encoder: {encoder}")
    coded_message = [encoder[c] for c in decoded_message]
    print(f"Coded message: {coded_message}")

    # add spaces between each character to prevent tokenization issues
    coded_message = " ".join(coded_message)

    instruction = f'Use the decoder in the image to decode this coded message: "{coded_message}". The decoded message should be an english word.'

    ending = 'Show your work in <think> </think> tags and return the answer in <answer> </answer> tags, for example <answer> "CAT" </answer>.'

    instruction = f"{instruction} {ending}"

    print(f"Instruction: {instruction}")
    r1_messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me solve this step by step.\n<think>"}
            ],
        },
    ]

    texts = processor.apply_chat_template(
        r1_messages, continue_final_message=True, tokenize=False
    )

    image_input, _ = process_vision_info(r1_messages)

    image_input = [image_input]

    batch = processor(
        text=texts,
        images=image_input,
        padding=True,
        return_tensors="pt",
    )

    return batch


def validate_and_submit(word):
    # Check if input contains only letters
    if not word.isalpha():
        return gr.update(), gr.update(), gr.update()

    word = word.lower()

    # Replace input with submitted word and show run button
    return (
        gr.update(value=word, interactive=False, label="Submitted Word"),
        gr.update(visible=False),  # Hide the submit button
        gr.update(visible=True),  # Show the run button
    )


def prepare_for_inference():
    """Setup function that runs before streaming starts"""
    return (
        gr.update(value="", visible=True),  # Clear and show output
        gr.update(interactive=False),  # Disable run button
        gr.update(visible=True),  # Show loading indicator
    )


def run_inference(word, image, mapping):
    """Main inference function, now focused just on generation"""
    if not word or not image or not mapping:
        raise gr.Error("Please submit a word and load a decoder first")

    # Prepare model input
    model_inputs = prepare_model_input(image, mapping, processor, word)
    model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}

    # Initialize streamer
    streamer = TextIteratorStreamer(
        tokenizer=processor,
        skip_special_tokens=True,
        decode_kwargs={"skip_special_tokens": True},
    )

    # Set up generation parameters
    generation_kwargs = dict(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=1.0,
        streamer=streamer,
    )

    # Start generation in a separate thread with torch.no_grad()
    def generate_with_no_grad():
        with torch.no_grad():
            model.generate(**generation_kwargs)

    thread = Thread(target=generate_with_no_grad)
    thread.start()

    # Stream the output
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text

    thread.join()
    return generated_text


# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Vision Language Model Demo")
    current_mapping = gr.State()
    current_image = gr.State()

    with gr.Row():
        # Image display component
        image_output = gr.Image(label="Decoder")

    # Button to load new random example
    next_button = gr.Button("Show Random Example")
    next_button.click(
        fn=show_random_example, outputs=[image_output, current_mapping, current_image]
    )

    # Text input for the word
    word_input = gr.Textbox(
        label="Enter a single word",
        placeholder="Enter word here...",
        max_lines=1,
        show_copy_button=False,
    )
    submit_button = gr.Button("Submit Word")

    # Output area for model response
    model_output = gr.Textbox(
        label="Model Output",
        interactive=False,
        visible=False,
        max_lines=10,  # Set maximum visible lines
        container=True,  # Enable scrolling container
        show_copy_button=True,  # Useful for long outputs
    )

    # Run model button (hidden initially)
    run_button = gr.Button("Run Model", visible=False)

    # Add loading indicator
    with gr.Row():
        loading_indicator = gr.HTML(visible=False)

    # Validate word on submit and update interface
    submit_button.click(
        fn=validate_and_submit,
        inputs=[word_input],
        outputs=[word_input, submit_button, run_button],
    )

    # Run inference when run button is clicked
    run_button.click(
        fn=prepare_for_inference,
        outputs=[model_output, run_button, loading_indicator],
    ).then(
        fn=run_inference,
        inputs=[word_input, current_image, current_mapping],
        outputs=model_output,
        api_name=False,
    ).then(
        # Reset interface after generation
        lambda: (
            gr.update(interactive=True),  # Re-enable run button
            gr.update(visible=False),  # Hide loading indicator
        ),
        None,
        [run_button, loading_indicator],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
