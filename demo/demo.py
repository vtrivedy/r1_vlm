import random
from threading import Thread

import gradio as gr
import spaces
import torch  # Need this for torch.no_grad()
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TextIteratorStreamer,
)
from trl import ModelConfig

# run with:
# CUDA_VISIBLE_DEVICES=0 uv run gradio demo/demo.py


def get_eval_dataset():
    full_dataset = load_dataset("sunildkumar/message-decoding-words-and-sequences")[
        "train"
    ]
    full_dataset = full_dataset.shuffle(seed=42)

    # split the dataset with the same seed as used in the training script
    splits = full_dataset.train_test_split(test_size=0.1, seed=42)
    test_dataset = splits["test"]

    return test_dataset


def load_model_and_tokenizer():
    model_config = ModelConfig(
        model_name_or_path="Groundlight/message-decoding-r1",
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


# Move resource loading inside a function
def load_resources():
    global eval_dataset, model, processor
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
    decoded_message = submitted_word.lower()
    print(f"Decoded message: {decoded_message}")

    # reverse the decoder to encode the word
    encoder = {v: k for k, v in mapping.items()}
    print(f"Encoder: {encoder}")
    # leaving the space as is
    coded_message = [encoder[c] if c in encoder else c for c in decoded_message]
    print(f"Coded message: {coded_message}")

    # add spaces between each character to prevent tokenization issues
    coded_message = " ".join(coded_message)

    instruction = (
        f'Use the decoder in the image to decode this coded message: "{coded_message}". '
        "The decoded message will be one or more words. Underscore characters "
        '("_") in the coded message should be mapped to a space (" ") when decoding.'
    )

    ending = (
        "Show your work in <think> </think> tags and return the answer in <answer> </answer> tags. "
        "While thinking, you must include a section with the decoded characters using <chars></chars> tags. "
        "The <chars> section should include the decoded characters in the order they are decoded. It should include the "
        "underscore character wherever there is a space in the decoded message. For example, if the coded message is "
        "a b c _ d e f, the <chars> section might be <chars> c a t _ d o g </chars>. Once you are done thinking, "
        "provide your answer in the <answer> section, e.g. <answer> cat dog </answer>."
    )
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


def encode_word(word, mapping):
    """
    Encode a word using the given mapping.
    """
    if not word or not mapping:
        return ""

    word = word.lower()
    # reverse the decoder to encode the word
    encoder = {v: k for k, v in mapping.items()}
    # leaving the space as is
    coded_message = [encoder[c] if c in encoder else c for c in word]
    return " ".join(coded_message)


def validate_and_submit(word, mapping):
    # Check if input contains only letters
    if not word.replace(" ", "").isalpha():
        gr.Warning(
            "Invalid input! Please enter only English letters and spaces. No numbers or punctuation allowed."
        )
        return (
            gr.update(),  # word input
            gr.update(),  # submit button
            gr.update(interactive=False),  # run button - disable but keep visible
            gr.update(visible=False),  # encoded word display
        )

    if not mapping:
        gr.Warning("Please generate a decoder first")
        return (
            gr.update(),  # word input
            gr.update(),  # submit button
            gr.update(interactive=False),  # run button - disable but keep visible
            gr.update(visible=False),  # encoded word display
        )

    word = word.lower()
    encoded_word = encode_word(word, mapping)

    # Only enable run button if we have a valid encoded word
    has_valid_encoded_word = bool(encoded_word.strip())

    if not has_valid_encoded_word:
        gr.Warning(
            "Invalid input! The word contains characters that cannot be encoded with the current decoder."
        )
        return (
            gr.update(),  # word input
            gr.update(),  # submit button
            gr.update(interactive=False),  # run button - disable but keep visible
            gr.update(visible=False),  # encoded word display
        )

    # Return updates for input, submit button, run button, and encoded word display
    return (
        gr.update(value=word, interactive=False, label="Submitted Word"),
        gr.update(interactive=False),  # Disable submit button
        gr.update(
            interactive=has_valid_encoded_word
        ),  # Enable run button only if valid, but always visible
        gr.update(
            value=f"Encoded word: {encoded_word}", visible=has_valid_encoded_word
        ),  # Show encoded word
    )


def prepare_for_inference():
    """Setup function that runs before streaming starts"""
    return (
        gr.update(value="", visible=True),  # Clear and show output
        gr.update(interactive=False),  # Disable run button
        gr.update(visible=True),  # Show loading indicator
    )


@spaces.GPU
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
    # Load resources when the app starts
    load_resources()

    gr.Markdown("# Message Decoding Demo")
    current_mapping = gr.State()
    current_image = gr.State()

    with gr.Row():
        # Left column - Inputs
        with gr.Column(scale=1):
            # Image display component
            image_output = gr.Image(label="Decoder")

            # Button to load new random example
            next_button = gr.Button("Generate Random Decoder")

            # Text input for the word
            word_input = gr.Textbox(
                label="Enter a single word",
                placeholder="Enter word here...",
                max_lines=1,
                show_copy_button=False,
            )

            # Add encoded word display
            encoded_word_display = gr.Textbox(
                label="Encoded Word",
                interactive=False,
                visible=False,
                max_lines=1,
                show_copy_button=True,
            )

            # Group submit and run buttons vertically
            with gr.Column():
                submit_button = gr.Button("Submit Word")
                run_button = gr.Button("Run Model", interactive=False)

        # Right column - Outputs
        with gr.Column(scale=1):
            # Output area for model response
            model_output = gr.Textbox(
                label="Model Output",
                interactive=False,
                lines=40,
                max_lines=40,
                container=True,
                show_copy_button=True,
            )

            # Add loading indicator
            loading_indicator = gr.HTML(visible=False)

    # Keep all the event handlers the same
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
    gr.Markdown(
        "Note: Only English letters and spaces are allowed. Please do not enter any numbers or punctuation."
    )

    # Add encoded word display
    encoded_word_display = gr.Textbox(
        label="Encoded Word",
        interactive=False,
        visible=False,
        max_lines=1,
        show_copy_button=True,
    )

    # Group submit and run buttons vertically
    with gr.Column():  # Use Column instead of Row for vertical layout
        submit_button = gr.Button("Submit Word")
        run_button = gr.Button(
            "Run Model", interactive=False
        )  # Initialize as visible but disabled

    # Output area for model response
    model_output = gr.Textbox(
        label="Model Output",
        interactive=False,
        visible=False,
        max_lines=10,
        container=True,
        show_copy_button=True,
    )

    # Add loading indicator
    with gr.Row():
        loading_indicator = gr.HTML(visible=False)

    # Validate word on submit and update interface
    submit_button.click(
        fn=validate_and_submit,
        inputs=[word_input, current_mapping],
        outputs=[word_input, submit_button, run_button, encoded_word_display],
    )

    run_button.click(
        fn=prepare_for_inference,
        outputs=[model_output, run_button, loading_indicator],
    ).then(
        fn=run_inference,
        inputs=[word_input, current_image, current_mapping],
        outputs=model_output,
        api_name=False,
    ).then(
        lambda: (
            gr.update(interactive=False),
            gr.update(visible=False),
            gr.update(interactive=True, label="Enter a single word"),
            gr.update(interactive=True),
            gr.update(visible=False),
        ),
        None,
        [
            run_button,
            loading_indicator,
            word_input,
            submit_button,
            encoded_word_display,
        ],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
    # demo.launch()
