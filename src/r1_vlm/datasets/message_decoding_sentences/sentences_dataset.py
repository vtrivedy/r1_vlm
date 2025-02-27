from datasets import load_dataset

# creates a dataset of sentences that we will use for this task, starting from https://huggingface.co/datasets/agentlans/high-quality-english-sentences
# see the dataset here: https://huggingface.co/datasets/sunildkumar/english-sentences

sentences_dataset = load_dataset(
    "agentlans/high-quality-english-sentences", split="train"
)
print(f"There are {len(sentences_dataset)} sentences in the dataset")


# filter out sentences that are too short or too long
shortest_length = 30
longest_length = 100
filtered_sentences = [
    s
    for s in sentences_dataset
    if len(s["text"]) >= shortest_length and len(s["text"]) <= longest_length
]
print(f"After filtering, there are {len(filtered_sentences)} sentences in the dataset")

# sort the sentences by length and interpolate over the range to get various lengths
dataset_size = 10000
sorted_sentences = sorted(filtered_sentences, key=lambda x: len(x["text"]))
selected_sentences = sorted_sentences[:: int(len(sorted_sentences) / dataset_size)]
# Convert back to a dataset
sentences_dataset = sentences_dataset.from_list(selected_sentences)

print(f"Selected {len(sentences_dataset)} sentences")

# upload to hub
sentences_dataset.push_to_hub("sunildkumar/english-sentences")
