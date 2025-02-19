from datasets import load_dataset

# creates a dataset of words that we will use for this task, starting from https://huggingface.co/datasets/Maximax67/English-Valid-Words
# Includes the 10k most popular words that are at least 3 characters long.

words_dataset = load_dataset(
    "Maximax67/English-Valid-Words", "sorted_by_frequency", split="train"
)
print(f"There are {len(words_dataset)} words in the dataset")

print(words_dataset[0])
# dict_keys(['Rank', 'Word', 'Frequency count', 'Stem', 'Stem valid probability'])
print(words_dataset[0].keys())

# filter out words that are None or less than 3 characters
words_dataset = words_dataset.filter(
    lambda x: x["Word"] is not None and len(x["Word"]) >= 3
)

# select the 10k most popular words
words_dataset = words_dataset.select(range(10000))

print(f"There are {len(words_dataset)} words in our dataset")


# drop all but the word column
words_dataset = words_dataset.map(
    lambda x: {"word": x["Word"]}, remove_columns=words_dataset.column_names
)
# upload to hub
words_dataset.push_to_hub("sunildkumar/popular_english_words")
