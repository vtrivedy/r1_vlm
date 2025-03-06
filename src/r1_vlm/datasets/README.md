Datasets for r1_vlm project. 

`r1_digits.py`: a dataset of 1-3 MNIST digits. The model must either return a list of the digits sorted from lowest to highest, or the sum of the digits. Good for demonstrating that the code works as MNIST is easy. The underlying data for this dataset is generated with: https://github.com/sunildkumar/counting-mnist

`/message_decoding_words`: a dataset that maps common english words to a scrambled version of the word. The model must decode the word back to the original word using a provided decoder image.

`/message_decoding_words_and_sequences`: a dataset that extends `message_decoding_words` to include word pairs and word triples. 
Used for first blog post on the project. We have a model that reliably solves this task.
