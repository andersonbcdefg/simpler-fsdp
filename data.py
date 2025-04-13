from tqdm.auto import tqdm
import numpy as np
import tiktoken
import datasets
import random
encoding = tiktoken.encoding_for_model("gpt-4")

def generate_data():
    ds = datasets.load_dataset('pszemraj/simple_wikipedia', split='train')
    texts = [x['text'] for x in ds] # pyright: ignore
    random.seed(42)
    random.shuffle(texts)
    with open('data.txt', 'w') as file:
        for text in texts:
            file.write(text.replace('\n', ' ') + '\n')

def tokenize_all():
    all_tokens = []
    with open('data.txt', 'r') as file:
        for line in tqdm(file, total=226_000, desc="Processing lines"):
            tokens = encoding.encode(line.strip())
            all_tokens.extend(tokens)

    final_array = np.array(all_tokens, dtype='int32')
    print("Total tokens:", len(final_array))
    final_array.tofile('final_data.bin')

def data_loader(batch_size_in_tokens: int):
    data = np.fromfile('final_data.bin', dtype='int32')
    num_tokens = len(data)
    while True:
        for i in range(0, num_tokens - batch_size_in_tokens, batch_size_in_tokens):
            yield data[i:i+batch_size_in_tokens]

if __name__ == "__main__":
    generate_data()
    tokenize_all()
