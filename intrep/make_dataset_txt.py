from searchformer.trace import TokenizedDataset
import os, sys, json
from tqdm import tqdm

tok_dataset = TokenizedDataset("maze.10-by-10-deterministic.simple")

os.makedirs("data/", exist_ok=True)
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)


print("---Doing Train Now ---")

train_tok_traces = iter(tok_dataset.train_it(tok_dataset.train_ids))

for i, tok_trace in tqdm(enumerate(train_tok_traces), total=1000000):
    with open(f"data/train/{i}.json", "w") as f:
        json.dump(tok_trace[0].to_dict(), f)


print("---Doing Test Now ---")
test_tok_traces = iter(tok_dataset.test_it(tok_dataset.test_ids))
for i, tok_trace in tqdm(enumerate(test_tok_traces), total=100000):
    with open(f"data/test/{i}.json", "w") as f:
        json.dump(tok_trace[0].to_dict(), f)