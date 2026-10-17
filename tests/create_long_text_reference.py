"""Create an original-model CPU reference beyond the 512-token local window."""

import argparse
from pathlib import Path
import numpy as np
import torch
from embeddinggemma2_rocm.preparation import load_model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
model = load_model(args.model)
text = "the cat in the photograph. " * 100
features = model[0].preprocess([text], prompt="task: search result | query: ")
length = features["input_ids"].shape[1]
assert 512 < length <= 1024
with torch.inference_mode():
    output = model(features)["sentence_embedding"].float().numpy()
path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)
np.savez(
    path,
    ids=features["input_ids"].numpy(),
    mask=features["attention_mask"].numpy(),
    embeddings=output,
    text=text,
)
print(dict(sequence=length, output=str(path)))
