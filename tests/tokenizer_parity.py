"""Native Rust tokenization parity with the supplied official tokenizer.json."""

import argparse
import numpy as np
from tokenizers import Tokenizer
from embeddinggemma2_rocm import Embedder

p = argparse.ArgumentParser()
p.add_argument("--bundle", required=True)
p.add_argument("--tokenizer", required=True)
a = p.parse_args()
texts = [
    "",
    "hello world",
    "日本語の検索テスト",
    "a  b\n c",
    "café naïve العربية",
    "👩‍💻 🐈",
    "<bos>literal<eos>",
    "\t leading and trailing ",
]
t = Tokenizer.from_file(a.tokenizer)
with Embedder(a.bundle) as model:
    for prompt, prefix in [
        (None, ""),
        ("query", "task: search result | query: "),
        ("document", "title: none | text: "),
    ]:
        ids, mask = model.tokenize(texts, prompt=prompt)
        for i, text in enumerate(texts):
            expected = t.encode(prefix + text).ids
            np.testing.assert_array_equal(ids[i, : len(expected)], expected)
            assert int(mask[i].sum()) == len(expected)
print("native tokenizer parity passed: 24 multilingual/prefix cases")
