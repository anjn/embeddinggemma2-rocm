# Measured GPU tuning recipes

This directory contains portable JSON settings produced by `eg2 export-tuning`.
Only complete searches with matching retrieval validation can be published.
Preparation automatically uses an exact compatible GPU/model/profile/software
match. It compiles the selected configuration locally, without performance search.
Unmatched systems use the compiled default; run `eg2 tune` to measure that GPU.
Model weights, GPU binaries, embeddings and absolute local paths are not stored here.
