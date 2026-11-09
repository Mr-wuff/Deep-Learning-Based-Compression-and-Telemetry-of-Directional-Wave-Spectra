# Public model-definition source snapshot

`CNN.py` and `VAE.py` are text copies of the publicly accessible root files at
[HF Wave_compress_system](https://huggingface.co/wuff-mann/Wave_compress_system/tree/main), fetched 2026-11-09.

These are **model-definition/reference files**, not a complete final training-and-evaluation pipeline.
The public VAE source contains training/archive helpers, but the manuscript's *exact frozen experiment
Notebook, model-state hashes, dataset sample manifest and stress-test evaluation code* are not
yet established by these two files alone.

Authoritative original versions:
- [CNN.py](https://huggingface.co/wuff-mann/Wave_compress_system/blob/main/CNN.py)
- [VAE.py](https://huggingface.co/wuff-mann/Wave_compress_system/blob/main/VAE.py)

For formal reproducibility, compare the code used with the checkpoint hash and an immutable
HF revision. Do not run model files or load untrusted pickled checkpoints without reviewing them.
