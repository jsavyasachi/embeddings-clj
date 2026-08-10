# Changes

## 0.1.0 (unreleased)

Initial release.

- `embeddings.core`: `load-model` / `with-model` / `close` for any
  sentence-transformers-style ONNX export (`model.onnx` + `tokenizer.json`),
  `embed` / `embed-batch` / `dimension`. The library introspects session inputs.
  It gives `input_ids` / `attention_mask` / `token_type_ids` only when the model
  declares them. It pads batches with attention-mask awareness. It pools rank-3
  token outputs and uses rank-2 pre-pooled outputs directly.
- `embeddings.pooling`: mask-aware `:mean` / `:cls` / `:max`.
- `embeddings.math`: `dot`, `norm`, `l2-normalize`, `cosine-similarity`
  on primitive `float[]`.
- Tests against real all-MiniLM-L6-v2 show parity with the Python
  sentence-transformers reference within 2e-2 per component. Batch results
  match single embeds.
