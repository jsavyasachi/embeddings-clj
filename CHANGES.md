# Changes

## 0.1.0 (unreleased)

Initial release.

- `embeddings.core`: `load-model` / `with-model` / `close` over any
  sentence-transformers-style ONNX export (`model.onnx` + `tokenizer.json`),
  `embed` / `embed-batch` / `dimension`. Session inputs are introspected
  (`input_ids` / `attention_mask` / `token_type_ids` fed only when the model
  declares them); batches are padded attention-mask-aware; rank-3 token
  outputs are pooled, rank-2 pre-pooled outputs used directly.
- `embeddings.pooling`: mask-aware `:mean` / `:cls` / `:max`.
- `embeddings.math`: `dot`, `norm`, `l2-normalize`, `cosine-similarity`
  on primitive `float[]`.
- Validated against real all-MiniLM-L6-v2: parity with the Python
  sentence-transformers reference within 2e-2 per component, batch results
  identical to single embeds.
