# Getting a model

Any sentence-transformers-style ONNX export works: a directory containing
`model.onnx` and `tokenizer.json`. `embeddings.hub/fetch-model` downloads one
from the Hugging Face Hub into `~/.cache/embeddings-clj` (tries
`onnx/model.onnx`, falls back to `model.onnx`; already-cached files are not
re-downloaded):

```clojure
(require '[embeddings.hub :as hub])

(def model-dir (hub/fetch-model "sentence-transformers/all-MiniLM-L6-v2"))
;; opts: {:cache-dir "..." :revision "main" :variant ...}
```

Quantized exports: `:variant :quantized` tries the common quantized paths
(`onnx/model_quantized.onnx`, `onnx/model_qint8_avx512_vnni.onnx`,
`onnx/model_int8.onnx`, `model_quantized.onnx`) in order; a string `:variant`
names an explicit repo-relative `.onnx` path (e.g. `"onnx/model_q4.onnx"`).
Variants are cached in their own subdirectory, so full-precision and quantized
copies of the same model coexist:

```clojure
(hub/fetch-model "sentence-transformers/all-MiniLM-L6-v2" {:variant :quantized})
```

Or fetch manually:

```bash
mkdir -p models/all-MiniLM-L6-v2
curl -fL -o models/all-MiniLM-L6-v2/model.onnx \
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx
curl -fL -o models/all-MiniLM-L6-v2/tokenizer.json \
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json
```

Compatible model families include all-MiniLM, all-mpnet, BGE, GTE, and E5
(anything whose ONNX graph takes `input_ids`/`attention_mask`/`token_type_ids`
and outputs token embeddings or a pre-pooled sentence embedding).

E5-style models expect instruction prefixes; pass `:prefix` per call:

```clojure
(emb/embed model "how do I bake bread" {:prefix "query: "})
(emb/embed-batch model documents {:prefix "passage: "})
```

