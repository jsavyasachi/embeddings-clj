# Getting a model

Any sentence-transformers-style ONNX export works: a directory that contains
`model.onnx` and `tokenizer.json`. `embeddings.hub/fetch-model` downloads it
from the Hugging Face Hub into `~/.cache/embeddings-clj`. It tries
`onnx/model.onnx` first, then `model.onnx`. It downloads a file again only when
the cache does not hold it:

```clojure
(require '[embeddings.hub :as hub])

(def model-dir (hub/fetch-model "sentence-transformers/all-MiniLM-L6-v2"))
;; opts: {:cache-dir "..." :revision "main" :variant ...
;;       :token "hf_..." :transport ...}
```

Downloads use the `:token` option when supplied, otherwise `:hf-token`, then
the conventional `HF_TOKEN` environment variable. Hugging Face file metadata
provides the SHA-256 hashes used to verify each download before atomic cache
finalization. Interrupted downloads continue from the local `.part` file with
an HTTP Range request, and a per-model file lock prevents concurrent writers.
Checksum failures remove the partial file and refuse to install it.

Quantized exports: `:variant :quantized` tries the common quantized paths in
this order: `onnx/model_quantized.onnx`, `onnx/model_qint8_avx512_vnni.onnx`,
`onnx/model_int8.onnx`, and `model_quantized.onnx`. A string `:variant` names an
explicit repo-relative `.onnx` path, for example `"onnx/model_q4.onnx"`. Each
variant is cached in its own subdirectory. Thus a full-precision copy and a
quantized copy of the same model can both exist:

```clojure
(hub/fetch-model "sentence-transformers/all-MiniLM-L6-v2" {:variant :quantized})
```

To fetch a model manually:

```bash
mkdir -p models/all-MiniLM-L6-v2
curl -fL -o models/all-MiniLM-L6-v2/model.onnx \
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx
curl -fL -o models/all-MiniLM-L6-v2/tokenizer.json \
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json
```

Compatible model families include all-MiniLM, all-mpnet, BGE, GTE, and E5. A
model is compatible if its ONNX graph takes `input_ids`, `attention_mask`, and
`token_type_ids`, and outputs token embeddings or a pre-pooled sentence
embedding.

E5-style models expect instruction prefixes; pass `:prefix` per call:

```clojure
(emb/embed model "how do I bake bread" {:prefix "query: "})
(emb/embed-batch model documents {:prefix "passage: "})
```
