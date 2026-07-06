# embeddings-clj

[![Clojars Project](https://img.shields.io/clojars/v/net.clojars.savya/embeddings-clj.svg)](https://clojars.org/net.clojars.savya/embeddings-clj)
[![cljdoc](https://cljdoc.org/badge/net.clojars.savya/embeddings-clj)](https://cljdoc.org/d/net.clojars.savya/embeddings-clj/CURRENT)
[![test](https://github.com/jsavyasachi/embeddings-clj/actions/workflows/test.yml/badge.svg)](https://github.com/jsavyasachi/embeddings-clj/actions/workflows/test.yml)

Local text-embedding inference for Clojure. Runs sentence-transformers ONNX
exports on the JVM via ONNX Runtime.

## Stack

<a href="https://clojure.org"><img src="https://img.shields.io/badge/Clojure-5881D8?style=flat&logo=clojure&logoColor=fff" alt="Clojure" /></a>
<a href="https://onnxruntime.ai"><img src="https://img.shields.io/badge/ONNX%20Runtime-5A5AB5?style=flat&logo=onnx&logoColor=fff" alt="ONNX Runtime" /></a>
<a href="https://github.com/jsavyasachi/tokenizers-clj"><img src="https://img.shields.io/badge/tokenizers--clj-5881D8?style=flat&logo=clojure&logoColor=fff" alt="tokenizers-clj" /></a>

## Installation

deps.edn:

```clojure
net.clojars.savya/embeddings-clj {:mvn/version "0.3.0"}
```

Leiningen:

```clojure
[net.clojars.savya/embeddings-clj "0.3.0"]
```

## Getting a model

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

## Usage

```clojure
(require '[embeddings.core :as emb]
         '[embeddings.math :as emb.math])

;; load-model returns an AutoCloseable-style handle; with-model scopes it
(emb/with-model [model "models/all-MiniLM-L6-v2" {:pooling :mean :normalize? true}]
  (emb/dimension model)                    ;; => 384
  (let [a (emb/embed model "A cat sits on the mat")
        b (emb/embed model "A kitten rests on the rug")
        c (emb/embed model "The stock market crashed today")]
    (emb.math/cosine-similarity a b)       ;; => ~0.6+ (similar)
    (emb.math/cosine-similarity a c)))     ;; => ~0.1  (unrelated)

;; batch (padding is attention-mask aware; results match single embeds)
(emb/with-model [model "models/all-MiniLM-L6-v2"]
  (emb/embed-batch model ["first text" "second text" "third text"]))
;; => [float[384] float[384] float[384]]
```

Options to `load-model` (defaults shown):

| option | default | meaning |
|---|---|---|
| `:pooling` | `:mean` | `:mean` (mask-weighted), `:mean-sqrt-len` (sum / sqrt of token count), `:cls`, or `:max` over token embeddings |
| `:normalize?` | `true` | L2-normalize output vectors (unit length, ready for cosine) |
| `:max-length` | `512` | truncate inputs to this many tokens |
| `:execution-providers` | none (CPU) | ONNX Runtime execution providers to try, e.g. `[:coreml]` or `[{:provider :cuda :device-id 0}]`; also `:rocm`, `:tensorrt`, `:directml`, `:xnnpack` |

Execution providers require an onnxruntime build that bundles them (the
default `com.microsoft.onnxruntime/onnxruntime` artifact is CPU-only; CUDA
needs `onnxruntime_gpu`). Requesting a provider the runtime lacks throws
`ex-info` with `{:embeddings/error :execution-provider-unavailable}`.

Models whose ONNX graph already outputs a pooled `[batch, hidden]` sentence
embedding are detected automatically and used as-is (`:pooling` is ignored).

`embeddings.math` ships the small vector toolkit: `dot`, `norm`,
`l2-normalize`, `cosine-similarity` - all on primitive `float[]`.

Errors are `ex-info` maps keyed `:embeddings/error`
(`:model-not-found`, `:tokenizer-not-found`, `:model-closed`,
`:unsupported-input`, `:unsupported-output`, `:dim-mismatch`).

## Running tests

```bash
clojure -M:test
```

The unit suite runs against tiny deterministic ONNX fixtures generated by
`python3 dev/gen_fixture.py` (requires `pip install onnx`; tests skip cleanly
when fixtures are absent, so the suite is green either way).

The opt-in integration suite exercises a real all-MiniLM-L6-v2 model:

```bash
./dev/fetch-model.sh          # ~90MB download from HuggingFace
clojure -M:test --focus-meta :integration
```

## License

Copyright © 2026 Savyasachi.

Distributed under the [Eclipse Public License 2.0](https://www.eclipse.org/legal/epl-2.0/).
