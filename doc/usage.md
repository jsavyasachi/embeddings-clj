# Usage

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

