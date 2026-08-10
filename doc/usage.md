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

Options for `load-model` with default values:

| option | default | meaning |
|---|---|---|
| `:pooling` | `:mean` | `:mean` (mask-weighted), `:mean-sqrt-len` (sum / sqrt of token count), `:cls`, or `:max` over token embeddings |
| `:normalize?` | `true` | L2-normalize output vectors (unit length, ready for cosine) |
| `:max-length` | `512` | truncate inputs to this many tokens |
| `:execution-providers` | none (CPU) | ONNX Runtime execution providers to try, e.g. `[:coreml]` or `[{:provider :cuda :device-id 0}]`; also `:rocm`, `:tensorrt`, `:directml`, `:xnnpack` |

Execution providers need an onnxruntime build that includes them. The default
`com.microsoft.onnxruntime/onnxruntime` artifact is CPU-only, and CUDA needs
`onnxruntime_gpu`. If you request a provider that the runtime does not supply,
the library throws `ex-info` with
`{:embeddings/error :execution-provider-unavailable}`.

Some ONNX graphs already output a pooled `[batch, hidden]` sentence embedding.
The library detects these models automatically and ignores `:pooling`.

`embeddings.math` supplies the small vector functions: `dot`, `norm`,
`l2-normalize`, and `cosine-similarity`. All of them operate on primitive
`float[]`.

Errors are `ex-info` maps keyed `:embeddings/error`
(`:model-not-found`, `:tokenizer-not-found`, `:model-closed`,
`:unsupported-input`, `:unsupported-output`, `:dim-mismatch`).
