# embeddings-clj

[![Clojars Project](https://img.shields.io/clojars/v/net.clojars.savya/embeddings-clj.svg)](https://clojars.org/net.clojars.savya/embeddings-clj)
[![cljdoc](https://cljdoc.org/badge/net.clojars.savya/embeddings-clj)](https://cljdoc.org/d/net.clojars.savya/embeddings-clj/CURRENT)
[![test](https://github.com/jsavyasachi/embeddings-clj/actions/workflows/test.yml/badge.svg)](https://github.com/jsavyasachi/embeddings-clj/actions/workflows/test.yml)

Text embeddings for Clojure with local sentence-transformers ONNX exports or
hosted OpenAI, Cohere, and Voyage models.

## Stack

<a href="https://clojure.org"><img src="https://img.shields.io/badge/Clojure-5881D8?style=flat&logo=clojure&logoColor=fff" alt="Clojure" /></a>
<a href="https://onnxruntime.ai"><img src="https://img.shields.io/badge/ONNX%20Runtime-5A5AB5?style=flat&logo=onnx&logoColor=fff" alt="ONNX Runtime" /></a>
<a href="https://github.com/jsavyasachi/tokenizers-clj"><img src="https://img.shields.io/badge/tokenizers--clj-5881D8?style=flat&logo=clojure&logoColor=fff" alt="tokenizers-clj" /></a>

## Installation

deps.edn:

```clojure
net.clojars.savya/embeddings-clj {:mvn/version "0.7.0"}
```

Leiningen:

```clojure
[net.clojars.savya/embeddings-clj "0.7.0"]
```

The library parses JSON with the declared `org.clojure/data.json` dependency. It
does not depend on Gson.

## Providers

Local and hosted models implement `embeddings.core/EmbeddingProvider`, with
`embed`, `embed-batch`, and `dimension` operations.

```clojure
(require '[embeddings.core :as emb]
         '[embeddings.providers :as providers])

(def openai
  (providers/openai {:api-key (System/getenv "OPENAI_API_KEY")
                     :model "text-embedding-3-small"
                     :dimensions 512}))

(emb/embed openai "A sentence to embed")

(def cohere
  (providers/cohere {:api-key (System/getenv "COHERE_API_KEY")
                     :model "embed-v4.0"
                     :input-type "search_document"}))

(def voyage
  (providers/voyage {:api-key (System/getenv "VOYAGE_API_KEY")
                     :model "voyage-3-large"
                     :input-type "query"}))
```

Hosted provider options include `:api-key`, `:model`, `:dimensions`,
`:batch-size` (default `128`), `:url` for an endpoint override, and
`:transport` for an injectable request function. Cohere and Voyage also accept
`:input-type`. OpenAI accepts `:encoding-format` (`"float"` or `"base64"`),
Cohere accepts `:truncate` and `:max-tokens`, and Voyage accepts
`:output-dtype`. Use `:headers` to add or override HTTP headers for hosted calls.
Requests use a 10-second connection timeout and 60-second request timeout by
default; configure these with `:connect-timeout-ms` and
`:request-timeout-ms`. Transient HTTP failures are retried up to three times
with exponential backoff and jitter. Configure this with `:max-retries`,
`:retry-base-delay-ms`, `:retry-max-delay-ms`, and `:retry-jitter` (or inject
`:sleep-fn` for tests). A numeric `Retry-After` response header on a 429 is
honored in preference to the calculated delay. The transport receives
`{:url :method :headers :body}` and must return `{:status :body}`, optionally
with `:headers` for retry handling.

## Similarity search

The `embeddings.search` namespace ranks in-memory `float[]` embeddings without
network or model access. It supports dot-product or cosine scoring, stable
top-k ranking, predicate filtering, and reusable brute-force indexes:

```clojure
(require '[embeddings.search :as search])

(search/search query candidates {:metric :cosine
                                 :k 5
                                 :predicate #(not (:deleted? %))
                                 :vector-fn :embedding})

(def index (search/build-index candidates :embedding))
(search/query index query {:metric :cosine :k 5})
```

Candidates must contain `float[]` vectors (or be `float[]` vectors themselves).
Ties retain candidate order; dimensions must match, and `k` must be a
non-negative integer.

## Local model options

`embeddings.core/load-model` accepts pooling, normalization,
maximum-length, and execution-provider options, plus:

- `embeddings.core/execution-provider-discovery` reports the providers exposed
  by the current ONNX Runtime, the providers this library can configure, and
  unresolved provider blockers. Configurable providers include CPU, CoreML,
  and WebGPU.

- `:output-name`: select a named ONNX graph output.
- `:input-schema`: map custom ONNX input names to an encoded source keyword or
  `{:source keyword :pad-value number}`. Built-in inputs include `input_ids`,
  `attention_mask`, `token_type_ids`, and `position_ids`.
- `:output-dimensions`: truncate Matryoshka embeddings to a positive dimension
  no larger than the model output. Truncation happens before normalization.

## Documentation

- [Getting a model](doc/getting-a-model.md)
- [Usage](doc/usage.md)

## Running tests

```bash
clojure -M:test
```

The unit suite runs against tiny deterministic ONNX fixtures. Generate them with
`python3 dev/gen_fixture.py`, which needs `pip install onnx`. If the fixtures are
absent, the tests skip and the suite stays green.

The optional integration suite uses a real all-MiniLM-L6-v2 model:

```bash
./dev/fetch-model.sh          # ~90MB download from HuggingFace
clojure -M:test --focus-meta :integration
```

## License

Copyright © 2026 Savyasachi.

Distributed under the [Eclipse Public License 2.0](https://www.eclipse.org/legal/epl-2.0/).
