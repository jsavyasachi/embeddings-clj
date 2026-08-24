# Changelog

## [0.6.0] - 2026-08-24

### Added

- Add batch dot-product and cosine similarity-search utilities with top-k
  ranking and a brute-force in-memory vector index in the new
  `embeddings.search` namespace.
- Add provider-specific request options: OpenAI `encoding_format`, Cohere
  `truncate` and token controls, and Voyage `output_dtype`.
- Add strict response-shape validation for hosted providers, including
  embedding-count and dimension consistency checks with typed errors for
  malformed payloads.
- Add ONNX session performance controls for threads, optimization, memory,
  profiling, and logging.
- Add ONNX input/output dtype compatibility for int32 inputs and float16 or
  float64 outputs inferred from session metadata.
- Add execution-provider discovery that advertises only providers shipped by
  the pinned ONNX Runtime artifact: CPU, CoreML, and WebGPU.
- Add resilient hosted-provider transport with timeouts, exponential retry
  with jitter, and `Retry-After` handling.

### Fixed

- Parse both integer seconds and RFC 1123 HTTP-date values in `Retry-After`.
- Retry transient transport-level exceptions, not only retryable HTTP status
  codes.
- Follow Hugging Face resolve redirects during downloads.
- Run checksum verification only for files with a genuine LFS content hash,
  rather than fabricating one from a non-LFS git blob ID.
- Check response status and handle pagination when listing manifests.

## [0.5.1] - 2026-08-17

### Fixed

- The model cache path now includes the `revision`, so pinning a revision no
  longer silently reuses another revision's cached files. The default `main`
  revision keeps its existing location.

## [0.5.0] - 2026-08-01
### Changed
- Move JSON handling to `org.clojure/data.json`.
### Fixed
- Remove the undeclared transitive Gson dependency and fix the cljdoc API build.

## [0.4.0] - 2026-07-16
### Added
- Add a shared embedding-provider protocol with hosted OpenAI, Cohere, and Voyage adapters using JDK HttpClient and injectable transports.
- Support named ONNX output selection, `position_ids`, and custom input schemas.
- Add Matryoshka `:output-dimensions` truncation.
### Fixed
- Close native model handles when models are closed or loading fails.

## [0.3.4] - 2026-07-16
### Changed
- Update tokenizers-clj to 0.1.3.
### Fixed
- Honor SentenceTransformers model configuration (`modules.json`, `config.json`, pooling config) for pooling mode, normalization, and max sequence length instead of hardcoded defaults.

## [0.3.3] - 2026-07-16
### Fixed
- Honor SentenceTransformers model configuration (`modules.json`, `config.json`, pooling config) for pooling mode, normalization, and max sequence length instead of hardcoded defaults.

## [0.3.2] - 2026-07-09
### Changed
- Reorganized the README into a cljdoc article tree under `doc/` (Getting a model, Usage). Documentation content is unchanged; no API changes.

All notable changes to this project are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.1] - 2026-07-09
### Fixed
- POM now includes the project description, homepage URL, and full SCM connection metadata, so Clojars shows a description/homepage and cljdoc has complete source-link data.
