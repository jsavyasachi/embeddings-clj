# Contributing to embeddings-clj

Thanks for your interest in improving `embeddings-clj`. Bug reports, fixes, and
focused feature contributions are all welcome.

## Before you start

- For anything beyond a trivial fix, **open an issue first**. This lets us agree
  on the approach before you invest time.
- Check existing issues and pull requests to avoid duplicate work.

## Project layout

This is a single `deps.edn` library. Source files are under `src/embeddings/`:

| Namespace | Purpose |
|---|---|
| `embeddings.core` | public API: `load-model` / `with-model`, `embed`, `embed-batch`, `dimension`, `close` |
| `embeddings.pooling` | mask-aware `:mean` / `:cls` / `:max` pooling over token embeddings |
| `embeddings.math` | `dot`, `norm`, `l2-normalize`, `cosine-similarity` on primitive `float[]` |

Hot paths operate on primitive float arrays and must stay reflection-free.
Expected failures throw `ex-info` with an `:embeddings/error` key.

## Building and testing

This project needs JDK 17 or later.

```bash
clojure -M:test            # full suite (Kaocha)
clojure -M:1.11:test       # Clojure 1.11 matrix cell
clojure -M:1.12:test       # Clojure 1.12 matrix cell
clojure -T:build jar       # build a jar
```

Unit tests use small deterministic ONNX fixtures. The repository does not
contain binary files:

```bash
python3 dev/gen_fixture.py   # requires `pip install onnx`; tests skip cleanly without it
./dev/fetch-model.sh         # optional ~90MB all-MiniLM-L6-v2 download for the
clojure -M:test --profile :integration --focus-meta :integration   # real-model suite
```

Requirements for a mergeable change:

- **Tests first.** Add or update tests for the behavior you change. For a bug
  fix, include a regression test that fails before your fix and passes after it.
- **Green build.** `clojure -M:test` passes and `src` compiles with **zero**
  reflection warnings (`*warn-on-reflection*` is on).
- **No scope creep.** Keep each pull request to one logical change.

## Commits and pull requests

- Follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:` …).
- Keep the subject in the imperative mood and under ~72 characters.
- Update `CHANGELOG.md` when your change is user-visible.
- Rebase on the latest `main` before you open the pull request.

## License

If you contribute, you agree to license your contributions under the Eclipse
Public License 2.0, the same license as this project. See `LICENSE`.
