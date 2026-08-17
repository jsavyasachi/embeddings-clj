(ns embeddings.hub
  "Download sentence-transformers ONNX exports from the Hugging Face Hub to a
  local cache for `embeddings.core/load-model`."
  (:require [clojure.java.io :as io]
            [clojure.string :as str])
  (:import (java.net URI)
           (java.net.http HttpClient HttpClient$Redirect HttpRequest
                          HttpResponse$BodyHandlers)
           (java.nio.file Files StandardCopyOption)))

(set! *warn-on-reflection* true)

(def ^:private model-id-pattern
  ;; owner/name: each segment has word characters, . or -; no traversal
  #"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

(defn- validate-model-id! [model-id]
  (when-not (and (string? model-id) (re-matches model-id-pattern model-id))
    (throw (ex-info (str "Invalid Hugging Face model id: " (pr-str model-id))
                    {:embeddings/error :invalid-model-id :model-id model-id}))))

(defn- resolve-url ^String [model-id revision path]
  (str "https://huggingface.co/" model-id "/resolve/" revision "/" path))

(defn- default-cache-dir []
  (io/file (System/getProperty "user.home") ".cache" "embeddings-clj"))

(def ^:private quantized-model-paths
  ["onnx/model_quantized.onnx"
   "onnx/model_qint8_avx512_vnni.onnx"
   "onnx/model_int8.onnx"
   "model_quantized.onnx"])

(defn- validate-variant! [variant]
  (when (and (string? variant)
             (or (not (str/ends-with? variant ".onnx"))
                 (str/includes? variant "..")
                 (str/starts-with? variant "/")))
    (throw (ex-info (str "Invalid model variant: " (pr-str variant))
                    {:embeddings/error :invalid-variant :variant variant}))))

(defn- variant-slug ^String [^String path]
  (let [filename (.getName (io/file path))
        basename (subs filename 0 (- (count filename) (count ".onnx")))]
    (str/replace basename #"[^A-Za-z0-9._-]" "")))

(defn- model-paths [variant]
  (validate-variant! variant)
  (cond
    (nil? variant) ["onnx/model.onnx" "model.onnx"]
    (= :quantized variant) quantized-model-paths
    (string? variant) [variant]
    :else (throw (ex-info (str "Invalid model variant: " (pr-str variant))
                          {:embeddings/error :invalid-variant :variant variant}))))

(defn- model-root [cache-dir model-id revision variant]
  (let [model-root (io/file (or cache-dir (default-cache-dir)) model-id)
        ;; Keep the historical default branch location, while isolating pins.
        root (if (= revision "main")
               model-root
               (io/file model-root revision))]
    (cond
      (nil? variant) root
      (= :quantized variant) (io/file root "quantized")
      (string? variant) (io/file root (variant-slug variant))
      :else root)))

(defn- http-download!
  "Download url to dest through a temp file. Return true for HTTP 200. Return
  false for HTTP 404 so callers can try a fallback url. Throw for other status codes."
  [^String url ^java.io.File dest]
  (let [client (-> (HttpClient/newBuilder)
                   (.followRedirects HttpClient$Redirect/ALWAYS)
                   (.build))
        req (-> (HttpRequest/newBuilder (URI/create url)) (.GET) (.build))
        tmp (java.io.File/createTempFile "embeddings-clj" ".part" (.getParentFile dest))
        resp (.send client req (HttpResponse$BodyHandlers/ofFile (.toPath ^java.io.File tmp)))
        status (.statusCode resp)]
    (case (int status)
      200 (do (Files/move (.toPath ^java.io.File tmp) (.toPath dest)
                          (into-array StandardCopyOption
                                      [StandardCopyOption/REPLACE_EXISTING]))
              true)
      404 (do (.delete ^java.io.File tmp) false)
      (do (.delete ^java.io.File tmp)
          (throw (ex-info (str "Download failed with HTTP " status ": " url)
                          {:embeddings/error :download-failed
                           :status status
                           :url url}))))))

(defn- fetch-file!
  "Try each url in order until one succeeds. Throw if all urls fail."
  [urls ^java.io.File dest download!]
  (io/make-parents dest)
  (or (some (fn [url] (when (download! url dest) dest)) urls)
      (throw (ex-info (str "No download source succeeded for " (.getName dest))
                      {:embeddings/error :download-failed
                       :urls (vec urls)}))))

(defn- fetch-model*
  [model-id {:keys [cache-dir revision variant]} download!]
  (validate-model-id! model-id)
  (let [revision (or revision "main")
        paths (model-paths variant)
        ^java.io.File root (model-root cache-dir model-id revision variant)
        model-file (io/file root "model.onnx")
        tokenizer-file (io/file root "tokenizer.json")]
    (when-not (and (.exists model-file) (pos? (.length model-file)))
      (fetch-file! (mapv #(resolve-url model-id revision %) paths)
                   model-file download!))
    (when-not (and (.exists tokenizer-file) (pos? (.length tokenizer-file)))
      (fetch-file! [(resolve-url model-id revision "tokenizer.json")]
                   tokenizer-file download!))
    (.getPath root)))

(defn fetch-model
  "Download the ONNX export for `model-id` and `tokenizer.json` from the Hugging
  Face Hub to a local cache. Try `onnx/model.onnx`, then `model.onnx`. Return
  the model directory path for `embeddings.core/load-model`. Do not download
  files that are already in the cache.

  Options: `:cache-dir` (default `~/.cache/embeddings-clj`), `:revision`
  (default \"main\"), and `:variant`. `:variant :quantized` tries common
  quantized ONNX paths. It stores them in a `quantized` cache subdir. A string
  `:variant` is an explicit repo-relative `.onnx` path. The library caches it
  in a subdir derived from the path.

  Errors are `ex-info` keyed `:embeddings/error`
  (`:invalid-model-id`, `:invalid-variant`, `:download-failed`)."
  ([model-id] (fetch-model model-id nil))
  ([model-id opts] (fetch-model* model-id opts http-download!)))
