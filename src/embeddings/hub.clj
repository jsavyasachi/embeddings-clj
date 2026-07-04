(ns embeddings.hub
  "Download sentence-transformers ONNX exports from the Hugging Face Hub into
  a local cache, for use with `embeddings.core/load-model`."
  (:require [clojure.java.io :as io])
  (:import (java.net URI)
           (java.net.http HttpClient HttpClient$Redirect HttpRequest
                          HttpResponse$BodyHandlers)
           (java.nio.file Files StandardCopyOption)))

(set! *warn-on-reflection* true)

(def ^:private model-id-pattern
  ;; owner/name, each segment word chars plus . and -, no traversal
  #"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

(defn- validate-model-id! [model-id]
  (when-not (and (string? model-id) (re-matches model-id-pattern model-id))
    (throw (ex-info (str "Invalid Hugging Face model id: " (pr-str model-id))
                    {:embeddings/error :invalid-model-id :model-id model-id}))))

(defn- resolve-url ^String [model-id revision path]
  (str "https://huggingface.co/" model-id "/resolve/" revision "/" path))

(defn- default-cache-dir []
  (io/file (System/getProperty "user.home") ".cache" "embeddings-clj"))

(defn- http-download!
  "Download url to dest via a temp file. Returns true on HTTP 200, false on
  404 (so callers can try a fallback url), throws on anything else."
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
  "Try each url in order until one succeeds; throws when all miss."
  [urls ^java.io.File dest download!]
  (io/make-parents dest)
  (or (some (fn [url] (when (download! url dest) dest)) urls)
      (throw (ex-info (str "No download source succeeded for " (.getName dest))
                      {:embeddings/error :download-failed
                       :urls (vec urls)}))))

(defn- fetch-model*
  [model-id {:keys [cache-dir revision]} download!]
  (validate-model-id! model-id)
  (let [revision (or revision "main")
        root (io/file (or cache-dir (default-cache-dir)) model-id)
        model-file (io/file root "model.onnx")
        tokenizer-file (io/file root "tokenizer.json")]
    (when-not (and (.exists model-file) (pos? (.length model-file)))
      (fetch-file! [(resolve-url model-id revision "onnx/model.onnx")
                    (resolve-url model-id revision "model.onnx")]
                   model-file download!))
    (when-not (and (.exists tokenizer-file) (pos? (.length tokenizer-file)))
      (fetch-file! [(resolve-url model-id revision "tokenizer.json")]
                   tokenizer-file download!))
    (.getPath root)))

(defn fetch-model
  "Download `model-id`'s ONNX export (`onnx/model.onnx`, falling back to
  `model.onnx`) and `tokenizer.json` from the Hugging Face Hub into a local
  cache, returning the model directory path for
  `embeddings.core/load-model`. Files already in the cache are not
  re-downloaded.

  Options: `:cache-dir` (default `~/.cache/embeddings-clj`) and `:revision`
  (default \"main\").

  Errors are `ex-info` keyed `:embeddings/error`
  (`:invalid-model-id`, `:download-failed`)."
  ([model-id] (fetch-model model-id nil))
  ([model-id opts] (fetch-model* model-id opts http-download!)))
