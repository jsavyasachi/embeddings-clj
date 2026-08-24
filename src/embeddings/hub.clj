(ns embeddings.hub
  "Download sentence-transformers ONNX exports from the Hugging Face Hub to a
  local cache for `embeddings.core/load-model`."
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.string :as str])
  (:import (java.math BigInteger)
           (java.net URI)
           (java.net.http HttpClient HttpClient$Redirect HttpRequest
                          HttpRequest$BodyPublishers HttpResponse$BodyHandlers)
           (java.nio.channels FileChannel)
           (java.nio.file Files StandardCopyOption StandardOpenOption)
           (java.security MessageDigest)))

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

(defn- environment-token []
  (System/getenv "HF_TOKEN"))

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

(defn- ^HttpClient default-http-client []
  (-> (HttpClient/newBuilder)
      ;; HF resolve URLs redirect to a content-serving URL. HttpClient's
      ;; default is NEVER, so opt into the normal safe redirect policy.
      (.followRedirects HttpClient$Redirect/NORMAL)
      (.build)))

(defn- default-transport [{:keys [url method headers]}]
  (let [builder (HttpRequest/newBuilder (URI/create ^String url))]
    (doseq [[name value] headers]
      (.header builder ^String name ^String value))
    (let [^java.net.http.HttpResponse resp (.send (default-http-client)
                      (.method builder ^String method
                              (HttpRequest$BodyPublishers/ofString ""))
                      (HttpResponse$BodyHandlers/ofByteArray))]
      {:status (.statusCode resp)
       :headers (into {} (.map (.headers resp)))
       :body (.body resp)})))

(defn- auth-token [opts]
  (or (:token opts) (:hf-token opts) (environment-token)))

(defn- request-headers [opts extra]
  (cond-> (merge {"Accept" "application/json"} extra)
    (auth-token opts) (assoc "Authorization" (str "Bearer " (auth-token opts)))))

(defn- manifest-hashes
  "Return genuine LFS content hashes from a `/tree` response.

  Ordinary Git-blob files expose a blob ID, not a content hash, so they are
  deliberately absent from this map and their downloaded bytes are not
  checksum-verified."
  [body]
  (into {} (keep (fn [entry]
                   (when-let [hash (get-in entry ["lfs" "sha256"])]
                     [(get entry "path") hash])))
            (json/read-str (if (bytes? body) (String. ^bytes body "UTF-8") body))))

(defn- next-page-url [headers]
  (some (fn [[name values]]
          (when (= "link" (str/lower-case (str name)))
            (some (fn [value]
                    (when-let [[_ url] (re-find #"(?i)<([^>]+)>\s*;\s*rel=\"?next\"?" value)]
                      url))
                  (if (sequential? values) values [values]))))
        headers))

(defn- manifest-entries [request! opts url]
  (loop [url url
         entries []]
    (let [response (request! {:url url
                              :method "GET"
                              :headers (request-headers opts {})})
          status (long (:status response))]
      (when-not (<= 200 status 299)
        (throw (ex-info (str "Manifest request failed with HTTP " status ": " url)
                        {:embeddings/error :manifest-failed
                         :status status
                         :url url})))
      (let [page (json/read-str (if (bytes? (:body response))
                                  (String. ^bytes (:body response) "UTF-8")
                                  (:body response)))
            entries (into entries page)]
        (if-let [next-url (next-page-url (:headers response))]
          (recur next-url entries)
          entries)))))

(defn- sha256-file ^String [^java.io.File file]
  (format "%064x" (BigInteger. 1 (.digest (doto (MessageDigest/getInstance "SHA-256")
                                           (.update (Files/readAllBytes (.toPath file))))))))

(defn- atomic-move! [^java.io.File source ^java.io.File dest]
  (try
    (Files/move (.toPath source) (.toPath dest)
                (into-array StandardCopyOption
                            [StandardCopyOption/ATOMIC_MOVE StandardCopyOption/REPLACE_EXISTING]))
    (catch java.nio.file.AtomicMoveNotSupportedException _
      (Files/move (.toPath source) (.toPath dest)
                  (into-array StandardCopyOption [StandardCopyOption/REPLACE_EXISTING])))))

(defonce ^:private download-locks (atom {}))

(defn- with-download-lock [^java.io.File root f]
  (let [key (.getAbsolutePath root)
        mutex (get (swap! download-locks #(if (contains? % key) % (assoc % key (Object.))))
                   key)
        lock-file (io/file root ".download.lock")]
    (locking mutex
      (io/make-parents lock-file)
      (with-open [^FileChannel channel (FileChannel/open (.toPath lock-file)
                                                        (into-array StandardOpenOption
                                                                    [StandardOpenOption/CREATE
                                                                     StandardOpenOption/WRITE]))
                  lock (.lock channel)]
        (f)))))

(defn- secure-download! [request! opts ^String url ^java.io.File dest expected]
  (let [part (io/file (str (.getPath dest) ".part"))
        size (if (.exists part) (.length part) 0)
        response (request! {:url url
                            :method "GET"
                            :headers (request-headers opts
                                                      (cond-> {} (pos? size)
                                                        (assoc "Range" (str "bytes=" size "-"))))})
        status (long (:status response))]
    (if (= 404 status)
      false
      (do
        (when-not (contains? #{200 206} status)
          (throw (ex-info (str "Download failed with HTTP " status ": " url)
                          {:embeddings/error :download-failed :status status :url url})))
        (io/make-parents part)
        (let [^bytes bytes (if (bytes? (:body response))
                             (:body response)
                             (.getBytes ^String (:body response) "UTF-8"))
              append? (and (= 206 status) (pos? size))]
          (with-open [out (java.io.FileOutputStream. part append?)]
            (.write ^java.io.FileOutputStream out ^bytes bytes)))
        (if (nil? expected)
          ;; Non-LFS files have no content checksum in the HF manifest. Their
          ;; Git blob ID is not equivalent, so cache them without verification.
          (do (atomic-move! part dest) true)
          (if (= expected (sha256-file part))
            (do (atomic-move! part dest) true)
            (do (.delete part)
                (throw (ex-info (str "Checksum mismatch for " url)
                                {:embeddings/error :checksum-mismatch
                                 :url url :expected expected})))))))))

(defn- fetch-file!
  "Try each url in order until one succeeds. Throw if all urls fail."
  [urls ^java.io.File dest download!]
  (io/make-parents dest)
  (or (some (fn [url] (when (download! url dest) dest)) urls)
      (throw (ex-info (str "No download source succeeded for " (.getName dest))
                      {:embeddings/error :download-failed
                       :urls (vec urls)}))))

(defn- fetch-model*
  ([model-id opts download!]
   (fetch-model* model-id opts download! nil))
  ([model-id {:keys [cache-dir revision variant] :as opts} download! request!]
   (validate-model-id! model-id)
   (let [revision (or revision "main")
         paths (model-paths variant)
         ^java.io.File root (model-root cache-dir model-id revision variant)
         model-file (io/file root "model.onnx")
         tokenizer-file (io/file root "tokenizer.json")]
     (if request!
       (with-download-lock root
         (fn []
           (let [manifest-url (str "https://huggingface.co/api/models/"
                                   model-id "/tree/" revision "?recursive=true")
                 manifest-body (json/write-str (manifest-entries request! opts manifest-url))
                 hashes (manifest-hashes manifest-body)
                 download (fn [candidate-paths dest]
                            (or (some (fn [path]
                                        (let [expected (get hashes path)]
                                          (when (secure-download! request! opts
                                                                   (resolve-url model-id revision path)
                                                                   dest expected)
                                            dest))) candidate-paths)
                                (throw (ex-info "No download source succeeded"
                                                {:embeddings/error :download-failed
                                                 :urls (vec candidate-paths)}))))]
             (doseq [[candidate-paths ^java.io.File dest]
                     [[paths model-file] [["tokenizer.json"] tokenizer-file]]]
               (when (and (.exists dest) (pos? (.length dest)))
                 (when-let [expected (some #(get hashes %) candidate-paths)]
                   (when-not (= expected (sha256-file dest))
                     (throw (ex-info (str "Checksum mismatch for " (.getPath dest))
                                     {:embeddings/error :checksum-mismatch
                                      :path (.getPath dest) :expected expected}))))))
             (when-not (and (.exists model-file) (pos? (.length model-file)))
               (download paths model-file))
             (when-not (and (.exists tokenizer-file) (pos? (.length tokenizer-file)))
               (download ["tokenizer.json"] tokenizer-file))
             (.getPath root))))
       (do
         (when-not (and (.exists model-file) (pos? (.length model-file)))
           (fetch-file! (mapv #(resolve-url model-id revision %) paths) model-file download!))
         (when-not (and (.exists tokenizer-file) (pos? (.length tokenizer-file)))
           (fetch-file! [(resolve-url model-id revision "tokenizer.json")] tokenizer-file download!))
         (.getPath root))))))

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
  ([model-id opts]
   (let [opts (or opts {})]
     (fetch-model* model-id opts nil (or (:transport opts) default-transport)))))
