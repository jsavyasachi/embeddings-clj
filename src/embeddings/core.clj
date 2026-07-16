(ns embeddings.core
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [embeddings.math :as math]
            [embeddings.pooling :as pooling]
            [tokenizers.core :as tokenizers])
  (:import (ai.onnxruntime NodeInfo OnnxTensor OnnxValue OrtEnvironment OrtSession
                           OrtException TensorInfo)
           (ai.onnxruntime OrtSession$Result)
           (ai.onnxruntime OrtSession$SessionOptions)
           (com.google.gson JsonElement JsonObject JsonParser)
           (java.io File)
           (java.util Collections HashMap Map)))

(set! *warn-on-reflection* true)

(def ^:private default-opts
  {:pooling :mean
   :normalize? true
   :max-length 512})

(defprotocol EmbeddingProvider
  "A source of fixed-width text embeddings."
  (embed [provider text] [provider text opts])
  (embed-batch [provider texts] [provider texts opts])
  (dimension [provider]))

(declare embed-batch* model-dimension)

(defrecord LocalModel [session tokenizer opts input-names output-name closed?]
  EmbeddingProvider
  (embed [model text]
    (embed model text nil))
  (embed [model text opts]
    (first (embed-batch model [text] opts)))
  (embed-batch [model texts]
    (embed-batch model texts nil))
  (embed-batch [model texts {:keys [prefix]}]
    (embed-batch* model (if prefix (mapv #(str prefix %) texts) texts)))
  (dimension [model]
    (model-dimension model)))

(defn- model-error
  [message data]
  (ex-info message data))

(defn- file-in
  ^File
  [model-dir child]
  (let [^File dir (if (instance? File model-dir)
                    model-dir
                    (File. ^String model-dir))]
    (File. dir ^String child)))

(defn- read-json
  ^JsonElement
  [^File file]
  (when (.isFile file)
    (with-open [reader (io/reader file)]
      (JsonParser/parseReader reader))))

(defn- json-object
  ^JsonObject
  [^JsonElement element]
  (when (and element (.isJsonObject element))
    (.getAsJsonObject element)))

(defn- json-value
  ^JsonElement
  [^JsonObject object ^String key]
  (when (and object (.has object key))
    (.get object key)))

(defn- json-string
  [^JsonObject object ^String key]
  (some-> (json-value object key) .getAsString))

(defn- json-boolean
  [^JsonObject object ^String key]
  (some-> (json-value object key) .getAsBoolean))

(defn- json-long
  [^JsonObject object ^String key]
  (some-> (json-value object key) .getAsLong))

(defn- modules
  [^JsonElement modules-json]
  (when (and modules-json (.isJsonArray modules-json))
    (map #(.getAsJsonObject ^JsonElement %)
         (iterator-seq (.iterator (.getAsJsonArray modules-json))))))

(defn- module-type?
  [^JsonObject module ^String suffix]
  (some-> (json-string module "type") (str/ends-with? suffix)))

(defn- pooling-module-config
  ^File
  [model-dir model-modules]
  (when-let [path (some #(when (module-type? % ".Pooling")
                           (json-string % "path"))
                        model-modules)]
    (let [^File root (.getCanonicalFile (file-in model-dir ""))
          ^File module-dir (.getCanonicalFile (File. root ^String path))]
      (when-not (.startsWith (.toPath module-dir) (.toPath root))
        (throw (model-error "pooling module path escapes model directory"
                            {:embeddings/error :invalid-model-config
                             :path path})))
      (File. module-dir "config.json"))))

(def ^:private pooling-config-keys
  [[:cls "pooling_mode_cls_token"]
   [:mean "pooling_mode_mean_tokens"]
   [:max "pooling_mode_max_tokens"]
   [:mean-sqrt-len "pooling_mode_mean_sqrt_len_tokens"]])

(defn- configured-pooling
  [^JsonObject config]
  (or (some-> (json-string config "pooling_mode") keyword)
      (some (fn [[strategy key]]
              (when (true? (json-boolean config key)) strategy))
            pooling-config-keys)))

(defn- model-config-opts
  [model-dir]
  (let [modules-json (read-json (file-in model-dir "modules.json"))
        model-modules (modules modules-json)
        root-config (json-object (read-json (file-in model-dir "config.json")))
        sentence-config (json-object
                         (read-json (file-in model-dir "sentence_bert_config.json")))
        pooling-config (some-> (pooling-module-config model-dir model-modules)
                               read-json
                               json-object)
        pooling (configured-pooling pooling-config)
        max-length (or (json-long sentence-config "max_seq_length")
                       (json-long root-config "max_position_embeddings"))]
    (cond-> {}
      modules-json (assoc :normalize? (boolean (some #(module-type? % ".Normalize")
                                                     model-modules)))
      pooling (assoc :pooling pooling)
      max-length (assoc :max-length max-length))))

(defn- ensure-open
  [model]
  (when @(:closed? model)
    (throw (model-error "model is closed"
                        {:embeddings/error :model-closed}))))

(defn- tensor-info
  ^TensorInfo
  [^NodeInfo node]
  (let [info (.getInfo node)]
    (if (instance? TensorInfo info)
      (cast TensorInfo info)
      (throw (model-error "model output is not a tensor"
                          {:embeddings/error :unsupported-output})))))

(defn- selected-output-name
  [output-names {:keys [output-name]}]
  (if output-name
    (if (contains? (set output-names) output-name)
      output-name
      (throw (model-error "model output not found"
                          {:embeddings/error :output-not-found
                           :output-name output-name})))
    (first output-names)))

(defn- provider-name [provider]
  (if (map? provider) (:provider provider) provider))

(defn- provider-device-id ^long [provider]
  (long (or (when (map? provider) (:device-id provider)) 0)))

(defn- provider-unavailable [provider cause]
  (throw (ex-info "execution provider unavailable"
                  {:embeddings/error :execution-provider-unavailable
                   :provider provider}
                  cause)))

(defn- add-execution-provider!
  [^OrtSession$SessionOptions session-options entry]
  (let [provider (provider-name entry)]
    (try
      (case provider
        :cuda (.addCUDA session-options (int (provider-device-id entry)))
        :coreml (.addCoreML session-options)
        :rocm (.addROCM session-options (int (provider-device-id entry)))
        :tensorrt (.addTensorrt session-options (int (provider-device-id entry)))
        :directml (.addDirectML session-options (int (provider-device-id entry)))
        :xnnpack (.addXnnpack session-options (Collections/emptyMap))
        (throw (ex-info "unknown execution provider"
                        {:embeddings/error :unknown-execution-provider
                         :provider provider})))
      (catch OrtException ex
        (provider-unavailable provider ex))
      (catch UnsatisfiedLinkError ex
        (provider-unavailable provider ex)))))

(defn- ->session-options
  ^OrtSession$SessionOptions
  [execution-providers]
  (when (seq execution-providers)
    (let [session-options (OrtSession$SessionOptions.)]
      (try
        (doseq [entry execution-providers]
          (add-execution-provider! session-options entry))
        session-options
        (catch Throwable ex
          (.close session-options)
          (throw ex))))))

(defn- first-provider [execution-providers]
  (provider-name (first execution-providers)))

(defn load-model
  "Load a local model directory containing `model.onnx` and `tokenizer.json`.

  Options include `:pooling` (`:mean`, `:mean-sqrt-len`, `:cls`, `:max`),
  `:normalize?`, `:max-length`, and `:execution-providers`, a vector of
  provider keywords or maps such as `[:coreml]`, `[:cuda]`, or
  `[{:provider :cuda :device-id 0}]`. CPU remains the implicit fallback when
  execution providers are absent or empty."
  ([model-dir]
   (load-model model-dir nil))
  ([model-dir opts]
   (let [model-file (file-in model-dir "model.onnx")
         tokenizer-file (file-in model-dir "tokenizer.json")]
     (when-not (.exists model-file)
       (throw (model-error "model.onnx not found"
                           {:embeddings/error :model-not-found
                            :path (.getPath model-file)})))
     (when-not (.exists tokenizer-file)
       (throw (model-error "tokenizer.json not found"
                           {:embeddings/error :tokenizer-not-found
                            :path (.getPath tokenizer-file)})))
     (let [resolved-opts (merge default-opts (model-config-opts model-dir) opts)
           env (OrtEnvironment/getEnvironment)
           session-options (->session-options (:execution-providers opts))
           session (if session-options
                     (try
                       (.createSession ^OrtEnvironment env (.getPath model-file) session-options)
                       (catch OrtException ex
                         (provider-unavailable (first-provider (:execution-providers opts)) ex))
                       (catch UnsatisfiedLinkError ex
                         (provider-unavailable (first-provider (:execution-providers opts)) ex))
                       (finally
                         (.close ^OrtSession$SessionOptions session-options)))
                     (.createSession ^OrtEnvironment env (.getPath model-file)))]
       (->LocalModel session
                     (tokenizers/from-file (.getPath tokenizer-file))
                     resolved-opts
                     (set (.getInputNames ^OrtSession session))
                     (selected-output-name (.getOutputNames ^OrtSession session)
                                           resolved-opts)
                     (atom false))))))

(defn close
  [model]
  (when (compare-and-set! (:closed? model) false true)
    (.close ^OrtSession (:session model)))
  nil)

(defmacro with-model
  [[sym dir opts] & body]
  `(let [~sym (load-model ~dir ~opts)]
     (try
       ~@body
       (finally
         (close ~sym)))))

(defn- model-dimension
  ^long
  [model]
  (ensure-open model)
  (let [^OrtSession session (:session model)
        ^Map output-info (.getOutputInfo session)
        ^NodeInfo node (.get output-info (:output-name model))
        ^longs shape (.getShape (tensor-info node))]
    (long (aget shape (dec (alength shape))))))

(defn- truncate-seq
  [xs ^long max-length]
  (vec (take max-length xs)))

(defn- encoded-row
  [tokenizer text ^long max-length]
  (let [encoded (tokenizers/encode tokenizer text)
        ids (truncate-seq (:ids encoded) max-length)
        mask (truncate-seq (or (:attention-mask encoded) (repeat (count ids) 1)) max-length)
        type-ids (truncate-seq (or (:type-ids encoded) (repeat (count ids) 0)) max-length)]
    {:ids ids
     :attention-mask mask
     :type-ids type-ids
     :position-ids (vec (range (count ids)))}))

(defn- batch-seq-length
  ^long
  [encoded]
  (long (max 1 (reduce max 0 (map #(count (:ids %)) encoded)))))

(defn- padded-matrix
  ^"[[J"
  [encoded key ^long seq-length ^long pad-value]
  (let [batch-size (count encoded)
        ^"[[J" out (make-array Long/TYPE batch-size seq-length)]
    (loop [i 0
           rows (seq encoded)]
      (when rows
        (let [values (seq (get (first rows) key))
              ^longs row (aget out i)]
          (loop [j 0
                 xs values]
            (when (< j seq-length)
              (aset-long row j (long (if xs (first xs) pad-value)))
              (recur (inc j) (next xs)))))
        (recur (inc i) (next rows))))
    out))

(defn- unsupported-input
  [name]
  (throw (model-error "unsupported model input"
                      {:embeddings/error :unsupported-input
                       :input name})))

(def ^:private default-input-schema
  {"input_ids" {:source :ids :pad-value 0}
   "attention_mask" {:source :attention-mask :pad-value 0}
   "token_type_ids" {:source :type-ids :pad-value 0}
   "position_ids" {:source :position-ids :pad-value 0}})

(defn- input-spec [schema name]
  (let [spec (get schema name)]
    (cond
      (keyword? spec) {:source spec :pad-value 0}
      (map? spec) spec
      :else (unsupported-input name))))

(defn- input-tensors
  ([^OrtEnvironment env input-names encoded]
   (input-tensors env input-names encoded nil))
  ([^OrtEnvironment env input-names encoded input-schema]
   (let [schema (merge default-input-schema input-schema)
         seq-length (batch-seq-length encoded)
         masks (padded-matrix encoded :attention-mask seq-length 0)
         inputs (HashMap.)
         tensors (transient [])]
     (doseq [name input-names]
       (let [{:keys [source pad-value]} (input-spec schema name)
             values (padded-matrix encoded source seq-length (long (or pad-value 0)))
             tensor (OnnxTensor/createTensor env values)]
         (.put inputs name tensor)
         (conj! tensors tensor)))
     {:inputs inputs
      :tensors (persistent! tensors)
      :masks masks})))

(defn- close-all
  [xs]
  (doseq [x xs]
    (.close ^java.lang.AutoCloseable x)))

(defn- normalize-if-needed
  [^floats embedding normalize?]
  (if normalize?
    (math/l2-normalize embedding)
    embedding))

(defn- rank-3-embeddings
  [^"[[[F" output ^"[[J" masks pooling normalize?]
  (let [batch-size (alength output)]
    (loop [i 0
           out []]
      (if (= i batch-size)
        out
        (let [pooled (pooling/pool pooling (aget output i) (aget masks i))]
          (recur (inc i)
                 (conj out (normalize-if-needed pooled normalize?))))))))

(defn- rank-2-embeddings
  [^"[[F" output normalize?]
  (let [batch-size (alength output)]
    (loop [i 0
           out []]
      (if (= i batch-size)
        out
        (let [row (aclone ^floats (aget output i))]
          (recur (inc i)
                 (conj out (normalize-if-needed row normalize?))))))))

(defn- output-embeddings
  [value masks opts]
  (cond
    (instance? (Class/forName "[[[F") value)
    (rank-3-embeddings value masks (:pooling opts) (:normalize? opts))

    (instance? (Class/forName "[[F") value)
    (rank-2-embeddings value (:normalize? opts))

    :else
    (throw (model-error "unsupported model output"
                        {:embeddings/error :unsupported-output
                         :class (class value)}))))

(defn- embed-batch*
  [model texts]
  (ensure-open model)
  (if (empty? texts)
    []
    (let [opts (:opts model)
          max-length (long (:max-length opts))
          encoded (mapv #(encoded-row (:tokenizer model) % max-length) texts)
          env (OrtEnvironment/getEnvironment)
          {:keys [^Map inputs tensors masks]} (input-tensors env
                                                              (:input-names model)
                                                              encoded
                                                              (:input-schema opts))
          ^OrtSession session (:session model)
          result (atom nil)]
      (try
        (let [^OrtSession$Result run-result (.run session
                                                   inputs
                                                   (Collections/singleton
                                                    (:output-name model)))]
          (reset! result run-result)
          (let [^OnnxValue output (.get run-result 0)]
            (output-embeddings (.getValue output) masks opts)))
        (finally
          (when-let [^OrtSession$Result run-result @result]
            (.close run-result))
          (close-all tensors))))))
