(ns embeddings.core
  (:require [embeddings.math :as math]
            [embeddings.pooling :as pooling]
            [tokenizers.core :as tokenizers])
  (:import (ai.onnxruntime NodeInfo OnnxTensor OnnxValue OrtEnvironment OrtSession
                           TensorInfo)
           (ai.onnxruntime OrtSession$Result)
           (java.io File)
           (java.util HashMap Map)))

(set! *warn-on-reflection* true)

(def ^:private default-opts
  {:pooling :mean
   :normalize? true
   :max-length 512})

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

(defn load-model
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
     (let [env (OrtEnvironment/getEnvironment)
           session (.createSession ^OrtEnvironment env (.getPath model-file))]
       {:session session
        :tokenizer (tokenizers/from-file (.getPath tokenizer-file))
        :opts (merge default-opts opts)
        :input-names (set (.getInputNames ^OrtSession session))
        :closed? (atom false)}))))

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

(defn dimension
  ^long
  [model]
  (ensure-open model)
  (let [^OrtSession session (:session model)
        ^Map output-info (.getOutputInfo session)
        ^java.util.Collection values (.values output-info)
        ^java.util.Iterator iterator (.iterator values)
        ^NodeInfo node (.next iterator)
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
     :type-ids type-ids}))

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

(defn- input-tensors
  [^OrtEnvironment env input-names encoded]
  (doseq [name input-names]
    (when-not (#{"input_ids" "attention_mask" "token_type_ids"} name)
      (unsupported-input name)))
  (let [seq-length (batch-seq-length encoded)
        ids (padded-matrix encoded :ids seq-length 0)
        masks (padded-matrix encoded :attention-mask seq-length 0)
        type-ids (padded-matrix encoded :type-ids seq-length 0)
        inputs (HashMap.)
        tensors (transient [])]
    (doseq [name input-names]
      (let [tensor (case name
                     "input_ids" (OnnxTensor/createTensor env ids)
                     "attention_mask" (OnnxTensor/createTensor env masks)
                     "token_type_ids" (OnnxTensor/createTensor env type-ids))]
        (.put inputs name tensor)
        (conj! tensors tensor)))
    {:inputs inputs
     :tensors (persistent! tensors)
     :masks masks}))

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

(defn embed-batch
  [model texts]
  (ensure-open model)
  (if (empty? texts)
    []
    (let [opts (:opts model)
          max-length (long (:max-length opts))
          encoded (mapv #(encoded-row (:tokenizer model) % max-length) texts)
          env (OrtEnvironment/getEnvironment)
          {:keys [^Map inputs tensors masks]} (input-tensors env (:input-names model) encoded)
          ^OrtSession session (:session model)
          result (atom nil)]
      (try
        (let [^OrtSession$Result run-result (.run session inputs)]
          (reset! result run-result)
          (let [^OnnxValue output (.get run-result 0)]
            (output-embeddings (.getValue output) masks opts)))
        (finally
          (when-let [^OrtSession$Result run-result @result]
            (.close run-result))
          (close-all tensors))))))

(defn embed
  [model text]
  (first (embed-batch model [text])))
