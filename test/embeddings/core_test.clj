(ns embeddings.core-test
  (:require [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [embeddings.core :as embeddings]
            [embeddings.math :as math])
  (:import (java.nio.file Files StandardCopyOption)))

(set! *warn-on-reflection* true)

(def ^:private epsilon 1.0E-5)
(def fixtures-present?
  (.exists (java.io.File. "fixtures/token-model/model.onnx")))

(defn- farray
  [& xs]
  (float-array xs))

(defn- approx=
  [^floats xs ^floats ys]
  (and (= (alength xs) (alength ys))
       (loop [i 0]
         (or (= i (alength xs))
             (and (<= (Math/abs (- (double (aget xs i))
                                  (double (aget ys i))))
                      epsilon)
                  (recur (inc i)))))))

(defmacro ^:private when-fixtures
  [& body]
  `(if fixtures-present?
     (do ~@body)
     (is true "fixtures absent; skipping ONNX corpus test")))

(defn- configured-model-dir []
  (let [dir (.toFile (Files/createTempDirectory
                      "configured-model"
                      (make-array java.nio.file.attribute.FileAttribute 0)))
        source (io/file "fixtures/token-model")]
    (doseq [filename ["model.onnx" "tokenizer.json"]]
      (Files/copy (.toPath (io/file source filename))
                  (.toPath (io/file dir filename))
                  ^"[Ljava.nio.file.CopyOption;"
                  (into-array StandardCopyOption
                              [StandardCopyOption/REPLACE_EXISTING])))
    (spit (io/file dir "modules.json")
          "[{
             \"idx\": 0,
             \"path\": \"\",
             \"type\": \"sentence_transformers.models.Transformer\"
           }, {
             \"idx\": 1,
             \"path\": \"1_Pooling\",
             \"type\": \"sentence_transformers.models.Pooling\"
           }]")
    (spit (io/file dir "config.json") "{\"max_position_embeddings\": 256}")
    (spit (io/file dir "sentence_bert_config.json") "{\"max_seq_length\": 128}")
    (doto (io/file dir "1_Pooling") .mkdir)
    (spit (io/file dir "1_Pooling" "config.json")
          "{\"pooling_mode_cls_token\": true,
            \"pooling_mode_mean_tokens\": false,
            \"pooling_mode_max_tokens\": false,
            \"pooling_mode_mean_sqrt_len_tokens\": false}")
    dir))

(deftest load-model-resolves-sentence-transformers-config-test
  (when-fixtures
    (let [dir (configured-model-dir)]
      (embeddings/with-model [model dir nil]
        (is (= {:pooling :cls :normalize? false :max-length 128}
               (select-keys (:opts model) [:pooling :normalize? :max-length]))))
      (embeddings/with-model [model dir {:pooling :max
                                         :normalize? true
                                         :max-length 64}]
        (is (= {:pooling :max :normalize? true :max-length 64}
               (select-keys (:opts model) [:pooling :normalize? :max-length])))))))

(deftest load-model-test
  (when-fixtures
    (embeddings/with-model [model "fixtures/token-model" {:normalize? false}]
      (is (= 4 (embeddings/dimension model)))
      (is (contains? (:input-names model) "input_ids"))
      (is (contains? (:input-names model) "attention_mask")))))

(deftest load-model-execution-provider-test
  (when-fixtures
    (testing "unknown provider"
      (try
        (embeddings/load-model "fixtures/token-model" {:execution-providers [:bogus]})
        (is false "expected exception")
        (catch clojure.lang.ExceptionInfo ex
          (is (= {:embeddings/error :unknown-execution-provider
                  :provider :bogus}
                 (ex-data ex))))))
    (testing "provider unavailable shape"
      (try
        (let [model (embeddings/load-model "fixtures/token-model"
                                           {:execution-providers [:cuda]})]
          (embeddings/close model)
          (is true "cuda provider loaded"))
        (catch clojure.lang.ExceptionInfo ex
          (is (= :execution-provider-unavailable
                 (:embeddings/error (ex-data ex))))
          (is (= :cuda (:provider (ex-data ex))))
          (is (some? (ex-cause ex))))))))

(deftest token-model-mean-pooling-test
  (when-fixtures
    (embeddings/with-model [model "fixtures/token-model" {:normalize? false
                                                          :pooling :mean}]
      (is (approx= (farray 2.5 3.0 -2.5 0.25)
                   (embeddings/embed model "hello world"))))))

(deftest token-model-cls-pooling-test
  (when-fixtures
    (embeddings/with-model [model "fixtures/token-model" {:normalize? false
                                                          :pooling :cls}]
      (is (approx= (farray 2.0 2.5 -2.0 0.2)
                   (embeddings/embed model "hello world"))))))

(deftest token-model-normalization-test
  (when-fixtures
    (embeddings/with-model [model "fixtures/token-model" {:normalize? true
                                                          :pooling :mean}]
      (let [embedding (embeddings/embed model "hello world")]
        (is (approx= (math/l2-normalize (farray 2.5 3.0 -2.5 0.25))
                     embedding))
        (is (<= (Math/abs (- 1.0 (math/norm embedding))) epsilon))))))

(deftest token-model-batch-padding-test
  (when-fixtures
    (embeddings/with-model [model "fixtures/token-model" {:normalize? false
                                                          :pooling :mean}]
      (let [embeddings (embeddings/embed-batch model ["hello world" "foo"])]
        (is (= 2 (count embeddings)))
        (is (approx= (farray 2.5 3.0 -2.5 0.25) (first embeddings)))
        (is (approx= (farray 4.0 4.5 -4.0 0.4) (second embeddings)))))))

(deftest pooled-model-test
  (when-fixtures
    (embeddings/with-model [model "fixtures/pooled-model" {:normalize? false}]
      (is (approx= (farray 2.5 3.0 -2.5 0.25)
                   (embeddings/embed model "hello world"))))))

(deftest lifecycle-and-errors-test
  (when-fixtures
    (testing "missing model"
      (try
        (embeddings/load-model "fixtures/missing-model")
        (is false "expected exception")
        (catch clojure.lang.ExceptionInfo ex
          (is (= :model-not-found (:embeddings/error (ex-data ex)))))))
    (testing "closed model"
      (let [model (embeddings/load-model "fixtures/token-model" {:normalize? false})]
        (embeddings/close model)
        (embeddings/close model)
        (try
          (embeddings/embed model "hello")
          (is false "expected exception")
          (catch clojure.lang.ExceptionInfo ex
            (is (= :model-closed (:embeddings/error (ex-data ex))))))))))

(deftest unknown-word-test
  (when-fixtures
    (embeddings/with-model [model "fixtures/token-model" {:normalize? false}]
      (is (= 4 (alength ^floats (embeddings/embed model "hello qqq")))))))

(deftest prefix-option
  (when-fixtures
    (embeddings/with-model [model "fixtures/token-model" {:normalize? false}]
      (testing ":prefix prepends to every input, matching manual prepending"
        (let [manual (embeddings/embed model "query: hello")
              via-opt (embeddings/embed model "hello" {:prefix "query: "})]
          (is (= (vec manual) (vec via-opt))))
        (let [manual (embeddings/embed-batch model ["query: a" "query: b"])
              via-opt (embeddings/embed-batch model ["a" "b"] {:prefix "query: "})]
          (is (= (mapv vec manual) (mapv vec via-opt)))))
      (testing "nil/absent opts unchanged"
        (is (= (vec (embeddings/embed model "x"))
               (vec (embeddings/embed model "x" nil))))))))

(deftest embedding-provider-protocol-test
  (is (some? (ns-resolve 'embeddings.core 'EmbeddingProvider)))
  (is (try
        (require 'embeddings.providers)
        true
        (catch java.io.FileNotFoundException _
          false))))

(deftest named-output-and-position-input-api-test
  (is (some? (ns-resolve 'embeddings.core 'selected-output-name)))
  (let [input-tensors #'embeddings/input-tensors
        encoded [{:ids [10 11]
                  :attention-mask [1 1]
                  :type-ids [0 0]
                  :position-ids [0 1]}]]
    (is (nil? (try
                (let [{:keys [tensors]} (input-tensors
                                         (ai.onnxruntime.OrtEnvironment/getEnvironment)
                                         #{"position_ids"}
                                         encoded)]
                  (doseq [tensor tensors] (.close ^java.lang.AutoCloseable tensor))
                  nil)
                (catch clojure.lang.ExceptionInfo ex
                  (ex-data ex)))))))

(deftest named-output-selection-test
  (let [select-output #'embeddings/selected-output-name]
    (is (= "sentence_embedding"
           (select-output ["token_embeddings" "sentence_embedding"]
                          {:output-name "sentence_embedding"})))
    (is (= "token_embeddings"
           (select-output ["token_embeddings" "sentence_embedding"] {})))
    (is (= {:embeddings/error :output-not-found
            :output-name "missing"}
           (try
             (select-output ["token_embeddings"] {:output-name "missing"})
             (catch clojure.lang.ExceptionInfo ex
               (ex-data ex)))))))

(deftest custom-input-schema-test
  (let [input-tensors #'embeddings/input-tensors
        encoded [{:ids [10 11]
                  :attention-mask [1 1]
                  :type-ids [0 0]
                  :position-ids [0 1]}
                 {:ids [12]
                  :attention-mask [1]
                  :type-ids [0]
                  :position-ids [0]}]
        result (try
                 (input-tensors
                  (ai.onnxruntime.OrtEnvironment/getEnvironment)
                  #{"position_ids" "custom_ids"}
                  encoded
                  {"custom_ids" {:source :ids :pad-value 9}})
                 (catch clojure.lang.ArityException _
                   :unsupported))]
    (is (not= :unsupported result))
    (when (map? result)
      (try
        (is (= [[0 1] [0 0]]
               (mapv vec (.getValue ^ai.onnxruntime.OnnxTensor
                                    (.get ^java.util.Map (:inputs result)
                                          "position_ids")))))
        (is (= [[10 11] [12 9]]
               (mapv vec (.getValue ^ai.onnxruntime.OnnxTensor
                                    (.get ^java.util.Map (:inputs result)
                                          "custom_ids")))))
        (finally
          (doseq [tensor (:tensors result)]
            (.close ^java.lang.AutoCloseable tensor)))))))

(deftest matryoshka-output-dimensions-test
  (let [postprocess #'embeddings/normalize-if-needed
        normalized (try
                     (postprocess (farray 3.0 4.0 12.0) true 2)
                     (catch clojure.lang.ArityException _
                       :unsupported))]
    (is (not= :unsupported normalized))
    (when-not (= :unsupported normalized)
      (is (approx= (farray 0.6 0.8) normalized))
      (is (<= (Math/abs (- 1.0 (math/norm normalized))) epsilon))
      (is (approx= (farray 3.0 4.0)
                   (postprocess (farray 3.0 4.0 12.0) false 2)))
      (is (= {:embeddings/error :invalid-output-dimensions
              :output-dimensions 4
              :dimensions 3}
             (try
               (postprocess (farray 3.0 4.0 12.0) true 4)
               (catch clojure.lang.ExceptionInfo ex
                 (ex-data ex))))))))

(deftest close-releases-native-resources-once-test
  (let [session-closes (atom 0)
        tokenizer-closes (atom 0)
        session (reify java.lang.AutoCloseable
                  (close [_] (swap! session-closes inc)))
        tokenizer (reify java.lang.AutoCloseable
                    (close [_] (swap! tokenizer-closes inc)))
        model (embeddings/->LocalModel session tokenizer {} #{} nil (atom false))
        result (try
                 (embeddings/close model)
                 (embeddings/close model)
                 :closed
                 (catch Throwable _
                   :failed))]
    (is (= :closed result))
    (is (= 1 @session-closes))
    (is (= 1 @tokenizer-closes))))

(deftest partial-model-construction-cleans-resources-test
  (let [construct-model (ns-resolve 'embeddings.core 'construct-local-model)
        session-closes (atom 0)
        session (reify java.lang.AutoCloseable
                  (close [_] (swap! session-closes inc)))
        result (if construct-model
                 (try
                   (construct-model session
                                    #(throw (ex-info "tokenizer failed" {}))
                                    {}
                                    #{"input_ids"}
                                    ["output"])
                   :unexpected-success
                   (catch clojure.lang.ExceptionInfo _
                     :failed-as-expected))
                 :unsupported)]
    (is (= :failed-as-expected result))
    (is (= 1 @session-closes)))
  (let [construct-model (ns-resolve 'embeddings.core 'construct-local-model)
        session-closes (atom 0)
        tokenizer-closes (atom 0)
        session (reify java.lang.AutoCloseable
                  (close [_] (swap! session-closes inc)))
        tokenizer (reify java.lang.AutoCloseable
                    (close [_] (swap! tokenizer-closes inc)))]
    (when construct-model
      (is (= :output-not-found
             (try
               (construct-model session
                                (constantly tokenizer)
                                {:output-name "missing"}
                                #{"input_ids"}
               ["output"])
               (catch clojure.lang.ExceptionInfo ex
                 (:embeddings/error (ex-data ex))))))
      (is (= 1 @session-closes))
      (is (= 1 @tokenizer-closes)))))
