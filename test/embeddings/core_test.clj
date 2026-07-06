(ns embeddings.core-test
  (:require [clojure.test :refer [deftest is testing]]
            [embeddings.core :as embeddings]
            [embeddings.math :as math]))

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
