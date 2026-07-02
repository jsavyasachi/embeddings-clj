(ns embeddings.integration-test
  (:require [clojure.test :refer [deftest is testing]]
            [embeddings.core :as embeddings]
            [embeddings.math :as math])
  (:import (java.io File)))

(set! *warn-on-reflection* true)

(def ^:private model-dir "fixtures/all-MiniLM-L6-v2")
(def ^:private model-file (File. ^String model-dir "model.onnx"))
(def ^:private model-present? (.exists ^File model-file))

(defn- approx-scalar?
  [expected actual tolerance]
  (<= (Math/abs (- (double expected) (double actual)))
      (double tolerance)))

(defn- approx-vector?
  [^floats expected ^floats actual tolerance]
  (and (= (alength expected) (alength actual))
       (let [n (alength expected)
             tolerance (double tolerance)]
         (loop [i 0]
           (or (= i n)
               (and (approx-scalar? (aget expected i) (aget actual i) tolerance)
                    (recur (inc i))))))))

(defn- farray
  [& xs]
  (float-array xs))

(defmacro ^:private when-real-model
  [& body]
  `(if model-present?
     (do ~@body)
     (is true "real model fixture absent; run dev/fetch-model.sh to enable integration tests")))

(deftest ^:integration all-minilm-dimension-test
  (when-real-model
    (embeddings/with-model [model model-dir nil]
      (is (= 384 (embeddings/dimension model))))))

(deftest ^:integration all-minilm-semantic-sanity-test
  (when-real-model
    (embeddings/with-model [model model-dir nil]
      (let [cat (embeddings/embed model "A cat sits on the mat")
            kitten (embeddings/embed model "A kitten rests on the rug")
            market (embeddings/embed model "The stock market crashed today")
            cat-kitten (math/cosine-similarity cat kitten)
            cat-market (math/cosine-similarity cat market)
            kitten-market (math/cosine-similarity kitten market)]
        (is (> cat-kitten cat-market))
        (is (> cat-kitten kitten-market))
        (is (> cat-kitten 0.5))
        (is (< cat-market 0.4))
        (is (< kitten-market 0.4))))))

(deftest ^:integration all-minilm-batch-padding-test
  (when-real-model
    (embeddings/with-model [model model-dir nil]
      (let [texts ["A cat sits on the mat"
                   "A kitten rests on the rug"
                   "The stock market crashed today"]
            singles (mapv #(embeddings/embed model %) texts)
            batch (embeddings/embed-batch model texts)]
        (is (= 3 (count batch)))
        (doseq [^floats embedding batch]
          (testing "batch embedding shape and normalization"
            (is (= 384 (alength embedding)))
            (is (approx-scalar? 1.0 (math/norm embedding) 1.0E-3))))
        (doseq [[^floats single ^floats batched] (map vector singles batch)]
          (is (approx-vector? single batched 1.0E-4)))))))

(deftest ^:integration all-minilm-reference-parity-test
  (when-real-model
    (embeddings/with-model [model model-dir nil]
      (let [embedding (embeddings/embed model "hello world")
            first-four (float-array (map #(aget ^floats embedding (long %)) (range 4)))]
        (is (approx-vector? (farray -0.0347 0.0313 0.0067 0.0263)
                            first-four
                            2.0E-2))))))
