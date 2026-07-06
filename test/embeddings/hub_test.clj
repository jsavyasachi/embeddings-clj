(ns embeddings.hub-test
  (:require [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [embeddings.hub :as hub])
  (:import (java.nio.file Files)))

(set! *warn-on-reflection* true)

(def fetch-model* #'hub/fetch-model*)

(defn- tmp-dir []
  (.toFile (Files/createTempDirectory "hub-test" (make-array java.nio.file.attribute.FileAttribute 0))))

(deftest fetches-into-cache-layout
  (let [^java.io.File dir (tmp-dir)
        calls (atom [])
        download! (fn [url dest]
                    (swap! calls conj url)
                    (io/make-parents dest)
                    (spit dest "fake")
                    true)
        path (fetch-model* "sentence-transformers/all-MiniLM-L6-v2"
                           {:cache-dir (.getPath dir)}
                           download!)]
    (is (= (.getPath (io/file dir "sentence-transformers" "all-MiniLM-L6-v2")) path))
    (is (.exists (io/file path "model.onnx")))
    (is (.exists (io/file path "tokenizer.json")))
    (testing "urls hit the HF resolve endpoints"
      (is (some #(= "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx" %) @calls))
      (is (some #(= "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json" %) @calls)))))

(deftest skips-existing-files
  (let [^java.io.File dir (tmp-dir)
        called (atom 0)
        download! (fn [_ dest] (swap! called inc) (io/make-parents dest) (spit dest "x") true)]
    (fetch-model* "org/model" {:cache-dir (.getPath dir)} download!)
    (let [before @called]
      (fetch-model* "org/model" {:cache-dir (.getPath dir)} download!)
      (is (= before @called)))))

(deftest falls-back-to-root-model-url
  (let [^java.io.File dir (tmp-dir)
        download! (fn [url dest]
                    (if (re-find #"/onnx/model\.onnx$" url)
                      false
                      (do (io/make-parents dest) (spit dest "x") true)))
        path (fetch-model* "org/model" {:cache-dir (.getPath dir)} download!)]
    (is (.exists (io/file path "model.onnx")))))

(deftest quantized-variant-uses-fallback-chain
  (let [^java.io.File dir (tmp-dir)
        calls (atom [])
        download! (fn [url dest]
                    (swap! calls conj url)
                    (if (or (re-find #"/onnx/model_qint8_avx512_vnni\.onnx$" url)
                            (re-find #"/tokenizer\.json$" url))
                      (do (io/make-parents dest) (spit dest "x") true)
                      false))
        path (fetch-model* "org/model" {:cache-dir (.getPath dir)
                                        :variant :quantized}
                           download!)]
    (is (= (.getPath (io/file dir "org" "model" "quantized")) path))
    (is (.exists (io/file path "model.onnx")))
    (is (.exists (io/file path "tokenizer.json")))
    (is (= ["https://huggingface.co/org/model/resolve/main/onnx/model_quantized.onnx"
            "https://huggingface.co/org/model/resolve/main/onnx/model_qint8_avx512_vnni.onnx"
            "https://huggingface.co/org/model/resolve/main/tokenizer.json"]
           @calls))))

(deftest string-variant-uses-explicit-path
  (let [^java.io.File dir (tmp-dir)
        calls (atom [])
        download! (fn [url dest]
                    (swap! calls conj url)
                    (io/make-parents dest)
                    (spit dest "x")
                    true)
        path (fetch-model* "org/model" {:cache-dir (.getPath dir)
                                        :variant "onnx/model_q4.onnx"}
                           download!)]
    (is (= (.getPath (io/file dir "org" "model" "model_q4")) path))
    (is (= ["https://huggingface.co/org/model/resolve/main/onnx/model_q4.onnx"
            "https://huggingface.co/org/model/resolve/main/tokenizer.json"]
           @calls))))

(deftest rejects-invalid-variants
  (doseq [bad ["onnx/model.bin" "../model.onnx" "/tmp/model.onnx" "onnx/../model.onnx"]]
    (is (= {:embeddings/error :invalid-variant
            :variant bad}
           (try (fetch-model* "org/model" {:variant bad} (constantly true))
                (catch clojure.lang.ExceptionInfo e (ex-data e)))))))

(deftest variant-cache-skips-existing-files
  (let [^java.io.File dir (tmp-dir)
        called (atom 0)
        download! (fn [_ dest]
                    (swap! called inc)
                    (io/make-parents dest)
                    (spit dest "x")
                    true)
        opts {:cache-dir (.getPath dir)
              :variant "onnx/model_q4.onnx"}]
    (fetch-model* "org/model" opts download!)
    (let [before @called]
      (fetch-model* "org/model" opts download!)
      (is (= before @called)))))

(deftest rejects-bad-model-ids
  (doseq [bad ["../etc" "a/b/../c" "" "a b" "no-slash"]]
    (is (= :invalid-model-id
           (try (fetch-model* bad {} (constantly true))
                (catch clojure.lang.ExceptionInfo e (:embeddings/error (ex-data e))))))))

(deftest download-failure-throws
  (let [^java.io.File dir (tmp-dir)]
    (is (= :download-failed
           (try (fetch-model* "org/model" {:cache-dir (.getPath dir)} (constantly false))
                (catch clojure.lang.ExceptionInfo e (:embeddings/error (ex-data e))))))))
