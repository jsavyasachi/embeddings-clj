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
  (let [dir (tmp-dir)
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
  (let [dir (tmp-dir)
        called (atom 0)
        download! (fn [_ dest] (swap! called inc) (io/make-parents dest) (spit dest "x") true)]
    (fetch-model* "org/model" {:cache-dir (.getPath dir)} download!)
    (let [before @called]
      (fetch-model* "org/model" {:cache-dir (.getPath dir)} download!)
      (is (= before @called)))))

(deftest falls-back-to-root-model-url
  (let [dir (tmp-dir)
        download! (fn [url dest]
                    (if (re-find #"/onnx/model\.onnx$" url)
                      false
                      (do (io/make-parents dest) (spit dest "x") true)))
        path (fetch-model* "org/model" {:cache-dir (.getPath dir)} download!)]
    (is (.exists (io/file path "model.onnx")))))

(deftest rejects-bad-model-ids
  (doseq [bad ["../etc" "a/b/../c" "" "a b" "no-slash"]]
    (is (= :invalid-model-id
           (try (fetch-model* bad {} (constantly true))
                (catch clojure.lang.ExceptionInfo e (:embeddings/error (ex-data e))))))))

(deftest download-failure-throws
  (let [dir (tmp-dir)]
    (is (= :download-failed
           (try (fetch-model* "org/model" {:cache-dir (.getPath dir)} (constantly false))
                (catch clojure.lang.ExceptionInfo e (:embeddings/error (ex-data e))))))))
