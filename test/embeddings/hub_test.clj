(ns embeddings.hub-test
  (:require [clojure.java.io :as io]
            [clojure.data.json :as json]
            [clojure.test :refer [deftest is testing]]
            [embeddings.hub :as hub])
  (:import (java.nio.file Files)
           (java.math BigInteger)
           (java.security MessageDigest)))

(set! *warn-on-reflection* true)

(def fetch-model* #'hub/fetch-model*)

(defn- tmp-dir []
  (.toFile (Files/createTempDirectory "hub-test" (make-array java.nio.file.attribute.FileAttribute 0))))

(defn- sha256 [^String value]
  (format "%064x" (BigInteger. 1 (.digest (doto (MessageDigest/getInstance "SHA-256")
                                           (.update (.getBytes ^String value "UTF-8")))))))

(defn- manifest [files]
  (json/write-str (mapv (fn [[path body]]
                          {"path" path
                           "type" "file"
                           "size" (count body)
                           "lfs" {"sha256" (sha256 body)}})
                        files)))

(defn- ordinary-manifest [files]
  (json/write-str (mapv (fn [[path body]]
                          {"path" path
                           "type" "file"
                           "size" (count body)
                           "oid" "ordinary-file-git-blob-id"})
                        files)))

(defn- secure-transport [requests files]
  (fn [{:keys [url headers method]}]
    (swap! requests conj {:url url :headers headers :method method})
    (if (re-find #"/api/models/.*/tree/" url)
      {:status 200 :headers {} :body (manifest files)}
      (let [path (last (re-find #"/resolve/[^/]+/(.*)$" url))
            body (get files path)]
        {:status (if body 200 404)
         :headers {"content-length" (str (count body))}
         :body body}))))

(deftest secure-download-sends-explicit-auth-token
  (let [^java.io.File dir (tmp-dir)
        requests (atom [])
        transport (secure-transport requests
                                    {"onnx/model.onnx" "model"
                                     "tokenizer.json" "tokenizer"})]
    (hub/fetch-model "org/model" {:cache-dir (.getPath dir)
                                   :token "explicit-token"
                                   :transport transport})
    (is (every? #(= "Bearer explicit-token" (get-in % [:headers "Authorization"]))
                @requests))))

(deftest default-transport-follows-normal-redirects
  (is (= java.net.http.HttpClient$Redirect/NORMAL
         (.followRedirects ^java.net.http.HttpClient
                           ((deref #'hub/default-http-client))))))

(deftest secure-download-skips-checksum-for-ordinary-files
  (let [^java.io.File dir (tmp-dir)
        transport (fn [{:keys [url]}]
                    (if (re-find #"/api/models/.*/tree/" url)
                      {:status 200 :headers {} :body (ordinary-manifest {"onnx/model.onnx" "model"
                                                                           "tokenizer.json" "tokenizer"})}
                      {:status 200 :headers {} :body (if (re-find #"tokenizer" url)
                                                        "tokenizer"
                                                        "model")}))]
    (hub/fetch-model "org/model" {:cache-dir (.getPath dir)
                                   :transport transport})
    (is (= "model" (slurp (io/file dir "org/model/model.onnx"))))))

(deftest manifest-http-errors-throw-clear-ex-info
  (let [^java.io.File dir (tmp-dir)]
    (is (= {:embeddings/error :manifest-failed
            :status 503}
           (try (hub/fetch-model "org/model" {:cache-dir (.getPath dir)
                                                :transport (fn [_] {:status 503
                                                                    :headers {}
                                                                    :body "unavailable"})})
                (catch clojure.lang.ExceptionInfo e
                  (select-keys (ex-data e) [:embeddings/error :status])))))))

(deftest manifest-pagination-follows-next-link
  (let [^java.io.File dir (tmp-dir)
        pages (atom 0)
        transport (fn [{:keys [url]}]
                    (if (re-find #"/api/models/.*/tree/" url)
                      (if (= 1 (swap! pages inc))
                        {:status 200
                         :headers {"link" ["<https://huggingface.co/api/models/org/model/tree/main?recursive=true&cursor=next>; rel=\"next\""]}
                         :body (ordinary-manifest {"onnx/model.onnx" "model"})}
                        {:status 200 :headers {} :body (ordinary-manifest {"tokenizer.json" "tokenizer"})})
                      {:status 200 :headers {} :body (if (re-find #"tokenizer" url)
                                                        "tokenizer"
                                                        "model")}))]
    (hub/fetch-model "org/model" {:cache-dir (.getPath dir)
                                   :transport transport})
    (is (= 2 @pages))
    (is (.exists (io/file dir "org/model/model.onnx")))
    (is (.exists (io/file dir "org/model/tokenizer.json")))))

(deftest secure-download-uses-hf-token-environment
  (let [^java.io.File dir (tmp-dir)
        requests (atom [])
        transport (secure-transport requests
                                    {"onnx/model.onnx" "model"
                                     "tokenizer.json" "tokenizer"})]
    (with-redefs [hub/environment-token (constantly "environment-token")]
      (hub/fetch-model "org/model" {:cache-dir (.getPath dir)
                                     :transport transport}))
    (is (every? #(= "Bearer environment-token" (get-in % [:headers "Authorization"]))
                @requests))))

(deftest secure-download-verifies-published-checksums
  (let [^java.io.File dir (tmp-dir)
        requests (atom [])
        files {"onnx/model.onnx" "model" "tokenizer.json" "tokenizer"}
        transport (secure-transport requests files)]
    (hub/fetch-model "org/model" {:cache-dir (.getPath dir)
                                   :transport transport})
    (spit (io/file dir "org/model/model.onnx") "tampered")
    (is (= :checksum-mismatch
           (try (hub/fetch-model "org/model" {:cache-dir (.getPath dir)
                                                :transport transport})
                (catch clojure.lang.ExceptionInfo e
                  (:embeddings/error (ex-data e))))))))

(deftest secure-download-resumes-partial-file
  (let [^java.io.File dir (tmp-dir)
        requests (atom [])
        files {"onnx/model.onnx" "model" "tokenizer.json" "tokenizer"}
        transport (fn [{:keys [url] :as request}]
                    (swap! requests conj request)
                    (if (re-find #"/api/models/.*/tree/" url)
                      {:status 200 :headers {} :body (manifest files)}
                      {:status 206
                       :headers {"content-range" "bytes 2-4/5"}
                       :body (if (re-find #"tokenizer" url) "tokenizer" "del")}))
        root (io/file dir "org/model")]
    (.mkdirs root)
    (spit (io/file root "model.onnx.part") "mo")
    (hub/fetch-model "org/model" {:cache-dir (.getPath dir)
                                   :transport transport})
    (is (= "model" (slurp (io/file root "model.onnx"))))
    (is (= "bytes=2-" (some #(get-in % [:headers "Range"]) @requests)))))

(deftest secure-download-serializes-concurrent-writers
  (let [^java.io.File dir (tmp-dir)
        requests (atom [])
        entered (promise)
        release (promise)
        transport (fn [{:keys [url] :as request}]
                    (swap! requests conj request)
                    (if (re-find #"/api/models/.*/tree/" url)
                      {:status 200 :headers {} :body (manifest {"onnx/model.onnx" "model"
                                                                 "tokenizer.json" "tokenizer"})}
                      (do (deliver entered true)
                          @release
                          {:status 200 :headers {} :body (if (re-find #"tokenizer" url)
                                                            "tokenizer" "model")})))
        f1 (future (hub/fetch-model "org/model" {:cache-dir (.getPath dir)
                                                  :transport transport}))
        _ @entered
        f2 (future (hub/fetch-model "org/model" {:cache-dir (.getPath dir)
                                                  :transport transport}))]
    (Thread/sleep 50)
    (is (not (realized? f2)))
    (deliver release true)
    @f1
    @f2
    (is (= 2 (count (filter #(not (re-find #"/api/models/" (:url %))) @requests))))))

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
    (testing "urls use the HF resolve endpoints"
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
