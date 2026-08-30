(ns embeddings.hub-test
  (:require [clojure.java.io :as io]
            [clojure.data.json :as json]
            [clojure.test :refer [deftest is testing]]
            [embeddings.hub :as hub])
  (:import (com.sun.net.httpserver HttpExchange HttpHandler HttpServer)
           (java.net InetSocketAddress URI)
           (java.nio.file Files)
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

(deftest default-http-client-does-not-auto-follow-redirects
  ;; `default-transport` now follows redirects itself (see the tests below)
  ;; so it can drop `Authorization` on a cross-origin hop. The underlying
  ;; `HttpClient` must therefore never auto-follow, or a same-process
  ;; redirect would be followed twice.
  (is (= java.net.http.HttpClient$Redirect/NEVER
         (.followRedirects ^java.net.http.HttpClient
                           ((deref #'hub/default-http-client))))))

;; --- Finding 1: default-transport must not leak Authorization across
;; --- origins on redirect. These use real loopback HTTP servers because the
;; --- bug lives in java.net.http.HttpClient's redirect handling, which stub
;; --- transports never exercise.

(defn- respond! [^HttpExchange exchange status ^String body]
  (let [^bytes bytes (.getBytes body "UTF-8")]
    (.sendResponseHeaders exchange status (count bytes))
    (with-open [os (.getResponseBody exchange)]
      (.write os bytes))))

(defn- redirect! [^HttpExchange exchange status ^String location]
  (.set (.getResponseHeaders exchange) "Location" location)
  (.sendResponseHeaders exchange status -1)
  (.close exchange))

(defn- start-server ^HttpServer [^HttpHandler handler]
  (let [server (HttpServer/create (InetSocketAddress. "127.0.0.1" 0) 0)]
    (.createContext server "/" handler)
    (.setExecutor server nil)
    (.start server)
    server))

(defn- server-url ^String [^HttpServer server ^String path]
  (str "http://127.0.0.1:" (.getPort (.getAddress server)) path))

(def ^:private default-transport (deref #'hub/default-transport))

(deftest default-transport-strips-authorization-across-origins
  (let [target-auth (atom :not-called)
        target (start-server
                (reify HttpHandler
                  (handle [_ exchange]
                    (reset! target-auth (.getFirst (.getRequestHeaders ^HttpExchange exchange) "Authorization"))
                    (respond! exchange 200 "target-body"))))
        origin (start-server
                (reify HttpHandler
                  (handle [_ exchange]
                    (redirect! exchange 302 (server-url target "/file")))))]
    (try
      (let [response (default-transport {:url (server-url origin "/redirect")
                                         :method "GET"
                                         :headers {"Authorization" "Bearer SECRET-HF-TOKEN"}})]
        (is (= 200 (:status response)))
        (is (= "target-body" (String. ^bytes (:body response) "UTF-8")))
        (is (nil? @target-auth)
            "Authorization must not reach a different origin"))
      (finally
        (.stop origin 0)
        (.stop target 0)))))

(deftest default-transport-preserves-authorization-and-range-within-origin
  (let [received (atom nil)
        server (start-server
                (reify HttpHandler
                  (handle [_ exchange]
                    (let [^HttpExchange exchange exchange]
                      (if (= "/redirect" (.getPath (.getRequestURI exchange)))
                        ;; Relative Location must resolve against the current URL.
                        (redirect! exchange 302 "/final")
                        (do (reset! received
                                    {:auth (.getFirst (.getRequestHeaders exchange) "Authorization")
                                     :range (.getFirst (.getRequestHeaders exchange) "Range")})
                            (respond! exchange 200 "ok")))))))]
    (try
      (let [response (default-transport {:url (server-url server "/redirect")
                                         :method "GET"
                                         :headers {"Authorization" "Bearer token"
                                                   "Range" "bytes=5-"}})]
        (is (= 200 (:status response)))
        (is (= "Bearer token" (:auth @received)))
        (is (= "bytes=5-" (:range @received))))
      (finally (.stop server 0)))))

(deftest default-transport-follows-all-redirect-statuses
  ;; 301, 302, 303, 307, 308 all matter; 303 also turns the follow-up into a
  ;; GET, which is a no-op here since every request in this namespace is GET.
  (doseq [status [301 302 303 307 308]]
    (let [server (start-server
                  (reify HttpHandler
                    (handle [_ exchange]
                      (let [^HttpExchange exchange exchange]
                        (if (= "/start" (.getPath (.getRequestURI exchange)))
                          (redirect! exchange status "/end")
                          (respond! exchange 200 "ok"))))))]
      (try
        (is (= 200 (:status (default-transport {:url (server-url server "/start")
                                                :method "GET" :headers {}})))
            (str "status " status))
        (finally (.stop server 0))))))

(deftest default-transport-caps-redirect-chain
  (let [server (start-server
                (reify HttpHandler
                  (handle [_ exchange]
                    (let [^HttpExchange exchange exchange]
                      (redirect! exchange 302 (.toString (.getRequestURI exchange)))))))]
    (try
      (is (= :too-many-redirects
             (try (default-transport {:url (server-url server "/loop")
                                      :method "GET" :headers {}})
                  (catch clojure.lang.ExceptionInfo e (:embeddings/error (ex-data e))))))
      (finally (.stop server 0)))))

(deftest same-origin-checks-scheme-host-and-port
  (let [same-origin? (deref #'hub/same-origin?)]
    (is (true? (same-origin? (URI/create "https://a.example/x") (URI/create "https://a.example/y"))))
    (is (true? (same-origin? (URI/create "https://a.example") (URI/create "https://a.example:443/y")))
        "default https port must be treated as equivalent to an explicit :443")
    (is (false? (same-origin? (URI/create "https://a.example") (URI/create "http://a.example")))
        "a scheme downgrade is a different origin even on the same host")
    (is (false? (same-origin? (URI/create "https://a.example") (URI/create "https://b.example")))
        "different host")
    (is (false? (same-origin? (URI/create "https://a.example:443") (URI/create "https://a.example:8443")))
        "different port")))

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

(deftest rejects-bad-revisions
  (doseq [bad ["../../../etc" "a/../../b" "/abs/path" "" "main " (str "main" \newline)
               (str "ma" \tab "in") (str "main" \backspace) "..\\windows" "main branch"
               "main?x=1" "main#frag" "%2e%2e" "main//dup" "refs/../pr" "main/"
               ".hidden" "-leading" "refs/heads/x.lock" "@{now}" "main~1" "main^" "a:b"]]
    (is (= {:embeddings/error :invalid-revision :revision bad}
           (try (fetch-model* "org/model" {:revision bad} (constantly true))
                (catch clojure.lang.ExceptionInfo e (ex-data e))))
        (pr-str bad)))
  (is (= :invalid-revision
         (try (fetch-model* "org/model" {:revision 42} (constantly true))
              (catch clojure.lang.ExceptionInfo e (:embeddings/error (ex-data e)))))))

(deftest accepts-legitimate-revisions
  (doseq [good ["main" "v1.0.0" "refs/pr/1" "feature/x" "my.branch-name_1"
                "0123456789abcdef0123456789abcdef01234567"]]
    (let [^java.io.File dir (tmp-dir)
          calls (atom [])
          download! (fn [url dest]
                      (swap! calls conj url)
                      (io/make-parents dest)
                      (spit dest "x")
                      true)
          path (fetch-model* "org/model" {:cache-dir (.getPath dir)
                                          :revision good}
                             download!)]
      (is (.startsWith ^String (.getCanonicalPath (io/file path)) (.getCanonicalPath dir))
          (pr-str good))
      (is (= [(str "https://huggingface.co/org/model/resolve/" good "/onnx/model.onnx")
              (str "https://huggingface.co/org/model/resolve/" good "/tokenizer.json")]
             @calls)
          (pr-str good)))))

(deftest bad-revision-never-escapes-the-cache-dir
  (let [^java.io.File dir (tmp-dir)
        _ (.mkdirs (io/file dir "org" "model"))
        dests (atom [])
        download! (fn [_ dest]
                    (swap! dests conj (.getCanonicalPath ^java.io.File dest))
                    (io/make-parents dest)
                    (spit dest "x")
                    true)]
    (is (= :invalid-revision
           (try (fetch-model* "org/model" {:cache-dir (.getPath dir)
                                           :revision "../../../PWNED"}
                              download!)
                (catch clojure.lang.ExceptionInfo e (:embeddings/error (ex-data e))))))
    (is (empty? @dests))))

(deftest bad-revision-rejected-through-the-transport-path
  (let [^java.io.File dir (tmp-dir)
        requests (atom [])
        transport (secure-transport requests {"onnx/model.onnx" "model"
                                              "tokenizer.json" "tokenizer"})]
    (is (= :invalid-revision
           (try (hub/fetch-model "org/model" {:cache-dir (.getPath dir)
                                              :revision "../../../PWNED"
                                              :transport transport})
                (catch clojure.lang.ExceptionInfo e (:embeddings/error (ex-data e))))))
    (is (empty? @requests))))

(deftest model-root-rejects-paths-outside-the-cache-dir
  ;; Defence in depth: even bypassing `validate-revision!`, a computed root
  ;; that escapes the cache dir must throw.
  (let [model-root #'hub/model-root
        ^java.io.File dir (tmp-dir)]
    (is (= :invalid-cache-path
           (try (model-root (.getPath dir) "org/model" "../../../../PWNED" nil)
                (catch clojure.lang.ExceptionInfo e (:embeddings/error (ex-data e))))))
    (is (= (.getCanonicalPath (io/file dir "org" "model"))
           (.getCanonicalPath ^java.io.File (model-root (.getPath dir) "org/model" "main" nil))))))
