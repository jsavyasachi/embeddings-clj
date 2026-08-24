(ns embeddings.providers-test
  (:require [clojure.data.json :as json]
            [clojure.test :refer [deftest is testing]]
            [embeddings.core :as embeddings]
            [embeddings.providers :as providers])
  (:import (java.io IOException)
           (java.nio ByteBuffer ByteOrder)
           (java.time ZonedDateTime ZoneOffset)
           (java.time.format DateTimeFormatter)
           (java.util Base64)))

(deftest hosted-adapters-implement-provider-test
  (doseq [provider [(providers/openai {})
                    (providers/cohere {})
                    (providers/voyage {})]]
    (is (satisfies? embeddings/EmbeddingProvider provider))))

(defn- request-body [request]
  (json/read-str (:body request)))

(defn- vectors [embeddings]
  (mapv vec embeddings))

(defn- base64-floats
  [& values]
  (let [buffer (doto (ByteBuffer/allocate (* 4 (count values)))
                 (.order ByteOrder/LITTLE_ENDIAN))]
    (doseq [value values]
      (.putFloat buffer (float value)))
    (.encodeToString (Base64/getEncoder) (.array buffer))))

(defn- thrown-data [f]
  (try
    (f)
    nil
    (catch clojure.lang.ExceptionInfo ex
      (ex-data ex))))

(defn- success-response [request]
  (let [inputs (get (request-body request) "input")]
    {:status 200
     :body (json/write-str
            {"data" (map-indexed
                      (fn [index text]
                        {"index" index
                         "embedding" [(count text) index]})
                      inputs)})}))

(deftest json-parsing-does-not-import-gson-test
  (is (not-any? #(-> ^Class % .getName (.startsWith "com.google.gson."))
                (concat (vals (ns-imports 'embeddings.core))
                        (vals (ns-imports 'embeddings.providers))))))

(deftest openai-request-and-response-test
  (let [requests (atom [])
        transport (fn [request]
                    (swap! requests conj request)
                    (let [inputs (get (request-body request) "input")]
                      {:status 200
                       :body (json/write-str
                              {"data" (map-indexed
                                       (fn [index text]
                                         {"index" index
                                          "embedding" [(count text) index]})
                                       inputs)})}))
        provider (providers/openai {:api-key "openai-key"
                                    :model "text-embedding-3-small"
                                    :dimensions 2
                                    :batch-size 2
                                    :transport transport})]
    (is (= [[1.0 0.0] [2.0 1.0] [3.0 0.0]]
           (vectors (embeddings/embed-batch provider ["a" "bb" "ccc"]))))
    (is (= 2 (count @requests)))
    (doseq [request @requests]
      (is (= "https://api.openai.com/v1/embeddings" (:url request)))
      (is (= {"Authorization" "Bearer openai-key"
              "Content-Type" "application/json"}
             (:headers request)))
      (is (= "POST" (:method request)))
      (is (= "text-embedding-3-small" (get (request-body request) "model")))
      (is (= 2 (get (request-body request) "dimensions"))))
    (is (= ["a" "bb"] (get (request-body (first @requests)) "input")))
    (is (= ["ccc"] (get (request-body (second @requests)) "input")))
    (is (= 2 (embeddings/dimension provider)))))

(deftest openai-encoding-format-options-test
  (let [request (atom nil)
        provider (providers/openai
                  {:api-key "openai-key"
                   :model "text-embedding-3-small"
                   :encoding-format "base64"
                   :transport (fn [value]
                                (reset! request value)
                                {:status 200
                                 :body (json/write-str
                                        {"data" [{"index" 0
                                                   "embedding" (base64-floats 1 2)}]})})})]
    (is (= [[1.0 2.0]]
           (vectors (embeddings/embed-batch provider ["one"]))))
    (is (= {"model" "text-embedding-3-small"
            "input" ["one"]
            "encoding_format" "base64"}
           (request-body @request)))))

(deftest cohere-request-and-response-test
  (let [request (atom nil)
        provider (providers/cohere
                  {:api-key "cohere-key"
                   :model "embed-v4.0"
                   :input-type "search_document"
                   :dimensions 2
                   :transport (fn [value]
                                (reset! request value)
                                {:status 200
                                 :body "{\"embeddings\":{\"float\":[[1,2],[3,4]]}}"})})]
    (is (= [[1.0 2.0] [3.0 4.0]]
           (vectors (embeddings/embed-batch provider ["one" "two"]))))
    (is (= "https://api.cohere.com/v2/embed" (:url @request)))
    (is (= "Bearer cohere-key" (get-in @request [:headers "Authorization"])))
    (is (= {"model" "embed-v4.0"
            "texts" ["one" "two"]
            "input_type" "search_document"
            "embedding_types" ["float"]
            "output_dimension" 2}
           (request-body @request)))
    (is (= 2 (embeddings/dimension provider)))))

(deftest cohere-request-options-test
  (let [request (atom nil)
        provider (providers/cohere
                  {:api-key "cohere-key"
                   :model "embed-v4.0"
                   :truncate "START"
                   :max-tokens 128
                   :transport (fn [value]
                                (reset! request value)
                                {:status 200
                                 :body "{\"embeddings\":{\"float\":[[1,2]]}}"})})]
    (embeddings/embed-batch provider ["one"])
    (is (= {"model" "embed-v4.0"
            "texts" ["one"]
            "embedding_types" ["float"]
            "truncate" "START"
            "max_tokens" 128}
           (request-body @request)))))

(deftest voyage-request-and-response-test
  (let [request (atom nil)
        provider (providers/voyage
                  {:api-key "voyage-key"
                   :model "voyage-3-large"
                   :input-type "query"
                   :dimensions 2
                   :transport (fn [value]
                                (reset! request value)
                                (let [inputs (get (request-body value) "input")]
                                  {:status 200
                                   :body (json/write-str
                                          {"data" (reverse
                                                    (map-indexed
                                                     (fn [index _]
                                                       {"index" index
                                                        "embedding" (if (zero? index)
                                                                      [1 2]
                                                                      [3 4])})
                                                     inputs))})}))})]
    (testing "response indexes restore the input order"
      (is (= [[1.0 2.0] [3.0 4.0]]
             (vectors (embeddings/embed-batch provider ["one" "two"])))))
    (is (= "https://api.voyageai.com/v1/embeddings" (:url @request)))
    (is (= "Bearer voyage-key" (get-in @request [:headers "Authorization"])))
    (is (= {"model" "voyage-3-large"
            "input" ["one" "two"]
            "input_type" "query"
            "output_dimension" 2}
           (request-body @request)))
    (is (= [1.0 2.0] (vec (embeddings/embed provider "one"))))
    (is (= 2 (embeddings/dimension provider)))))

(deftest voyage-output-dtype-option-test
  (let [request (atom nil)
        provider (providers/voyage
                  {:api-key "voyage-key"
                   :model "voyage-3-large"
                   :output-dtype "int8"
                   :transport (fn [value]
                                (reset! request value)
                                {:status 200
                                 :body "{\"data\":[{\"index\":0,\"embedding\":[1,2]}]}"})})]
    (is (= [[1.0 2.0]]
           (vectors (embeddings/embed-batch provider ["one"]))))
    (is (= {"model" "voyage-3-large"
            "input" ["one"]
            "output_dtype" "int8"}
           (request-body @request)))))

(deftest hosted-response-count-must-match-input-count-test
  (let [provider (providers/openai
                  {:model "model"
                   :transport (fn [_]
                                {:status 200
                                 :body "{\"data\":[{\"index\":0,\"embedding\":[1,2]}]}"})})
        data (thrown-data #(embeddings/embed-batch provider ["one" "two"]))]
    (is (= :invalid-provider-response (:embeddings/error data)))
    (is (= :count-mismatch (:reason data)))
    (is (= 2 (:expected-count data)))
    (is (= 1 (:actual-count data)))))

(deftest hosted-response-dimensions-must-be-consistent-test
  (let [provider (providers/cohere
                  {:model "model"
                   :transport (fn [_]
                                {:status 200
                                 :body "{\"embeddings\":{\"float\":[[1,2],[3]]}}"})})
        data (thrown-data #(embeddings/embed-batch provider ["one" "two"]))]
    (is (= :invalid-provider-response (:embeddings/error data)))
    (is (= :inconsistent-dimensions (:reason data)))
    (is (= [2 1] (:dimensions data)))))

(deftest hosted-provider-error-payload-is-typed-test
  (let [provider-error {"message" "rate limited"
                        "type" "rate_limit_error"}
        provider (providers/voyage
                  {:model "model"
                   :transport (fn [_]
                                {:status 429
                                 :body (json/write-str {"error" provider-error})})})
        data (thrown-data #(embeddings/embed-batch provider ["one"]))]
    (is (= :provider-error (:embeddings/error data)))
    (is (= :voyage (:provider data)))
    (is (= 429 (:status data)))
    (is (= provider-error (:provider-error data)))))

(deftest hosted-provider-custom-headers-test
  (doseq [[constructor provider-name]
          [[providers/openai :openai]
           [providers/cohere :cohere]
           [providers/voyage :voyage]]]
    (let [request (atom nil)
          provider (constructor {:api-key "key"
                                 :model "model"
                                 :headers {"X-Trace-Id" "trace"
                                           "Content-Type" "custom/type"}
                                 :transport (fn [value]
                                              (reset! request value)
                                              (if (= provider-name :cohere)
                                                {:status 200
                                                 :body "{\"embeddings\":{\"float\":[[1,2]]}}"}
                                                (success-response value)))})]
      (embeddings/embed provider "text")
      (is (= "trace" (get-in @request [:headers "X-Trace-Id"])))
      (is (= "custom/type" (get-in @request [:headers "Content-Type"]))))))

(deftest hosted-provider-retries-transient-failure-test
  (let [attempts (atom 0)
        delays (atom [])
        provider (providers/openai
                  {:api-key "key"
                   :model "model"
                   :max-retries 1
                   :retry-base-delay-ms 25
                   :retry-jitter 0
                   :sleep-fn #(swap! delays conj %)
                   :transport (fn [request]
                                (if (= 1 (swap! attempts inc))
                                  {:status 500 :body "temporary"}
                                  (success-response request)))})]
    (is (= [[4.0 0.0]] (vectors (embeddings/embed-batch provider ["text"]))))
    (is (= 2 @attempts))
    (is (= [25] @delays))))

(deftest hosted-provider-honors-retry-after-test
  (let [attempts (atom 0)
        delays (atom [])
        provider (providers/openai
                  {:api-key "key"
                   :model "model"
                   :max-retries 1
                   :retry-base-delay-ms 25
                   :retry-jitter 0
                   :sleep-fn #(swap! delays conj %)
                   :transport (fn [request]
                                (if (= 1 (swap! attempts inc))
                                  {:status 429
                                   :headers {"Retry-After" "2"}
                                   :body "rate limited"}
                                  (success-response request)))})]
    (is (= [[4.0 0.0]] (vectors (embeddings/embed-batch provider ["text"]))))
    (is (= [2000] @delays))))

(deftest hosted-provider-retries-transport-exception-test
  (let [attempts (atom 0)
        delays (atom [])
        provider (providers/openai
                  {:api-key "key"
                   :model "model"
                   :max-retries 1
                   :retry-base-delay-ms 25
                   :retry-jitter 0
                   :sleep-fn #(swap! delays conj %)
                   :transport (fn [request]
                                (if (= 1 (swap! attempts inc))
                                  (throw (IOException. "connection reset"))
                                  (success-response request)))})]
    (is (= [[4.0 0.0]] (vectors (embeddings/embed-batch provider ["text"]))))
    (is (= 2 @attempts))
    (is (= [25] @delays))))

(deftest hosted-provider-honors-http-date-retry-after-test
  (let [attempts (atom 0)
        delays (atom [])
        retry-at (-> (ZonedDateTime/now ZoneOffset/UTC)
                     (.plusSeconds 5)
                     (.withNano 0))
        provider (providers/openai
                  {:api-key "key"
                   :model "model"
                   :max-retries 1
                   :retry-base-delay-ms 25
                   :retry-jitter 0
                   :sleep-fn #(swap! delays conj %)
                   :transport (fn [request]
                                (if (= 1 (swap! attempts inc))
                                  {:status 429
                                   :headers {"Retry-After"
                                             (.format retry-at DateTimeFormatter/RFC_1123_DATE_TIME)}
                                   :body "rate limited"}
                                  (success-response request)))})]
    (is (= [[4.0 0.0]] (vectors (embeddings/embed-batch provider ["text"]))))
    (is (= 2 @attempts))
    (is (<= 3000 (first @delays) 5000))))
