(ns embeddings.providers
  "Hosted embedding providers that implement `embeddings.core/EmbeddingProvider`."
  (:require [clojure.data.json :as json]
            [embeddings.core :as embeddings])
  (:import (java.io IOException)
           (java.net URI)
           (java.net.http HttpClient HttpRequest HttpRequest$BodyPublishers
                          HttpResponse$BodyHandlers)
           (java.time Duration Instant ZonedDateTime)
           (java.time.format DateTimeFormatter)))

(set! *warn-on-reflection* true)

(defn- default-transport
  [{:keys [url method headers body]} opts]
  (let [client (-> (HttpClient/newBuilder)
                   (.connectTimeout (Duration/ofMillis
                                     (long (or (:connect-timeout-ms opts) 10000))))
                   (.build))
        builder (HttpRequest/newBuilder (URI/create ^String url))]
    (doseq [[name value] headers]
      (.header builder ^String name ^String value))
    (.timeout builder (Duration/ofMillis
                       (long (or (:request-timeout-ms opts) 60000))))
    (.method builder
             ^String method
             (HttpRequest$BodyPublishers/ofString ^String body))
    (let [response (.send client
                          (.build builder)
                          (HttpResponse$BodyHandlers/ofString))]
      {:status (.statusCode response)
       :headers (into {} (.map (.headers response)))
       :body (.body response)})))

(defn- json-map [value]
  (json/write-str value))

(defn- parse-object
  [^String body]
  (json/read-str body))

(defn- object-value
  [object key]
  (get object key))

(defn- object
  [parent key]
  (object-value parent key))

(defn- array
  [parent key]
  (object-value parent key))

(defn- float-vector
  ^floats
  [values]
  (float-array (map float values)))

(defn- indexed-response
  [response]
  (->> (array response "data")
       (map (fn [entry]
              [(int (object-value entry "index"))
               (float-vector (array entry "embedding"))]))
       (sort-by first)
       (mapv second)))

(defn- cohere-response
  [response]
  (mapv float-vector (array (object response "embeddings") "float")))

(defn- endpoint [provider opts]
  (or (:url opts)
      (case provider
        :openai "https://api.openai.com/v1/embeddings"
        :cohere "https://api.cohere.com/v2/embed"
        :voyage "https://api.voyageai.com/v1/embeddings")))

(defn- request-body
  [provider opts texts]
  (case provider
    :openai
    (cond-> {"model" (:model opts)
             "input" texts}
      (:dimensions opts) (assoc "dimensions" (:dimensions opts)))

    :cohere
    (cond-> {"model" (:model opts)
             "texts" texts
             "embedding_types" ["float"]}
      (:input-type opts) (assoc "input_type" (:input-type opts))
      (:dimensions opts) (assoc "output_dimension" (:dimensions opts)))

    :voyage
    (cond-> {"model" (:model opts)
             "input" texts}
      (:input-type opts) (assoc "input_type" (:input-type opts))
      (:dimensions opts) (assoc "output_dimension" (:dimensions opts)))))

(defn- parse-response [provider body]
  (let [response (parse-object body)]
    (case provider
      :cohere (cohere-response response)
      (:openai :voyage) (indexed-response response))))

(defn- retryable-status? [status]
  (contains? #{408 429 500 502 503 504} status))

(defn- retry-after-ms [response]
  (let [value (or (get-in response [:headers "Retry-After"])
                  (get-in response [:headers "retry-after"]))
        value (if (coll? value) (first value) value)]
    (when value
      (try
        (* 1000 (Long/parseLong ^String value))
        (catch NumberFormatException _
          (let [retry-at (ZonedDateTime/parse
                          ^String value DateTimeFormatter/RFC_1123_DATE_TIME)]
            (max 0 (.toMillis (Duration/between (Instant/now)
                                                 (.toInstant retry-at))))))))))

(defn- retry-delay-ms [opts response retry-number]
  (or (retry-after-ms response)
      (let [base (long (or (:retry-base-delay-ms opts) 100))
            maximum (long (or (:retry-max-delay-ms opts) 10000))
            jitter (double (or (:retry-jitter opts) 0.2))
            backoff (min maximum (* base (long (Math/pow 2 (dec retry-number)))))
            spread (* backoff jitter (- (* 2 (rand)) 1))]
        (long (max 0 (+ backoff spread))))))

(defn- request-headers [opts]
  (merge {"Authorization" (str "Bearer " (:api-key opts))
          "Content-Type" "application/json"}
         (:headers opts)))

(defn- request-batch
  [provider opts texts]
  (let [transport (or (:transport opts) #(default-transport % opts))
        request {:url (endpoint provider opts)
                 :method "POST"
                 :headers (request-headers opts)
                 :body (json-map (request-body provider opts texts))}
        max-retries (long (or (:max-retries opts) 3))
        response (loop [retry-number 0]
                   (let [attempt (try
                                   {:response (transport request)}
                                   (catch IOException exception
                                     {:exception exception}))]
                     (if-let [exception (:exception attempt)]
                       (if (< retry-number max-retries)
                         (do
                           ((or (:sleep-fn opts) #(Thread/sleep (long %)))
                            (retry-delay-ms opts nil (inc retry-number)))
                           (recur (inc retry-number)))
                         (throw exception))
                       (let [response (:response attempt)
                             status (long (:status response))]
                         (if (and (< retry-number max-retries)
                                  (retryable-status? status))
                           (do
                             ((or (:sleep-fn opts) #(Thread/sleep (long %)))
                              (retry-delay-ms opts response (inc retry-number)))
                             (recur (inc retry-number)))
                           response)))))
        status (long (:status response))]
    (when-not (<= 200 status 299)
      (throw (ex-info (str "embedding request failed with HTTP " status)
                      {:embeddings/error :provider-request-failed
                       :provider provider
                       :status status
                       :body (:body response)})))
    (parse-response provider (:body response))))

(declare hosted-embed-batch)

(defrecord HostedProvider [provider opts dimensions]
  embeddings/EmbeddingProvider
  (embed [hosted text]
    (embeddings/embed hosted text nil))
  (embed [hosted text call-opts]
    (first (embeddings/embed-batch hosted [text] call-opts)))
  (embed-batch [hosted texts]
    (embeddings/embed-batch hosted texts nil))
  (embed-batch [hosted texts call-opts]
    (hosted-embed-batch hosted texts call-opts))
  (dimension [_]
    (or @dimensions
        (throw (ex-info "embedding dimension is unknown until the first response"
                        {:embeddings/error :dimension-unknown
                         :provider provider})))))

(defn- hosted-embed-batch
  [{:keys [provider opts dimensions]} texts {:keys [prefix]}]
  (let [texts (if prefix (mapv #(str prefix %) texts) (vec texts))
        batch-size (long (or (:batch-size opts) 128))
        embeddings (if (empty? texts)
                     []
                     (into [] (mapcat #(request-batch provider opts (vec %)))
                           (partition-all batch-size texts)))]
    (when-let [^floats first-embedding (first embeddings)]
      (reset! dimensions (alength first-embedding)))
    embeddings))

(defn- hosted-provider [provider opts]
  (->HostedProvider provider opts (atom (:dimensions opts))))

(defn openai
  "Create an OpenAI `/v1/embeddings` provider."
  [opts]
  (hosted-provider :openai opts))

(defn cohere
  "Create a Cohere `v2/embed` provider."
  [opts]
  (hosted-provider :cohere opts))

(defn voyage
  "Create a Voyage `/v1/embeddings` provider."
  [opts]
  (hosted-provider :voyage opts))
