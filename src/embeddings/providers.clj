(ns embeddings.providers
  "Hosted embedding providers that implement `embeddings.core/EmbeddingProvider`."
  (:require [clojure.data.json :as json]
            [embeddings.core :as embeddings])
  (:import (java.net URI)
           (java.net.http HttpClient HttpRequest HttpRequest$BodyPublishers
                          HttpResponse$BodyHandlers)))

(set! *warn-on-reflection* true)

(defn- default-transport
  [{:keys [url method headers body]}]
  (let [builder (HttpRequest/newBuilder (URI/create ^String url))]
    (doseq [[name value] headers]
      (.header builder ^String name ^String value))
    (.method builder
             ^String method
             (HttpRequest$BodyPublishers/ofString ^String body))
    (let [response (.send (HttpClient/newHttpClient)
                          (.build builder)
                          (HttpResponse$BodyHandlers/ofString))]
      {:status (.statusCode response)
       :body (.body response)})))

(defn- json-map [value]
  (json/write-str value))

(defn- parse-object
  [^String body]
  (json/read-str body))

(declare object-value)

(defn- invalid-response
  [provider reason data]
  (throw (ex-info "invalid embedding provider response"
                  (merge {:embeddings/error :invalid-provider-response
                          :provider provider
                          :reason reason}
                         data))))

(defn- provider-error-payload
  [response]
  (when (map? response)
    (or (object-value response "error")
        (when (and (or (contains? response "message")
                       (contains? response "detail"))
                   (not (contains? response "data"))
                   (not (contains? response "embeddings")))
          response))))

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
  (if (sequential? values)
    (float-array (map (fn [value]
                        (if (number? value)
                          (float value)
                          (throw (ex-info "embedding contains a non-numeric value"
                                          {:value value}))))
                      values))
    (throw (ex-info "embedding is not an array" {}))))

(defn- validate-embeddings
  [provider expected-count embeddings]
  (let [actual-count (count embeddings)]
    (when (not= expected-count actual-count)
      (invalid-response provider :count-mismatch
                        {:expected-count expected-count
                         :actual-count actual-count}))
    (let [dimensions (mapv (fn [^floats embedding]
                             (alength embedding))
                           embeddings)]
      (when (and (seq dimensions)
                 (not (apply = dimensions)))
        (invalid-response provider :inconsistent-dimensions
                          {:dimensions dimensions})))
    embeddings))

(defn- indexed-response
  [provider response expected-count]
  (let [data (array response "data")]
    (if (sequential? data)
      (->> data
           (map (fn [entry]
                  (if (map? entry)
                    (let [index (object-value entry "index")]
                      (if (number? index)
                        [(int index)
                         (try
                           (float-vector (array entry "embedding"))
                           (catch clojure.lang.ExceptionInfo ex
                             (invalid-response provider :invalid-embedding
                                               {:cause (ex-data ex)})))]
                        (invalid-response provider :invalid-entry
                                          {:entry entry})))
                    (invalid-response provider :invalid-entry
                                      {:entry entry}))))
           (sort-by first)
           (mapv second)
           (validate-embeddings provider expected-count))
      (invalid-response provider :invalid-data {:data data}))))

(defn- cohere-response
  [provider response expected-count]
  (let [embeddings (array (object response "embeddings") "float")]
    (if (sequential? embeddings)
      (try
        (validate-embeddings provider expected-count (mapv float-vector embeddings))
        (catch clojure.lang.ExceptionInfo ex
          (if (= :invalid-provider-response
                 (:embeddings/error (ex-data ex)))
            (throw ex)
            (invalid-response provider :invalid-embedding
                              {:cause (ex-data ex)}))))
      (invalid-response provider :invalid-embeddings
                        {:embeddings embeddings}))))

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

(defn- parse-response [provider response expected-count]
    (case provider
      :cohere (cohere-response provider response expected-count)
      (:openai :voyage) (indexed-response provider response expected-count)))

(defn- request-batch
  [provider opts texts]
  (let [transport (or (:transport opts) default-transport)
        response (transport {:url (endpoint provider opts)
                             :method "POST"
                             :headers {"Authorization" (str "Bearer " (:api-key opts))
                                       "Content-Type" "application/json"}
                             :body (json-map (request-body provider opts texts))})
        status (long (:status response))
        parsed (try
                 (parse-object (:body response))
                 (catch Exception ex
                   (invalid-response provider :malformed-json
                                     {:status status
                                      :body (:body response)
                                      :cause (.getMessage ex)})))]
    (when-let [provider-error (provider-error-payload parsed)]
      (throw (ex-info "embedding provider returned an error"
                      {:embeddings/error :provider-error
                       :provider provider
                       :status status
                       :provider-error provider-error
                       :body (:body response)})))
    (when-not (<= 200 status 299)
      (throw (ex-info (str "embedding request failed with HTTP " status)
                      {:embeddings/error :provider-request-failed
                       :provider provider
                       :status status
                       :body (:body response)})))
    (parse-response provider parsed (count texts))))

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
    (validate-embeddings provider (count texts) embeddings)
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
