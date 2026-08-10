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

(defn- request-batch
  [provider opts texts]
  (let [transport (or (:transport opts) default-transport)
        response (transport {:url (endpoint provider opts)
                             :method "POST"
                             :headers {"Authorization" (str "Bearer " (:api-key opts))
                                       "Content-Type" "application/json"}
                             :body (json-map (request-body provider opts texts))})
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
