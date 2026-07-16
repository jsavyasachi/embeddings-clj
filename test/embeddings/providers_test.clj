(ns embeddings.providers-test
  (:require [clojure.data.json :as json]
            [clojure.test :refer [deftest is testing]]
            [embeddings.core :as embeddings]
            [embeddings.providers :as providers]))

(deftest hosted-adapters-implement-provider-test
  (doseq [provider [(providers/openai {})
                    (providers/cohere {})
                    (providers/voyage {})]]
    (is (satisfies? embeddings/EmbeddingProvider provider))))

(defn- request-body [request]
  (json/read-str (:body request)))

(defn- vectors [embeddings]
  (mapv vec embeddings))

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

(deftest voyage-request-and-response-test
  (let [request (atom nil)
        provider (providers/voyage
                  {:api-key "voyage-key"
                   :model "voyage-3-large"
                   :input-type "query"
                   :dimensions 2
                   :transport (fn [value]
                                (reset! request value)
                                {:status 200
                                 :body "{\"data\":[{\"index\":1,\"embedding\":[3,4]},{\"index\":0,\"embedding\":[1,2]}]}"})})]
    (testing "response indexes restore input order"
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
