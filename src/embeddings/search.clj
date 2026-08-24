(ns embeddings.search
  "Brute-force similarity search over in-memory float arrays."
  (:require [embeddings.math :as math]))

(set! *warn-on-reflection* true)

(defn- invalid-k [k]
  (ex-info "k must be a non-negative integer"
           {:embeddings/error :invalid-k
            :k k}))

(defn- vector-array?
  [value]
  (instance? (Class/forName "[F") value))

(defn- require-vector
  ^floats
  [value]
  (if (vector-array? value)
    value
    (throw (ex-info "embedding must be a float array"
                    {:embeddings/error :invalid-embedding
                     :value value}))))

(defn- validate-k
  ^long
  [k]
  (when (or (not (integer? k)) (neg? (long k)))
    (throw (invalid-k k)))
  (long k))

(defn- vector-fn
  [candidate vector-selector]
  (let [value (if (nil? vector-selector)
                candidate
                (if (fn? vector-selector)
                  (vector-selector candidate)
                  (get candidate vector-selector)))]
    (require-vector value)))

(defn- dimension-mismatch
  [expected actual]
  (ex-info "vector dimensions do not match"
           {:embeddings/error :dim-mismatch
            :expected expected
            :actual actual}))

(defn- validate-dimensions!
  [candidates vector-selector]
  (when-let [first-candidate (first candidates)]
    (let [^floats first-vector (vector-fn first-candidate vector-selector)
          expected (alength first-vector)]
      (doseq [candidate (next candidates)]
        (let [^floats candidate-vector (vector-fn candidate vector-selector)
              actual (alength candidate-vector)]
          (when-not (= expected actual)
            (throw (dimension-mismatch expected actual)))))
      expected)))

(defn dot-product-scores
  "Return dot-product scores for QUERY against each candidate vector."
  [^floats query candidates]
  (let [query (require-vector query)]
    (mapv #(math/dot query (require-vector %)) candidates)))

(defn cosine-similarity-scores
  "Return cosine-similarity scores for QUERY against each candidate vector."
  [^floats query candidates]
  (let [query (require-vector query)]
    (mapv #(math/cosine-similarity query (require-vector %)) candidates)))

(defn top-k
  "Rank CANDIDATES by SCORES, descending, retaining original-order ties."
  [candidates scores k]
  (let [k (validate-k k)
        candidates (vec candidates)
        scores (vec scores)]
    (when-not (= (count candidates) (count scores))
      (throw (ex-info "candidates and scores must have the same length"
                      {:embeddings/error :length-mismatch
                       :candidates (count candidates)
                       :scores (count scores)})))
    (->> (map-indexed (fn [index [candidate score]]
                        {:index index :candidate candidate :score score})
                      (map vector candidates scores))
         (sort-by (juxt (comp - double :score) :index))
         (take k)
         vec)))

(defn- scores-for
  [metric ^floats query candidates vector-selector]
  (let [score-fn (case metric
                   :dot dot-product-scores
                   :cosine cosine-similarity-scores
                   (throw (ex-info "unknown similarity metric"
                                   {:embeddings/error :unknown-metric
                                    :metric metric})))]
    (score-fn query (mapv #(vector-fn % vector-selector) candidates))))

(defn search
  "Search CANDIDATES for QUERY.

  Options are :metric (:dot or :cosine, default :dot), :k (required),
  :predicate (applied to candidates before scoring), and :vector-fn (a
  keyword or function selecting a float[] from each candidate)."
  [^floats query candidates {:keys [metric k predicate vector-fn]
                             :or {metric :dot k 10}}]
  (let [query (require-vector query)
        candidates (vec (if predicate (filter predicate candidates) candidates))
        _ (validate-k k)]
    (top-k candidates
           (scores-for metric query candidates vector-fn)
           k)))

(defrecord VectorIndex [candidates vector-fn dimensions])

(defn build-index
  "Build a reusable brute-force index over CANDIDATES.

  An optional :vector-fn keyword or function selects each candidate's float[]."
  ([candidates]
   (build-index candidates nil))
  ([candidates vector-selector]
   (let [candidates (vec candidates)]
     (->VectorIndex candidates vector-selector
                    (validate-dimensions! candidates vector-selector)))))

(defn query
  "Run a similarity search against a previously built VectorIndex."
  [^VectorIndex index ^floats query opts]
  (let [query (require-vector query)]
    (when (and (:dimensions index)
               (not= (:dimensions index) (alength query)))
      (throw (dimension-mismatch (:dimensions index) (alength query))))
    (search query (:candidates index)
            (assoc (or opts {}) :vector-fn (:vector-fn index)))))
