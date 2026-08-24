(ns embeddings.search-test
  (:require [clojure.test :refer [deftest is testing]]
            [embeddings.search :as search]))

(set! *warn-on-reflection* true)

(defn- farray [& xs]
  (float-array xs))

(defn- approx=
  [expected actual]
  (<= (Math/abs (- (double expected) (double actual))) 1.0E-5))

(deftest batch-score-test
  (let [query (farray 1.0 2.0)
        candidates [(farray 3.0 4.0)
                    (farray -1.0 2.0)]]
    (is (= [11.0 3.0]
           (search/dot-product-scores query candidates)))
    (is (every? true?
                (map approx=
                     [0.9838699100999075 0.6]
                     (search/cosine-similarity-scores query candidates))))))

(deftest top-k-test
  (let [candidates [:first :second :third :fourth]
        scores [0.5 0.9 0.9 0.1]]
    (is (= [{:index 1 :candidate :second :score 0.9}
            {:index 2 :candidate :third :score 0.9}]
           (search/top-k candidates scores 2)))
    (is (= 4 (count (search/top-k candidates scores 10))))
    (testing "negative k is rejected"
      (is (= :invalid-k
             (:embeddings/error
              (try
                (search/top-k candidates scores -1)
                (catch clojure.lang.ExceptionInfo ex (ex-data ex)))))))))

(deftest search-filter-and-metric-test
  (let [candidates [{:id :a :embedding (farray 1.0 0.0)}
                    {:id :b :embedding (farray 0.0 1.0)}
                    {:id :skip :embedding (farray 100.0 100.0)}]
        query (farray 1.0 0.0)]
    (is (= [:a :b]
           (mapv (comp :id :candidate)
                 (search/search query candidates
                                {:metric :cosine
                                 :k 2
                                 :vector-fn :embedding
                                 :predicate #(not= :skip (:id %))}))))))

(deftest index-test
  (let [index (search/build-index [(farray 1.0 0.0)
                                   (farray 0.0 1.0)])]
    (is (= [0 1]
           (mapv :index (search/query index (farray 1.0 0.0) {:k 2}))))
    (is (= :dim-mismatch
           (:embeddings/error
            (try
              (search/query index (farray 1.0) {})
              (catch clojure.lang.ExceptionInfo ex (ex-data ex))))))))
