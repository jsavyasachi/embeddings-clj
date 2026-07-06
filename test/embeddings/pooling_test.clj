(ns embeddings.pooling-test
  (:require [clojure.test :refer [deftest is testing]]
            [embeddings.pooling :as pooling]))

(def ^:private epsilon 1.0E-5)

(defn approx=
  [^floats xs ^floats ys]
  (and (= (alength xs) (alength ys))
       (loop [i 0]
         (or (= i (alength xs))
             (and (<= (Math/abs (- (double (aget xs i))
                                  (double (aget ys i))))
                      epsilon)
                  (recur (inc i)))))))

(defn row
  [& xs]
  (float-array xs))

(defn tokens
  [& rows]
  (into-array (Class/forName "[F") rows))

(deftest mean-test
  (let [token-vecs (tokens (row 1.0 2.0)
                           (row 3.0 6.0)
                           (row 100.0 200.0))
        pooled (pooling/mean token-vecs (long-array [1 1 0]))]
    (is (approx= (row 2.0 4.0) pooled))
    (is (not-any? #(identical? pooled %) token-vecs))))

(deftest mean-sqrt-len-test
  (let [token-vecs (tokens (row 1.0 2.0)
                           (row 3.0 6.0)
                           (row 100.0 200.0))
        pooled (pooling/mean-sqrt-len token-vecs (long-array [1 1 0]))
        denom (Math/sqrt 2.0)]
    (is (approx= (row (/ 4.0 denom) (/ 8.0 denom)) pooled))
    (is (not-any? #(identical? pooled %) token-vecs))))

(deftest cls-test
  (let [first-row (row 1.0 2.0)
        token-vecs (tokens first-row
                           (row 3.0 6.0)
                           (row 100.0 200.0))
        pooled (pooling/cls token-vecs (long-array [0 1 1]))]
    (is (approx= first-row pooled))
    (is (not (identical? first-row pooled)))))

(deftest max-pool-test
  (let [token-vecs (tokens (row 1.0 7.0)
                           (row 3.0 6.0)
                           (row 100.0 200.0))
        pooled (pooling/max-pool token-vecs (long-array [1 1 0]))]
    (is (approx= (row 3.0 7.0) pooled))
    (is (not-any? #(identical? pooled %) token-vecs))))

(deftest all-zero-mask-test
  (let [token-vecs (tokens (row 1.0 7.0)
                           (row 3.0 6.0))]
    (is (approx= (row 0.0 0.0)
                 (pooling/mean token-vecs (long-array [0 0]))))
    (is (approx= (row 0.0 0.0)
                 (pooling/mean-sqrt-len token-vecs (long-array [0 0]))))
    (is (approx= (row 0.0 0.0)
                 (pooling/max-pool token-vecs (long-array [0 0]))))))

(deftest pool-dispatch-test
  (let [token-vecs (tokens (row 1.0 7.0)
                           (row 3.0 6.0))
        mask (long-array [1 1])]
    (is (approx= (pooling/mean token-vecs mask)
                 (pooling/pool :mean token-vecs mask)))
    (is (approx= (pooling/mean-sqrt-len token-vecs mask)
                 (pooling/pool :mean-sqrt-len token-vecs mask)))
    (is (approx= (pooling/cls token-vecs mask)
                 (pooling/pool :cls token-vecs mask)))
    (is (approx= (pooling/max-pool token-vecs mask)
                 (pooling/pool :max token-vecs mask)))))

(deftest unknown-strategy-test
  (try
    (pooling/pool :median (tokens (row 1.0 2.0)) (long-array [1]))
    (is false "expected exception")
    (catch clojure.lang.ExceptionInfo ex
      (is (= {:embeddings/error :unknown-pooling
              :strategy :median}
             (ex-data ex)))
      (is (= "unknown pooling strategy" (ex-message ex))))))
