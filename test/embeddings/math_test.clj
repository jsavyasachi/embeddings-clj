(ns embeddings.math-test
  (:require [clojure.test :refer [deftest is testing]]
            [embeddings.math :as math]))

(def ^:private epsilon 1.0E-5)

(defn approx=
  [x y]
  (if (and (instance? (Class/forName "[F") x)
           (instance? (Class/forName "[F") y))
    (let [^floats xs x
          ^floats ys y]
      (and (= (alength xs) (alength ys))
           (loop [i 0]
             (or (= i (alength xs))
                 (and (approx= (aget xs i) (aget ys i))
                      (recur (inc i)))))))
    (<= (Math/abs (- (double x) (double y))) epsilon)))

(defn farray
  [& xs]
  (float-array xs))

(deftest dot-test
  (is (approx= 32.0 (math/dot (farray 1.0 2.0 3.0)
                              (farray 4.0 5.0 6.0))))
  (testing "dimension mismatch"
    (try
      (math/dot (farray 1.0 2.0) (farray 1.0))
      (is false "expected exception")
      (catch clojure.lang.ExceptionInfo ex
        (is (= {:embeddings/error :dim-mismatch
                :expected 2
                :actual 1}
               (ex-data ex)))
        (is (= "vector dimensions do not match" (ex-message ex)))))))

(deftest norm-test
  (is (approx= 5.0 (math/norm (farray 3.0 4.0))))
  (is (approx= 0.0 (math/norm (farray 0.0 0.0)))))

(deftest l2-normalize-test
  (let [input (farray 3.0 4.0)
        normalized (math/l2-normalize input)]
    (is (approx= (farray 0.6 0.8) normalized))
    (is (not (identical? input normalized)))
    (is (approx= (farray 3.0 4.0) input)))
  (let [input (farray 0.0 0.0)
        normalized (math/l2-normalize input)]
    (is (approx= input normalized))
    (is (not (identical? input normalized)))))

(deftest cosine-similarity-test
  (is (approx= 1.0 (math/cosine-similarity (farray 1.0 2.0 3.0)
                                           (farray 1.0 2.0 3.0))))
  (is (approx= 0.0 (math/cosine-similarity (farray 1.0 0.0)
                                           (farray 0.0 1.0))))
  (is (approx= -1.0 (math/cosine-similarity (farray 1.0 0.0)
                                            (farray -1.0 0.0))))
  (is (approx= 0.0 (math/cosine-similarity (farray 0.0 0.0)
                                           (farray 1.0 2.0))))
  (testing "dimension mismatch"
    (try
      (math/cosine-similarity (farray 1.0 2.0) (farray 1.0))
      (is false "expected exception")
      (catch clojure.lang.ExceptionInfo ex
        (is (= :dim-mismatch (:embeddings/error (ex-data ex))))
        (is (= "vector dimensions do not match" (ex-message ex)))))))
