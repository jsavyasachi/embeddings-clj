(ns embeddings.math)

(set! *warn-on-reflection* true)

(defn- dimension-mismatch
  [expected actual]
  (ex-info "vector dimensions do not match"
           {:embeddings/error :dim-mismatch
            :expected expected
            :actual actual}))

(defn dot
  ^double
  [^floats a ^floats b]
  (let [n (alength a)
        m (alength b)]
    (when-not (= n m)
      (throw (dimension-mismatch n m)))
    (loop [i 0
           acc 0.0]
      (if (= i n)
        acc
        (recur (inc i)
               (+ acc (* (double (aget a i))
                         (double (aget b i)))))))))

(defn norm
  ^double
  [^floats a]
  (let [n (alength a)]
    (loop [i 0
           acc 0.0]
      (if (= i n)
        (Math/sqrt acc)
        (let [x (double (aget a i))]
          (recur (inc i) (+ acc (* x x))))))))

(defn l2-normalize
  [^floats a]
  (let [out (aclone a)
        n (norm a)]
    (when-not (zero? n)
      (let [length (alength out)]
        (loop [i 0]
          (when (< i length)
            (aset-float out i (float (/ (double (aget out i)) n)))
            (recur (inc i))))))
    out))

(defn cosine-similarity
  ^double
  [^floats a ^floats b]
  (let [n (alength a)
        m (alength b)]
    (when-not (= n m)
      (throw (dimension-mismatch n m)))
    (loop [i 0
           dot-acc 0.0
           a-acc 0.0
           b-acc 0.0]
      (if (= i n)
        (let [denom (* (Math/sqrt a-acc) (Math/sqrt b-acc))]
          (if (zero? denom)
            0.0
            (max -1.0 (min 1.0 (/ dot-acc denom)))))
        (let [x (double (aget a i))
              y (double (aget b i))]
          (recur (inc i)
                 (+ dot-acc (* x y))
                 (+ a-acc (* x x))
                 (+ b-acc (* y y))))))))
