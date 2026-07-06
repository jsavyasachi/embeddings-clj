(ns embeddings.pooling)

(set! *warn-on-reflection* true)

(defn- hidden-size
  [^"[[F" token-vecs]
  (alength ^floats (aget token-vecs 0)))

(defn mean
  [^"[[F" token-vecs ^longs mask]
  (let [n (alength token-vecs)
        hidden (hidden-size token-vecs)
        out (float-array hidden)
        token-count (loop [i 0
                           count 0]
                      (if (= i n)
                        count
                        (if (= 1 (aget mask i))
                          (let [^floats row (aget token-vecs i)]
                            (loop [j 0]
                              (when (< j hidden)
                                (aset-float out j
                                            (float (+ (double (aget out j))
                                                      (double (aget row j)))))
                                (recur (inc j))))
                            (recur (inc i) (inc count)))
                          (recur (inc i) count))))]
    (when (pos? token-count)
      (loop [j 0]
        (when (< j hidden)
          (aset-float out j (float (/ (double (aget out j)) token-count)))
          (recur (inc j)))))
    out))

(defn mean-sqrt-len
  [^"[[F" token-vecs ^longs mask]
  (let [n (alength token-vecs)
        hidden (hidden-size token-vecs)
        out (float-array hidden)
        token-count (loop [i 0
                           count 0]
                      (if (= i n)
                        count
                        (if (= 1 (aget mask i))
                          (let [^floats row (aget token-vecs i)]
                            (loop [j 0]
                              (when (< j hidden)
                                (aset-float out j
                                            (float (+ (double (aget out j))
                                                      (double (aget row j)))))
                                (recur (inc j))))
                            (recur (inc i) (inc count)))
                          (recur (inc i) count))))]
    (when (pos? token-count)
      (let [denom (Math/sqrt (double token-count))]
        (loop [j 0]
          (when (< j hidden)
            (aset-float out j (float (/ (double (aget out j)) denom)))
            (recur (inc j))))))
    out))

(defn cls
  [^"[[F" token-vecs ^longs _mask]
  (aclone ^floats (aget token-vecs 0)))

(defn max-pool
  [^"[[F" token-vecs ^longs mask]
  (let [n (alength token-vecs)
        hidden (hidden-size token-vecs)
        out (float-array hidden)]
    (loop [i 0
           seen? false]
      (if (= i n)
        out
        (if (= 1 (aget mask i))
          (let [^floats row (aget token-vecs i)]
            (loop [j 0]
              (when (< j hidden)
                (when (or (not seen?)
                          (> (float (aget row j)) (float (aget out j))))
                  (aset-float out j (aget row j)))
                (recur (inc j))))
            (recur (inc i) true))
          (recur (inc i) seen?))))))

(defn pool
  [strategy token-vecs mask]
  (case strategy
    :mean (mean token-vecs mask)
    :mean-sqrt-len (mean-sqrt-len token-vecs mask)
    :cls (cls token-vecs mask)
    :max (max-pool token-vecs mask)
    (throw (ex-info "unknown pooling strategy"
                    {:embeddings/error :unknown-pooling
                     :strategy strategy}))))
