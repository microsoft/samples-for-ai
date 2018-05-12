import numpy as np
import os

file_path = "glove.840B.300d.txt"

vectors = {}
with open(file_path, encoding='utf8') as f:
    for line in f:
        line_split = line.strip().split(" ")
        vec = np.array(line_split[-300:], dtype=float)
        word = line_split[0]

        for char in word:
            if char in vectors:
                vectors[char] = (vectors[char][0] + vec,
                                    vectors[char][1] + 1)
            else:
                vectors[char] = (vec, 1)

base_name = os.path.splitext(os.path.basename(file_path))[0] + '-char.txt'
with open(base_name, 'w', encoding='utf8') as f2:
    for word in vectors:
        avg_vector = np.round(
            (vectors[word][0] / vectors[word][1]), 6).tolist()
        f2.write(word + " " + " ".join(str(x) for x in avg_vector) + "\n")