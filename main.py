import numpy as np
import pickle

glove_mat = np.load("glove_mat.npya")
with open("words_to_ind.pkl", "rb") as f:
    words_to_ind = pickle.load(f)


