import numpy as np
import pickle

glove_mat = np.load("./data/glove_mat.npy")
# words = np.load("./data/words.npy")
with open("./data/word_to_ind.pkl", "rb") as f:
    word_to_ind = pickle.load(f)

given_f, given_s, ques_f = "father", "mother", "brother"
given_s_opts = ["grandmother", "grandfather", "daughter", "sister", "mother"]

given_f_v = glove_mat[word_to_ind[given_f]]
given_s_v = glove_mat[word_to_ind[given_s]]
ques_f_v = glove_mat[word_to_ind[ques_f]]

given_diff = given_s_v - given_f_v

max_sim, max_i = 0, 0

# for i, opt in enumerate(given_s_opts):
#     opt_v = glove_mat[word_to_ind[opt]]
#     diff_vec = opt_v - ques_f_v
#     sim = given_diff @ diff_vec / np.linalg.norm(given_diff) / np.linalg.norm(diff_vec)
#     print(opt, round(sim, 4))
#     if (sim > max_sim): max_sim, max_i = sim, i

diff_mat = glove_mat - ques_f_v
diff_mat_norms = np.linalg.norm(diff_mat, axis=1, keepdims=True)
print(diff_mat.shape, diff_mat_norms.shape)

given_diff = np.reshape(given_diff, (-1, 1)) # to col vec
given_diff_norm = np.linalg.norm(given_diff, keepdims=True)
print(given_diff.shape, given_diff_norm.shape)

diff_mat_sims = diff_mat @ given_diff
print(diff_mat_sims.shape)
diff_mat_sims = diff_mat_sims / diff_mat_norms / given_diff_norm
print(diff_mat_sims.shape)

max_sim = np.amax(diff_mat_sims)
max_i = np.argmax(diff_mat_sims)

print("Prediction:", given_s_opts[max_i], round(max_sim, 4))

    