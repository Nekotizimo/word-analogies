import numpy as np
import pickle

READVECS = True if input("Read word vectors? (y/n): ") == "y" else False
READWORDS = True if input("Read words? (y/n): ") == "y" else False

if READVECS:
    # with open("./glove_6B_50d.txt") as f:
    #     for i, line in enumerate(f):
    #         if (i == 100194 - 1 or i == 148623 - 1 or i == 166545 - 1):
    #             print(line)
    #         if (i > 166545): break
    glove_mat = np.loadtxt("./data/glove_6B_50d.txt", usecols=(range(1, 301)), comments=None)
    print(glove_mat.shape)
    np.save("./data/glove_6B_50d.npy", glove_mat)

if READWORDS:
    words = np.loadtxt("./data/glove_6B_50d.txt", dtype=np.str, comments=None, usecols=0)
    print(words.shape)
    word_to_ind = {}
    for i, word in enumerate(words):
        word_to_ind[word] = i
    print(word_to_ind["the"])
    with open("./data/word_to_ind_6B_50d.pkl", "wb") as f:    
        pickle.dump(word_to_ind, f, pickle.HIGHEST_PROTOCOL)
    np.save("./data/words_6B_50d", words)