import numpy as np
from tqdm import tqdm
import pickle

def load_w2v_embedding(path, dim, word_index):  #word_index是 {单词：索引}字典
	print('Generating word2vector embedding...')
	embeddings_index = pickle.load(open(path, 'rb'))

	embedding_matrix = np.zeros((len(word_index) + 1, dim))
	
	for word, i in word_index.items():
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
	        # words not found in embedding index will be all-zeros.
	        embedding_matrix[i] = embedding_vector
	print('Loaded word2vector embedding')

	return embedding_matrix