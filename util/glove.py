import numpy as np
from tqdm import tqdm

def load_w2v_embedding(path, dim, word_index):  #word_index是 {单词：索引}字典
	embeddings_index = {} #{单词：词向量}
	f = open(path)

	print('Generating GloVe embedding...')
	for line in tqdm(f):
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
	f.close()

	embedding_matrix = np.zeros((len(word_index) + 1, dim))
	
	for word, i in word_index.items():
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
	        # words not found in embedding index will be all-zeros.
	        embedding_matrix[i] = embedding_vector
	print('Loaded GloVe embedding')

	return embedding_matrix