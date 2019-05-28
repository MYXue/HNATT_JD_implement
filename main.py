import util.jdData as jd
from hnatt import HNATT
from sklearn.metrics import confusion_matrix

JD_DATA_PATH = 'data_jd/extract_comments1.txt'
SAVED_MODEL_DIR = 'saved_models'
SAVED_MODEL_FILENAME = 'model.h5'
EMBEDDINGS_PATH = 'saved_models/w2v_vec.pkl'

if __name__ == '__main__':
	(train_x, train_y), (test_x, test_y) = jd.load_data(path=JD_DATA_PATH, size=1e4, binary=True) #读取10000条数据

	# print ("train_x from main:",train_x)
	# initialize HNATT 
	h = HNATT()	
	h.train(train_x, train_y, 
		batch_size=16,
		epochs=16,
		embeddings_path=EMBEDDINGS_PATH, 
		saved_model_dir=SAVED_MODEL_DIR,
		saved_model_filename=SAVED_MODEL_FILENAME)

	h.load_weights(SAVED_MODEL_DIR, SAVED_MODEL_FILENAME)

	# embeddings = h.word_embeddings(train_x)
	# preds = h.predict(train_x)
	# print(preds)
	# import pdb; pdb.set_trace()

	
	pred_y = h.predict(test_x)
	matrix = confusion_matrix(test_y.argmax(axis=1), pred_y.argmax(axis=1))
	# print ("pred_y:\n",pred_y)
	# print ("test_y:\n",test_y)
	print ("\n\nprediction on the test data, confusion matrix:")
	print (matrix)


	# print attention activation maps across sentences and words per sentence'they have some pretty interesting things here. i will definitely go back again.'
	testSentence = "喜欢在京东买东西，因为今天买明天就可以送到。我为什么每个商品的评价都一样，而且京东购买的东西品质很有保证，价格也很合适。"
	print ("\n\nprint attention activation maps across below sentence:")
	print (testSentence)
	activation_maps = h.activation_maps(testSentence)
	print(activation_maps) 


