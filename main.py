import util.jdData as jd
import util.weibo_rumorData as weibo
from hnatt import HNATT
from sklearn.metrics import confusion_matrix,classification_report

RUMOR_DATA_PATH_1 = 'data_rumors_weibo/real_rumor_data.csv'
RUMOR_DATA_PATH_2 = 'data_rumors_weibo/rumor_data_24.csv'
SAVED_MODEL_DIR = 'saved_models'
SAVED_MODEL_FILENAME = 'model.h5'
EMBEDDINGS_PATH = 'D:/disaster/weibo_word2vec/200/' + 'weibo_59g_embedding_200.model'

if __name__ == '__main__':
	(train_x, train_y), (test_x, test_y) = weibo.load_data(path=RUMOR_DATA_PATH_2) #读取10000条数据
	print("from main(): data loading done")

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
	print ("\n\nprediction on the test data")
	print("\n------------confusion_matrix------------")
	print(confusion_matrix(test_y.argmax(axis=1), pred_y.argmax(axis=1)))
	print("\n------------classification_report------------")
	print(classification_report(test_y.argmax(axis=1), pred_y.argmax(axis=1)))


	# print attention activation maps across sentences and words per sentence'they have some pretty interesting things here. i will definitely go back again.'
	testSentence = "发生在安徽太和强拆一家七口灭门惨案，当地有关部门已封杀，速传！！"
	print ("\n\nprint attention activation maps across below sentence:")
	print (testSentence)
	activation_maps = h.activation_maps(testSentence)
	print(activation_maps) 


