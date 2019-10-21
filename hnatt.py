import datetime, pickle, os
import numpy as np
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.utils import CustomObjectScope
from keras.engine.topology import Layer
from keras import initializers

from util.text_util import normalize_ch
from util.w2v import load_w2v_embedding

# Uncomment below for debugging
# from tensorflow.python import debug as tf_debug
# sess = K.get_session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)


class Attention(Layer): #自定义attention层
	def __init__(self, regularizer=None, **kwargs):
		super(Attention, self).__init__(**kwargs)
		self.regularizer = regularizer
		self.supports_masking = True

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.context = self.add_weight(name='context', 
									   shape=(input_shape[-1], 1), #shape:(词向量维数，1)
									   initializer=initializers.RandomNormal(
									   		mean=0.0, stddev=0.05, seed=None),
									   regularizer=self.regularizer,
									   trainable=True)
		super(Attention, self).build(input_shape) #调用了超类的build函数，添加了一个可训练的context参数

	def call(self, x, mask=None): #Attention层的运算逻辑，x是输入的tensor, 句子层面为例，输入是一个句子中所有词的词向量表示
		attention_in = K.exp(K.squeeze(K.dot(x, self.context), axis=-1)) #比照X和context向量
		attention = attention_in/K.expand_dims(K.sum(attention_in, axis=-1), -1) #计算权重

		if mask is not None:
			# use only the inputs specified by the mask
			# import pdb; pdb.set_trace()
			attention = attention*K.cast(mask, 'float32')

		weighted_sum = K.batch_dot(K.permute_dimensions(x, [0, 2, 1]), attention) #计算加权和，permute_dimensions函数：按照给定的模式重排一个张量的轴
		return weighted_sum #本层的返回值：经过attention加权后的上层隐状态之和，用以表示这个句子/这个文档

	def compute_output_shape(self, input_shape):
		# print(input_shape)
		return (input_shape[0], input_shape[-1]) #如果输入是shape是(5,20,200),5个句子为一批，每个句子20个词，每个词用200维向量表示，最后的输出就是(5,200)

class HNATT(): # 此处不是一个layer的子类，而是相当于定义了一整个神经网络模型
	def __init__(self):
		self.model = None
		self.MAX_SENTENCE_LENGTH = 0  #最大句子长度(一句最多有几个词)
		self.MAX_SENTENCE_COUNT = 0  #最大文章长度(一篇最多有几个句子)
		self.VOCABULARY_SIZE = 0
		self.word_embedding = None
		self.model = None
		self.word_attention_model = None
		self.tokenizer = None
		self.class_count = 2

	def _generate_embedding(self, path, dim): #加载w2v矩阵，生成词向量矩阵
		return load_w2v_embedding(path, dim, self.tokenizer.word_index)

	def _build_model(self, n_classes=2, embedding_dim=200, embeddings_path=False): #embeddings_path由train()函数传参
		l2_reg = regularizers.l2(1e-8) #正则项
		# embedding_weights = np.random.normal(0, 1, (len(self.tokenizer.word_index) + 1, embedding_dim))
		# embedding_weights = np.zeros((len(self.tokenizer.word_index) + 1, embedding_dim))
		embedding_weights = np.random.normal(0, 1, (len(self.tokenizer.word_index) + 1, embedding_dim)) #标准正态分布生成一个矩阵
		if embeddings_path:
			embedding_weights = self._generate_embedding(embeddings_path, embedding_dim) #由单词索引决定词向量

		# Generate word-attention-weighted sentence scores  # 在句子层面使用注意力机制，决定句子中每个词的注意力权重，然后用加权和表示句子
		sentence_in = Input(shape=(self.MAX_SENTENCE_LENGTH,), dtype='int32') #特殊的input层，其中每个样本/document的长度即句子长度MAX_SENTENCE_LENGTH
		embedded_word_seq = Embedding(
			self.VOCABULARY_SIZE,
			embedding_dim,
			weights=[embedding_weights],
			input_length=self.MAX_SENTENCE_LENGTH,
			trainable=True,
			mask_zero=True,
			name='word_embeddings',)(sentence_in) # embedding层,嵌入层将正整数（下标）转换为具有固定大小的向量;第一个参数是输入数据的最大下标加1
		word_encoder = Bidirectional(
			GRU(150, return_sequences=True, kernel_regularizer=l2_reg))(embedded_word_seq) #双向GRU层
		dense_transform_w = Dense(
			200, 
			activation='relu', 
			name='dense_transform_w', 
			kernel_regularizer=l2_reg)(word_encoder) #全连接(embedding层)
		attention_weighted_sentence = Model(
			sentence_in, Attention(name='word_attention', regularizer=l2_reg)(dense_transform_w)) #全连接层的输出给注意力层
		self.word_attention_model = attention_weighted_sentence
		attention_weighted_sentence.summary()

		# Generate sentence-attention-weighted document scores  #在文档层面使用attention机制，决定每个句子的权重，用加权和表示整篇文本
		#TimeDistributed使用时间序列来处理张量信息，源码中RNN等的实现方式；
		texts_in = Input(shape=(self.MAX_SENTENCE_COUNT, self.MAX_SENTENCE_LENGTH), dtype='int32') #注意此处的input shape
		attention_weighted_sentences = TimeDistributed(attention_weighted_sentence)(texts_in) #这里是对于text in 的每一句话都训练attention_weighted_sentence模型，且它们使用相同的参数,最后这一层的输出相当于原始论文中的s向量
		sentence_encoder = Bidirectional(
			GRU(150, return_sequences=True, kernel_regularizer=l2_reg))(attention_weighted_sentences) #双向GRU
		dense_transform_s = Dense(
			200, 
			activation='relu', 
			name='dense_transform_s',
			kernel_regularizer=l2_reg)(sentence_encoder)	#全连接层
		attention_weighted_text = Attention(name='sentence_attention', regularizer=l2_reg)(dense_transform_s)  #attention层
		prediction = Dense(n_classes, activation='softmax')(attention_weighted_text)  #全连接层
		model = Model(texts_in, prediction)  #最终模型
		model.summary()

		model.compile(#optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),
					  #optimizer=SGD(lr=0.01, decay=1e-6, nesterov=True),
					  optimizer=Adam(lr=0.001),
		              loss='categorical_crossentropy',
		              metrics=['acc'])

		return model

	def load_weights(self, saved_model_dir, saved_model_filename):
		with CustomObjectScope({'Attention': Attention}):
			self.model = load_model(os.path.join(saved_model_dir, saved_model_filename))
			self.word_attention_model = self.model.get_layer('time_distributed_1').layer
			tokenizer_path = os.path.join(
				saved_model_dir, self._get_tokenizer_filename(saved_model_filename))
			tokenizer_state = pickle.load(open(tokenizer_path, "rb" ))
			self.tokenizer = tokenizer_state['tokenizer']
			self.MAX_SENTENCE_COUNT = tokenizer_state['maxSentenceCount']
			self.MAX_SENTENCE_LENGTH = tokenizer_state['maxSentenceLength']
			self.VOCABULARY_SIZE = tokenizer_state['vocabularySize']
			self._create_reverse_word_index()

	def _get_tokenizer_filename(self, saved_model_filename):
		return saved_model_filename + '.tokenizer'

	def _fit_on_texts(self, texts):
		self.tokenizer = Tokenizer(filters='"()*,-/;[\]^_`{|}~');
		all_sentences = []
		max_sentence_count = 0
		max_sentence_length = 0
		for text in texts:
			sentence_count = len(text)
			if sentence_count > max_sentence_count:
				max_sentence_count = sentence_count
			for sentence in text:
				sentence_length = len(sentence)
				if sentence_length > max_sentence_length:
					max_sentence_length = sentence_length
				all_sentences.append(sentence)

		self.MAX_SENTENCE_COUNT = min(max_sentence_count, 10)  #一篇中的最大句子数量(不超过10)
		self.MAX_SENTENCE_LENGTH = min(max_sentence_length, 30) #一个句子中最大的单词数量(不超过30)
		self.tokenizer.fit_on_texts(all_sentences)
		self.VOCABULARY_SIZE = len(self.tokenizer.word_index) + 1 # texts中所有单词的数量
		self._create_reverse_word_index()

		# print ("self.tokenizer.word_index:\n",self.tokenizer.word_index)
		print ("self.MAX_SENTENCE_COUNT:",self.MAX_SENTENCE_COUNT)
		print ("self.MAX_SENTENCE_LENGTH):",self.MAX_SENTENCE_LENGTH)

	def _create_reverse_word_index(self):
		self.reverse_word_index = {value:key for key,value in self.tokenizer.word_index.items()} #单词：索引

	def _encode_texts(self, texts):
		# print ("texts form _encode_texts:",texts)
		encoded_texts = np.zeros((len(texts), self.MAX_SENTENCE_COUNT, self.MAX_SENTENCE_LENGTH))
		for i, text in enumerate(texts):
			# print ("i:",i)
			# print ("text:",text)
			if len(text) != 0:
				encoded_text = np.array(pad_sequences( #pad_sequences将多个序列截断或补齐为相同长度，这里是把每一句都固定为长度是MAX_SENTENCE_LENGTH
					self.tokenizer.texts_to_sequences(text), #将单词转化为索引形式
					maxlen=self.MAX_SENTENCE_LENGTH))[:self.MAX_SENTENCE_COUNT] 
				encoded_texts[i][-len(encoded_text):] = encoded_text #一段话不够MAX_SENTENCE_COUNT句的前面一些句全是0向量			

		# print ("texts:", texts)
		# print ("encoded_texts",encoded_texts)
		return encoded_texts

	def _save_tokenizer_on_epoch_end(self, path, epoch):
		if epoch == 0:
			tokenizer_state = {
				'tokenizer': self.tokenizer,
				'maxSentenceCount': self.MAX_SENTENCE_COUNT,
				'maxSentenceLength': self.MAX_SENTENCE_LENGTH,
				'vocabularySize': self.VOCABULARY_SIZE
			}
			pickle.dump(tokenizer_state, open(path, "wb" ) )  #dump tokenizer_state一个字典，包含4项信息

	def train(self, train_x, train_y, 
		batch_size=16, epochs=1, 
		embedding_dim=200,
		embeddings_path=False, 
		saved_model_dir='saved_models', saved_model_filename=None,):
		# fit tokenizer
		self._fit_on_texts(train_x) #创建训练集文本中的 {单词：索引} 字典
		self.model = self._build_model(
			n_classes=train_y.shape[-1],  #每个y的最后一维是one-hot之后的向量，有几维说明有几类
			embedding_dim=200,
			embeddings_path=embeddings_path)
		# print ("train_x", train_x)
		encoded_train_x = self._encode_texts(train_x) #对输入做处理，单词索引化，每个输入样本变成大小相同的矩阵
		callbacks = [
			# EarlyStopping(
			# 	monitor='acc',
			# 	patience=2,
			# ),
			ReduceLROnPlateau(),
			# keras.callbacks.TensorBoard(
			# 	log_dir="logs/final/{}".format(datetime.datetime.now()), 
			# 	histogram_freq=1, 
			# 	write_graph=True, 
			# 	write_images=True
			# )
			LambdaCallback(
				on_epoch_end=lambda epoch, logs: self._save_tokenizer_on_epoch_end(
					os.path.join(saved_model_dir, 
						self._get_tokenizer_filename(saved_model_filename)), epoch))
		]

		if saved_model_filename:
			callbacks.append(
				ModelCheckpoint(
					filepath=os.path.join(saved_model_dir, saved_model_filename),
					monitor='val_acc',
					save_best_only=True,
					save_weights_only=False,
				)
			)
		self.model.fit(x=encoded_train_x, y=train_y, 
					   batch_size=batch_size, 
					   epochs=epochs, 
					   verbose=1, 
					   callbacks=callbacks,
					   validation_split=0.1,  
					   shuffle=True)

	def _encode_input(self, x, log=False):
		x = np.array(x)
		if not x.shape:
			x = np.expand_dims(x, 0)
		texts = np.array([normalize_ch(text) for text in x])
		return self._encode_texts(texts)

	def predict(self, x):
		encoded_x = self._encode_texts(x)
		return self.model.predict(encoded_x)

	def activation_maps(self, text, websafe=False):
		normalized_text = normalize_ch(text)
		encoded_text = self._encode_input(text)[0] #默认只选择第一句？

		# get word activations
		hidden_word_encoding_out = Model(inputs=self.word_attention_model.input,
		                             	 outputs=self.word_attention_model.get_layer('dense_transform_w').output)
		hidden_word_encodings = hidden_word_encoding_out.predict(encoded_text) #这时候的模型是hidden_word_encoding_out，还不是“成品”,用这个模型预测相当于输出HNATT的中间结果
		word_context = self.word_attention_model.get_layer('word_attention').get_weights()[0]
		u_wattention = encoded_text*np.exp(np.squeeze(np.dot(hidden_word_encodings, word_context)))
		if websafe:
			u_wattention = u_wattention.astype(float)

		# generate word, activation pairs
		nopad_encoded_text = encoded_text[-len(normalized_text):]
		nopad_encoded_text = [list(filter(lambda x: x > 0, sentence)) for sentence in nopad_encoded_text]
		reconstructed_texts = [[self.reverse_word_index[int(i)] 
								for i in sentence] for sentence in nopad_encoded_text]
		nopad_wattention = u_wattention[-len(normalized_text):]
		nopad_wattention = nopad_wattention/np.expand_dims(np.sum(nopad_wattention, -1), -1)
		nopad_wattention = np.array([attention_seq[-len(sentence):] 
							for attention_seq, sentence in zip(nopad_wattention, nopad_encoded_text)])
		word_activation_maps = []
		for i, text in enumerate(reconstructed_texts):
			word_activation_maps.append(list(zip(text, nopad_wattention[i])))

		# get sentence activations
		hidden_sentence_encoding_out = Model(inputs=self.model.input,
											 outputs=self.model.get_layer('dense_transform_s').output)
		hidden_sentence_encodings = np.squeeze(
			hidden_sentence_encoding_out.predict(np.expand_dims(encoded_text, 0)), 0)
		sentence_context = self.model.get_layer('sentence_attention').get_weights()[0]
		u_sattention = np.exp(np.squeeze(np.dot(hidden_sentence_encodings, sentence_context), -1))
		if websafe:
			u_sattention = u_sattention.astype(float)
		nopad_sattention = u_sattention[-len(normalized_text):]

		nopad_sattention = nopad_sattention/np.expand_dims(np.sum(nopad_sattention, -1), -1)

		activation_map = list(zip(word_activation_maps, nopad_sattention))	

		return activation_map

if __name__ == '__main__':
	h = HNATT()
	train_x = np.array([['喜欢 在 京东 买 东西 ， 因为 今天 买 明天 就 可以 送到 。', '我 为什么 每个 商品 的 评价 都 一样 ， 而且 京东 购买 的 东西 品质 很 有 保证 ， 价格 也 很 合适 。']])
	train_y = np.array([[0,0,0,0,1]])
	h.train(train_x, train_y)
