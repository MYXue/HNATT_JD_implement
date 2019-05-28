import pandas as pd
import numpy as np
from tqdm import tqdm #一个快速、可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)

from util.text_util import normalize_ch

tqdm.pandas()

def chunk_to_arrays(chunk, binary=False): #从dataframe中抽取X和Y,返回两个numpy.ndarray
	x = chunk['text_tokens'].values 
	if binary:
		y = chunk['polarized_stars'].values
	else:
		y = chunk['stars'].values

	# print ("len(x):",len(x))
	# print ("len(y):",len(y))

	# i = 0
	# for i in range(10):
	# 	print (x[i],y[i])
	return x, y

def balance_classes(x, y, dim, train_ratio): #如果是二分类，则保证训练集和测试集有相同数目的样本；如果两类样本数量差别太大，这样会导致大量样本在训练中被浪费
	x_negative = x[np.where(y == 1)]
	y_negative = y[np.where(y == 1)]
	x_positive = x[np.where(y == 2)]
	y_positive = y[np.where(y == 2)]

	n = min(len(x_negative), len(x_positive)) #两类中较小的数量
	train_n = int(round(train_ratio * n)) 
	train_x = np.concatenate((x_negative[:train_n], x_positive[:train_n]), axis=0)
	train_y = np.concatenate((y_negative[:train_n], y_positive[:train_n]), axis=0)
	test_x = np.concatenate((x_negative[train_n:], x_positive[train_n:]), axis=0)
	test_y = np.concatenate((y_negative[train_n:], y_positive[train_n:]), axis=0)

	# import pdb; pdb.set_trace()
	print ("Binary Data Split_____")
	print ("number of train:",len(train_x))
	print ("number of test:",len(test_y))
	return (train_x, to_one_hot(train_y, dim=2)), (test_x, to_one_hot(test_y, dim=2))

def to_one_hot(labels, dim=5): #对传进来的向量做one-hot编码
	results = np.zeros((len(labels), dim)) #这是一个二维全0矩阵
	for i, label in enumerate(labels):
		results[i][int(label) - 1] = 1
	return results

def polarize(v): #如果想把1-5分打分的情况变成2分类问题的话，调用此函数可将3分以下和3分以上(含3分的)变成两类
	if v >= 3:
		return 2
	else:
		return 1

def load_data(path, size=1e4, train_ratio=0.8, binary=False):
	# print('loading JD reviews...')
	columns = ['text','stars']
	df = pd.read_table(path, nrows=size, header=None,sep='\t',names=columns) # 读取指定行数的数据(这里是10000条)，训练集+测试集一共的数量
	# print (df['stars'])
	df.dropna(axis=0, how='any', inplace=True) #删除有空值的行
	# print ("df from jdData:",df)
	df['text_tokens'] = df['text'].progress_apply(lambda x: normalize_ch(x)) #progress_apply显示进度条；df['text_tokens']中每一行中的元素数代表句子数，每一句是一个完整字符串单词之间空格连接句中删除了标点
	
	dim = 5
	if binary: #如果没有指定2分类就是多分类，分类数(标签数由dim决定)
		dim = 2

	if binary:
		df['polarized_stars'] = df['stars'].apply(lambda x: polarize(x)) #若考虑两分类问题，则把评星数由1-5变成1-2
		x, y = chunk_to_arrays(df, binary=binary) #输入数据框，输出x序列和y序列
		return balance_classes(x, y, dim, train_ratio)

	train_size = round(size * train_ratio)  #在调用函数时指定加载数据集的数量size，和训练集占比train_ratio
	test_size = size - train_size;

	# training + validation set 
	train_x = np.empty((0,)) #效果上类似于[]？注意一下这种写法
	train_y = np.empty((0,))

	train_set = df[0:train_size].copy() #直接选择前train_size条作为训练集
	train_set['len'] = train_set['text_tokens'].apply(lambda x: len(x)) #len是每个样本中的句子数
	# train_set.sort_values('len', inplace=True, ascending=True)
	train_x, train_y = chunk_to_arrays(train_set, binary=binary)
	train_y = to_one_hot(train_y, dim=dim)

	test_set = df[train_size:]
	test_x, test_y = chunk_to_arrays(test_set, binary=binary)
	test_y = to_one_hot(test_y)
	print('finished loading JD reviews')

	print ("Data Split_____")
	print ("number of train:",len(train_x))
	print ("number of test:",len(test_y))

	return (train_x, train_y), (test_x, test_y)

if __name__ == '__main__':
	JD_DATA_PATH = 'D:/HNATT_JD_implement/data_jd/extract_comments1.txt'
	load_data(path=JD_DATA_PATH, size=100, binary=False)
