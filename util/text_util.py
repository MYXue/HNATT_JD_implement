import string
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
import jieba
import re

STOP_WORDS = [' '] #停用词
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))

def normalize(text): #分句，每句变成一个字符串,句内单词之间用空格连接,删除句中的标点符号
	text = text.lower().strip()
	# print ("text:",text)
	doc = nlp(text)
	# print ("doc:",doc)
	filtered_sentences = []
	for sentence in doc.sents:
		# print ("sentence:",sentence)
		filtered_tokens = list()
		for i, w in enumerate(sentence):
			# print (i,w)
			s = w.string.strip()
			if len(s) == 0 or s in string.punctuation and i < len(doc) - 1: #过滤掉了标点符号
				# print ("be passed:",s)
				continue
			if s not in STOP_WORDS:
				s = s.replace(',', '.')
				filtered_tokens.append(s)
		filtered_sentences.append(' '.join(filtered_tokens))
	return filtered_sentences


def normalize_ch(text): #中文分句分词，每句变成一个字符串,句内单词之间用空格连接
	#分句
	# print (text)
	sentences = re.split('(。|！|\!|\.|？|\?)',text)         # 保留分割符 
	new_sents = []
	for i in range(int(len(sentences)/2)):
	    sent = sentences[2*i] + sentences[2*i+1]
	    new_sents.append(sent)
	# print (new_sents)

	filtered_sentences = []
	for sentence in new_sents:
		# print ("sentence:",sentence)
		filtered_tokens = list()
		for word in jieba.cut(sentence):
			# print ("word:",word)
			if len(word) == 0:
				# print ("be passed:",s)
				continue
			if word not in STOP_WORDS:
				filtered_tokens.append(word)
		filtered_sentences.append(' '.join(filtered_tokens))

	return filtered_sentences

if __name__ == '__main__':
	text_en = "hello world, I like this book! welcome to this store, and hope you having fun"
	print (normalize(text_en))

	text_ch = "喜欢在京东买东西，因为今天买明天就可以送到。我为什么每个商品的评价都一样，而且京东购买的东西品质很有保证，价格也很合适。   "
	print (normalize_ch(text_ch))