import numpy as np
import os.path

def build_glove():
	words_dict = {}
	with open('glove.twitter.27B.200d.txt') as f:
	    for line_num, line in enumerate(f):
	        text = line.decode('utf-8').split()
	        if len(text) != 201:
	            continue
	        words_dict[text[0]] = np.array([float(x) for x in text[1:]], dtype='float32')
	word_matrix = np.zeros((len(words_dict) + 2, 200))
	word_num = {word: i + 1 for i, word in enumerate(words_dict)}
	for word, i in word_num.items():
	    word_matrix[i] = words_dict[word]
	np.savez('glove.npz', word_matrix=word_matrix, word_num=word_num)

def get_glove():
	if not os.path.exists('glove.npz'):
		build_glove()
	file = np.load('glove.npz')
	return file['word_matrix'], file['word_num']

if __name__ == '__main__':
	build_glove()