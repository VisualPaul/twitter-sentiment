import numpy as np
import argparse
import pandas as pd
from keras import layers, models, metrics, optimizers, preprocessing, losses, regularizers
from keras.preprocessing import sequence
import re
from gensim.models import Word2Vec
from collections import Counter
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
import sys
from sklearn import metrics
import glove

argument_parser = argparse.ArgumentParser(
    description='NN model training and feed-forward tool')
subparsers = argument_parser.add_subparsers()

parser_train = subparsers.add_parser('train')
parser_train.add_argument('train',
                             type=argparse.FileType('r'))
parser_train.add_argument('--save', '-s')
parser_train.add_argument('--epochs', '-e', type=int, default=15)
parser_eval = subparsers.add_parser('eval')
parser_eval.add_argument('model')
parser_eval.set_defaults(train=None)

argument_parser.add_argument('test',
                             type=argparse.FileType('r'))
args = argument_parser.parse_args()

date_re = re.compile(r'(?P<week_day>\w{3}) (?P<month>\w{3}) (?P<day>\d\d) (?P<hour>\d\d):(?P<minute>\d\d):(?P<second>\d\d) (?P<tz>\w{3}) (?P<year>\d{4})')
split_re = re.compile(r'[^\w<>]')

def prepare_dataframe(file, shuffle=True):
    df = pd.read_csv(file.name, header=None, 
                       names=['pol', 'id', 'date', 'lyx', 'user', 'txt'])
    df.drop(df.index[df.pol == 2], inplace=True)
    df.pol = (df.pol == 4)
    #df[['week_day', 'day', 'hour']] = \
    #   df.date.str.extract(date_re, expand=True).loc[:,['week_day', 'day', 'hour']]
    #df = pf.get_dummies(df, columns=['week_day', 'day', 'hour'])
    df.txt = (df.txt
        .str.replace(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', r' <URL> ')
        .str.replace(r'@\w*', r' <USER> ')
        .str.replace(r'#([A-Z0-9\-]+)', r' <HASHTAG> \1 <ALLCAPS> ')
        .str.replace(r'#(\S+)', lambda x: ' <HASHTAG> ' + ' '.join(re.split(r'(?=[A-Z])', x.group(1))))
        .str.replace(r'<3', r' <HEART> ')
        .str.replace(r'([!?.]){2,}', r' \1 <REPEAT> ')
        .str.replace(r'\b(\S*?)(.)\2{2,}\b', r' \1\2 <ELONG> ')
        .str.replace(r'\s([^a-z0-9()<>\'`\-]){2,}\s',
                     lambda x: x.group(0).lower() + '<ALLCAPS> ')).str.lower()
    if shuffle:
        df = df.loc[np.random.permutation(df.index)]
    return df
emb_dim = 200

"""def build_word_map():
    words_dict = {}
    with open('glove.twitter.27B.200d.txt') as f:
        for line_num, line in enumerate(f):
            text = line.decode('utf-8').split()
            if len(text) != 201:
                continue
            words_dict[text[0]] = np.array([float(x) for x in text[1:]], dtype='float32')
    word_matrix = np.zeros((len(words_dict) + 2, emb_dim))
    word_num = {word: i + 1 for i, word in enumerate(words_dict)}
    for word, i in word_num.items():
        word_matrix[i] = words_dict[word]
    return word_matrix, word_num"""

print 'Loading glove'
word_matrix, word_num = glove.get_glove()
print 'Loaded'
maxlen = 60
def df_to_matrix(df):
    X = np.zeros((df.shape[0], maxlen), dtype='float32')
    for i, tweet in enumerate(df.txt):
        for j, word in enumerate(w for w in re.split(split_re, tweet) if w in word_num):
            if j >= maxlen:
                break
            X[i, j] = word_num[word]
    return X, df.pol

tweet = layers.Input((maxlen,), dtype='int32')
embedded = layers.Embedding(word_matrix.shape[0], 200, input_length=maxlen,
                            weights=[word_matrix], trainable=False)(tweet)                       
embedded_normalized = layers.BatchNormalization()(embedded)
lstm = layers.Bidirectional(layers.LSTM(150, dropout=.2, recurrent_dropout=.2))(embedded_normalized)
lstm_dropout = layers.Dropout(.5)(layers.BatchNormalization()(lstm))
result = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-3))(lstm_dropout)
model = models.Model(tweet, result)
model.compile(optimizer=optimizers.Adam(lr=1e-5), loss=losses.binary_crossentropy, metrics=['accuracy'])

if args.train:
    print 'processing data'
    X_train, y_train = df_to_matrix(prepare_dataframe(args.train))
    print 'data processed'
    model.fit(X_train, y_train, batch_size=256)
    if args.save is not None:
        model.save_weights(args.save)
else:
    model.load_weights(args.model)
print 'received a model'
X_test, y_test = df_to_matrix(prepare_dataframe(args.test))
y_pred = model.predict(X_test, batch_size=256)
print 'accuracy {}'.format(metrics.accuracy_score(y_test, y_pred > .5))
print 'ROC-AUC {}'.format(metrics.roc_auc_score(y_test, y_pred))
