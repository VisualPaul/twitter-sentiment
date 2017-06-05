import argparse
import csv
import sys
import numpy as np
from gensim.models import Word2Vec
from multiprocessing import cpu_count
import random
from datetime import datetime
import re
import time
from collections import Counter
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

stops = ENGLISH_STOP_WORDS

argument_parser = argparse.ArgumentParser(
    description='A small tool that converts csv to vw format')

argument_parser.add_argument('csv_file',
                             type=argparse.FileType('r'),
                             default=sys.stdin,
                             nargs='?')
argument_parser.add_argument('vw_file',
                             type=argparse.FileType('w'),
                             default=sys.stdout,
                             nargs='?')
argument_parser.add_argument('test_csv',
                             type=argparse.FileType('r'),
                             default=sys.stdin,
                             nargs='?')
argument_parser.add_argument('test_vw',
                             type=argparse.FileType('w'),
                             default=sys.stdout,
                             nargs='?')
argument_parser.add_argument('--vectorizer', '-v' '--vectoriser',
                             choices=['tfidf', 'glove', 'w2v'],
                             default='tfidf')
argument_parser.add_argument('--write-immediately', '-i', action='store_true')
argument_parser.add_argument('--progress', '-p', action='store_true')

args = argument_parser.parse_args()
user_tweets = Counter()
user_positive = Counter()
date_re = re.compile(
    r'(?P<week_day>\w{3}) (?P<month>\w{3}) (?P<day>\d\d) (?P<hour>\d\d):(?P<minute>\d\d):(?P<second>\d\d) (?P<tz>\w{3}) (?P<year>\d{4})')
replace_pattern = re.compile(r'@\w*|[0-9]+|[a-zA-Z0-9\.\+_]+@[a-zA-Z0-9\._]+|(https?://)?[\w\d]+\.[\w\d\.]+/\S+|[:;][\-\=]*D|[xX]+D|[^a-zA-Z0-9\s@]')

if args.progress:
    def report_progress(iterable, total=None, period=1000):
        start_time = time.time()
        nsecs = 1
        i = -1
        for i, result in enumerate(iterable):
            if i % period == 0:
                current_time = time.time()
                if current_time >= start_time + nsecs:
                    nsecs += 1
                    if total is None:
                        print '\rProcessed: {:8,d} in {:4d} seconds'.format(i, int(round(current_time - start_time))),
                    else:
                        elapsed = current_time - start_time
                        eta = elapsed / float(max(1, i)) * total
                        print '\rProcessed: {:8,d} of {:8,d} in {:4d} seconds (ETA: {})'.format(
                            i, total, int(round(elapsed)), int(round(eta - elapsed))),
                    sys.stdout.flush()
            yield result
        if nsecs != 1:
            print
        print 'Completed {:8,d} in {:.2f} seconds'.format(i + 1, time.time() - start_time)
else:
    def report_progress(iterable, *args, **kwargs):
        return iterable

class Vectorizer(object):
    def __init__(self, tweets):
        pass
    def get_features(self, tweet):
        raise ValueError('Not implemented!')

class TfIdfVectorizer(Vectorizer):
    def __init__(self, tweets):
        self.df = Counter()
        self.tweets = 0
        for tweet in tweets:
            for word in set(tweet):
                if word not in stops:
                    self.df[word] += 1
            self.tweets += 1
        self.tweets_f = float(self.tweets)

    def get_features(self, tweet):
        tf = Counter(tweet)
        return ['{}:{}'.format(word, tf[word] * np.log(self.tweets_f / (self.df[word] + 1)))
                for word in tf
                if (word not in stops and self.df[word] < 4 )] # I am (still) a Lisp programmer


class GloveVectorizer(Vectorizer):
    def __init__(self, tweets):
        self.words_dict = {}
        if args.progress:
            print 'Loading GloVe'
        with open('glove.twitter.27B.200d.txt') as f:
            for line in report_progress(f):
                text = line.decode('utf-8').split()
                if len(text) != 201:
                    continue
                self.words_dict[text[0]] = np.array([float(x) for x in text[1:]],
                                                     dtype='float32')
    def get_features(self, tweet):
        vectors = [self.words_dict[word] for word in tweet if word in self.words_dict]
        vectors.append(np.zeros(200))
        vector = np.mean(vectors, axis=0)
        return ['_{}:{}'.format(x, vector[x]) for x in xrange(200)]

class Word2VecVectorizer(Vectorizer):
    def __init__(self, tweets):
        self.w2v = Word2Vec(iter(tweets), size=100, workers=cpu_count())
    def get_features(self, tweet):
        vectors = [self.w2v.wv[word] for word in tweet if word in self.w2v.wv] + [np.zeros(100)]
        vector = np.mean(vectors, axis=0)
        return ['_{}:{}'.format(x, vector[x]) for x in xrange(100)]

def replace_function(*args):
    if len(args) % 2 != 0:
        raise ValueError('args must be in pairs')
    patterns = [re.compile(pat) for pat in args[0::2]]
    replacements = args[1::2]
    everything = zip(patterns, replacements)
    def fun(text):
        for pat, rep in everything:
            text = re.sub(pat, rep, text)
        return text
    return fun

tweet_replace = replace_function(
    r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', r' <URL> ',
    r'@\w*', r' <USER> ',
    r'#([A-Z0-9\-]+)', r' <HASHTAG> \1 <ALLCAPS> ',
    r'#(\S+)', lambda x: ' <HASHTAG> ' + ' '.join(re.split(r'(?=[A-Z])', x.group(1))),
    r'<3', r'<HEART>',
    r'([!?.]){2,}', r' \1 <REPEAT> ',
    r'\b(\S*?)(.)\2{2,}\b', r' \1\2 <ELONG> ',
    r'\s([^a-z0-9()<>\'`\-]){2,}\s', lambda x: x.group(0).lower() + '<ALLCAPS> ')

def tweet_to_words(tweet):
    return [x for x in re.split(r'[^\w<>]', tweet_replace(tweet).lower()) if x != '']

def generator_to_iterable(gen):
    class Iterator(object):
        def __init__(self, gen):
            self.gen = gen
        def __iter__(self):
            return self
        def next(self):
            return next(self.gen)
    class Iterable(object):
        def __iter__(self):
            return Iterator(gen())
    return Iterable()

class TweetsCsv(object):
    def __init__(self, file):
        self.file = file
        self.n_tweets = None

    def columns(self, progress_period=1000, pass_name=None):
        def gen():
            if args.progress:
                if pass_name is None:
                    print 'Passing over {} file'.format(self.file.name)
                else:
                    print 'Passing over {} file ({})'.format(self.file.name, pass_name)
            tweets = 0
            self.file.seek(0)
            for pol, id, date, lyx, user, tweet in report_progress(csv.reader(self.file), total=self.n_tweets):
                tweets += 1
                if pol == '2':
                    continue
                pol = pol == '4'
                date_match = re.match(date_re, date)
                assert date_match is not None
                date_string = '{} {} {} {}:{}:{}'.format(
                    date_match.group('year'),
                    date_match.group('month'),
                    date_match.group('day'),
                    date_match.group('hour'),
                    date_match.group('minute'),
                    date_match.group('second'))
                date = datetime.strptime(date_string, '%Y %b %d %H:%M:%S')
                tweet = tweet_to_words(tweet)
                yield pol, id, date, lyx, user, tweet
            self.n_tweets = tweets
        return generator_to_iterable(gen)

    def tweets(self, progress_period=1000, pass_name=None):
        def gen():
            for pol, id, date, lyx, user, tweet in self.columns(progress_period, pass_name):
                yield tweet
        return generator_to_iterable(gen)

    def vw_lines(self, user_tweets, user_positive, vectorizer, progress_period=1000, pass_name=None):
        def gen():    
            for pol, id, date, _, user, tweet in self.columns(progress_period=progress_period, pass_name=pass_name):
                features = ['.user_tweets:' + str(user_tweets[user]),
                            '.week_day_' + str(date.isoweekday()),
                            '.hour_' + str(date.hour)] + vectorizer.get_features(tweet)
                lbl = 1 if pol else -1

                yield '{lbl} {tag}|x {features}'.format(
                    lbl=(1 if pol else -1), tag=id,
                    features=(' '.join(features)))
        return generator_to_iterable(gen)   
    def get_tweets_number(self):
        if self.n_tweets is None:
            for x in yield_everything(pass_name='counting number of lines'):
                pass
        return self.n_tweets

train = TweetsCsv(args.csv_file)
test = TweetsCsv(args.test_csv)

for pol, id, date, lyx, user, tweet in train.columns(pass_name='getting per user info'):
    user_tweets[user] += 1
    if pol:
        user_positive[user] += 1

if args.vectorizer == 'tfidf':
    vectorizer_class = TfIdfVectorizer
elif args.vectorizer == 'glove':
    vectorizer_class = GloveVectorizer
elif args.vectorizer == 'w2v':
    vectorizer_class = Word2VecVectorizer
else:
    raise ValueError('incorrect vectorizer argument: ' + repr(args.vectorizer))


vec = vectorizer_class(train.tweets(pass_name='training vectorization'))

def print_lines_to_file(lines, file):
    for line in lines:
        print >> file, line

if args.write_immediately:
    print_lines_to_file(train.vw_lines(user_tweets, user_positive, vec, pass_name='train conversion'), args.vw_file)
else:
    lines = list(train.vw_lines(user_tweets, user_positive, vec, pass_name='train conversion'))
    random.shuffle(lines)
    print_lines_to_file(lines, args.vw_file)
    del lines

print_lines_to_file(test.vw_lines(user_tweets, user_positive, vec, pass_name='train conversion'), args.test_vw)
