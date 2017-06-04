import argparse
import csv
import sys
import numpy as np
import random
import datetime
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
                             choices=['tfidf', 'glove'],
                             default='tfidf')
argument_parser.add_argument('--write-immediately', '-i', action='store_true')
argument_parser.add_argument('--progress', '-p', action='store_true')

args = argument_parser.parse_args()
user_tweets = Counter()
user_positive = Counter()
df = Counter()
date_re = re.compile(r'(?P<week_day>\w{3}) (?P<month>\w{3}) (?P<day>\d\d) (?P<hour>\d\d):(?P<minute>\d\d):(?P<second>\d\d) (?P<tz>\w{3}) (?P<year>\d{4})')
replace_pattern = re.compile(r'@\w*|[0-9]+|[a-zA-Z0-9\.\+_]+@[a-zA-Z0-9\._]+|(https?://)?[\w\d]+\.[\w\d\.]+/\S+|[:;][\-\=]*D|[xX]+D|[^a-zA-Z0-9\s@]')
tweets = 0
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
            yield result
        if nsecs != 1:
            print
        print 'Completed {:8,d} in {:.2f} seconds'.format(i + 1, time.time() - start_time)
else:
    def report_progress(iterable, *args, **kwargs):
        return iterable

class Vectorizer(object):
    def __init__(self):
        pass
    def fit_tweet(self, tweet):
        pass
    def finalize_fit(self):
        pass
    def vectorize(self, tweet):
        raise ValueError('Not implemented!')

class TfIdfVectorizer(Vectorizer):
    def __init__(self):
        self.df = Counter()
        self.tweets = 0
    def fit_tweet(self, tweet):
        for word in set(tweet):
            if word not in stops:
                self.df[word] += 1
        self.tweets += 1
    def finalize_fit(self):
        self.tweets_f = float(self.tweets)
    def vectorize(self, tweet):
        tf = Counter(tweet)
        return {word: tf[word] * np.log(self.tweets_f / (df[word] + 1))
                for word in tf
                if (word not in stops and df[word] < 4 )} # I am (still) a Lisp programmer

class GloveVectorizer(Vectorizer):
    def finalize_fit(self):
        self.words_dict = {}
        with open('glove.twitter.27B.200d.txt') as f:
            for line in report_progress(f):
                text = line.decode('utf-8').split()
                if len(text) != 201:
                    continue
                self.words_dict[text[0]] = np.array([float(x) for x in text[1:]],
                                                     dtype='float32')
    def vectorize(self, tweet):
        vector = np.mean([self.words_dict[word] for word in tweet])
        return {'_' + x: vector[x] for x in xrange(200)}


if args.vectorizer == 'tfidf':
    vec = TfIdfVectorizer()
elif args.vectorizer == 'glove':
    vec = GloveVectorizer()
else:
    raise ValueError('incorrect vectorizer argument: ' + repr(args.vectorizer))

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

def csv_rows(f, row_num=None):
    f.seek(0)
    for pol, id, date, lyx, user, tweet in report_progress(csv.reader(f), total=row_num):
        if pol == '2':
            continue
        yield pol, id, date, lyx, user, tweet

if args.progress:
    print 'Train pass [1/2]:'

for pol, id, date, lyx, user, tweet in csv_rows(args.csv_file):
    tweets += 1
    user_tweets[user] += 1
    if pol == '4':
        user_positive[user] += 1
    vec.fit_tweet(tweet_to_words(tweet))
        

ADDED_TWEETS = 1
vec.finalize_fit()

def process_file(input_file, line_visitor, train, row_num=None):
    for pol, id, date, lyx, user, tweet in csv_rows(input_file, row_num=row_num):
        features = []
        lbl = 1 if pol == '4' else -1
        pos = 1 if pol == '4' else 0
        features.append('.user_tweets:' + str(user_tweets[user]))
        date_match = re.match(date_re, date)
        assert date_match is not None
        features.append('.week_day_' + date_match.group('day'))
        features.append('.hour_' + date_match.group('hour'))
        tf = Counter(tweet_to_words(tweet))
        features = vec.vectorize(tweet_to_words(tweet))
        line_visitor.visit('{} {}|a '.format(lbl, id) + 
            ' '.join([str(x) + ':' + str(features[x]) for x in features]))

class PrintLineVisitor(object):
    def __init__(self, file):
        self.file = file

    def visit(self, line):
        print >> self.file, line

class StoreLineVisitor(object):
    def __init__(self):
        self.list = []
    def visit(self, line):
        self.list.append(line)

if args.progress:
    print 'Train pass [2/2]:'
if args.write_immediately:
    visitor = PrintLineVisitor(args.vw_file)
else:
    visitor = StoreLineVisitor()
process_file(args.csv_file, visitor, train=True, row_num=tweets)
if not args.write_immediately:
    random.shuffle(visitor.list)
    for line in visitor.list:
        print >> args.vw_file, line
del visitor
if args.progress:
    print 'Test pass'
process_file(args.test_csv, PrintLineVisitor(args.test_vw), train=False)
