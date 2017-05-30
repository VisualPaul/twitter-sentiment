import argparse
import csv
import sys
import numpy as np
import datetime
import re
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
argument_parser.add_argument('--progress', '-p', action='store_true')
args = argument_parser.parse_args()
user_tweets = Counter()
user_positive = Counter()
df = Counter()
date_re = re.compile(r'(?P<week_day>\w{3}) (?P<month>\w{3}) (?P<day>\d\d) (?P<hour>\d\d):(?P<minute>\d\d):(?P<second>\d\d) (?P<tz>\w{3}) (?P<year>\d{4})')
replace_pattern = re.compile(r'@\w*|[0-9]+|[a-zA-Z0-9\.\+_]+@[a-zA-Z0-9\._]+|(https?://)?[\w\d]+\.[\w\d\.]+/\S+|[:;][\-\=]*D|[xX]+D|[^a-zA-Z0-9\s@]')
tweets = 0

def csv_rows(f):
    iters = 0
    next_print = 1000
    f.seek(0)
    for pol, id, date, lyx, user, tweet in csv.reader(f):
        if pol == '2':
            continue
        yield pol, id, date, lyx, user, tweet
        iters += 1
        if iters == next_print and args.progress:
            next_print *= 2
            print iters, 'processed'

if args.progress:
    print 'Train pass [1/2]:'

for pol, id, date, lyx, user, tweet in csv_rows(args.csv_file):
    tweets += 1
    user_tweets[user] += 1
    if pol == '4':
        user_positive[user] += 1
    for word in set(re.sub(replace_pattern, ' ', tweet).lower().split()):
        if word in stops:
            continue
        df[word] += 1
        

ADDED_TWEETS = 1

def process_file(input_file, output_file, train):
    for pol, id, date, lyx, user, tweet in csv_rows(input_file):
        features = []
        lbl = 1 if pol == '4' else -1
        pos = 1 if pol == '4' else 0
        features.append('.user_tweets:' + str(user_tweets[user]))
        if train:
            counter = ((user_positive[user] - pos + .5 * ADDED_TWEETS)
                       / (user_tweets[user] - 1 + ADDED_TWEETS))
        else:
            counter = ((user_positive[user] + .5 * ADDED_TWEETS)
                       / (user_tweets[user] + ADDED_TWEETS))
#        features.append('.user_counter:' + str(counter))
        date_match = re.match(date_re, date)
        assert date_match is not None
        features.append('.week_day_' + date_match.group('day'))
        features.append('.hour_' + date_match.group('hour'))
        tf = Counter(re.sub(replace_pattern, ' ', tweet).lower().split())
        for word in tf:
            if word in stops:
                continue
            tfidf = tf[word] * np.log(float(tweets) / (df[word] + 1))
            features.append(word + ':' + str(tfidf))
        print >> output_file, '{} {}|a'.format(lbl, id), ' '.join(features)
if args.progress:
    print 'Train pass [2/2]:'
process_file(args.csv_file, args.vw_file, train=True)
if args.progress:
    print 'Test pass'
process_file(args.test_csv, args.test_vw, train=False)
