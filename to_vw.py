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
argument_parser.add_argument('--write-immediately', '-i', action='store_true')
argument_parser.add_argument('--progress', '-p', action='store_true')

args = argument_parser.parse_args()
user_tweets = Counter()
user_positive = Counter()
df = Counter()
date_re = re.compile(r'(?P<week_day>\w{3}) (?P<month>\w{3}) (?P<day>\d\d) (?P<hour>\d\d):(?P<minute>\d\d):(?P<second>\d\d) (?P<tz>\w{3}) (?P<year>\d{4})')
replace_pattern = re.compile(r'@\w*|[0-9]+|[a-zA-Z0-9\.\+_]+@[a-zA-Z0-9\._]+|(https?://)?[\w\d]+\.[\w\d\.]+/\S+|[:;][\-\=]*D|[xX]+D|[^a-zA-Z0-9\s@]')
tweets = 0

def csv_rows(f, row_num=None):
    iters = 0
    start_time = time.time()
    nsecs = 1
    f.seek(0)
    for pol, id, date, lyx, user, tweet in csv.reader(f):
        if pol == '2':
            continue
        yield pol, id, date, lyx, user, tweet
        iters += 1
        if iters % 1000 == 0 and args.progress:
            current_time = time.time()
            if current_time < start_time + nsecs:
                continue
            nsecs += 1
            if row_num is None:
                print '\rProcessed: {:8,d} in {:4d} seconds'.format(iters, int(round(current_time - start_time))),
            else:
                eta = (current_time - start_time) / float(iters) * row_num
                print '\rProcessed: {:8,d} of {:8,d} in {:4d} seconds (ETA: {})'.format(
                    iters, row_num, int(round(current_time - start_time)), int(round(eta - current_time + start_time))),
            sys.stdout.flush()
    if nsecs != 1:
        print

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

def process_file(input_file, line_visitor, train, row_num=None):
    for pol, id, date, lyx, user, tweet in csv_rows(input_file, row_num=row_num):
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
        line_visitor.visit('{} {}|a '.format(lbl, id) + ' '.join(features))

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
