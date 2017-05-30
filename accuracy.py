import pandas as pd
import argparse
import sys
from sklearn.metrics import roc_auc_score, accuracy_score

argument_parser = argparse.ArgumentParser(
    description='A tool to get the precision and roc-auc metrics')
argument_parser.add_argument('predictions',
                             type=argparse.FileType('r'),
                             default=sys.stdin,
                             nargs='?')
argument_parser.add_argument('true',
                             type=argparse.FileType('r'),
                             default=sys.stdout,
                             nargs='?')
args = argument_parser.parse_args()
df_true = pd.read_csv(args.true, header=None, names=['pol', 'id', 'date', 'lyx', 'user', 'txt'])
df_pred = pd.read_table(args.predictions, header=None, names=['pred', 'id'], sep=' ')
result_score = pd.merge(df_true, df_pred, on='id')
result_score.pol = result_score.pol == 4
print 'roc_auc', roc_auc_score(result_score.pol, result_score.pred)
print 'accuracy', accuracy_score(result_score.pol, result_score.pred > .5)
