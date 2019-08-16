import csv
import tensorflow as tf
import pandas as pd
import tokenization

TEST_TSV = '/home/ydu/BERT/DATA/reddit/test.tsv'
RESULTS_TSV = '/home/ydu/BERT/bert_mgpu/predict/081419/no-pretrain-imdbtune/test_results.tsv'
DIR = '/home/ydu/BERT/bert_mgpu/predict/081419/no-pretrain-imdbtune/'

def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines

lines = _read_tsv(TEST_TSV)
lines = [tokenization.convert_to_unicode(line[0]) for line in lines]

results = open(RESULTS_TSV).readlines()
results = [result.strip().split('\t') for result in results]

df = pd.DataFrame()
df['text'] = lines
df['0'] = [result[0] for result in results]
df['1'] = [result[1] for result in results]

df.to_csv(DIR+'imdbtune.tsv', index=False, sep='\t')