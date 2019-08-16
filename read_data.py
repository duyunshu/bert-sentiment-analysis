import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
# from sklearn.model_selection import train_test_split
from collections import Counter
import csv
import tensorflow as tf
import os.path
# from os import listdir  
from tensorflow import keras
import os
import re
import spacy
import time
import math
import xml.etree.ElementTree as ET
import codecs
from collections import defaultdict


def pause():
    int(input("enter a num to cont..."))


def clean(path, dataset=None):
    filename = 'reviews_clean.json'
    with open(filename, 'w') as f:
        for line in open(path):
            if dataset == 'meta':
                line = line.replace('\\r', ' ')
            elif dataset == 'amazon':
                line = line.replace('\000','')
            f.write(line)
    return filename


def parse(path):    
    # g = gzip.open(path, 'rb')
    g = open(path, 'r')
    for l in g:
        # yield eval(l)
        yield json.loads(l)  # deal with null


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def get_child(child, df, nlp, f):
    if isinstance(child, float):
        f.write('\n')
        return 

    for c in child.split():
        if c in df.keys():
            line = str(df[c]['clean_text'])
            doc = nlp(line)
            for sent in doc.sents:
                f.write(str(sent).rstrip())
                f.write('\n')
            df[c]['visited'] = True

            return get_child(df[c]['children_ids'], 
                df, nlp, f)

    
def read_tsv(input_file):
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t")
      lines = []
      for line in reader:
        lines.append(line)
      return lines


def train_test_split(counts, pct_train=0.7, pct_dev=0.2, pct_test=0.1):
    total = sum([v for _, v in counts.items()])
    n_train = int(total * pct_train)
    n_dev = int(total * pct_dev)
    # n_test = int(total * pct_test)
    
    train = []
    dev = []
    test = []
    
    current_train = 0
    current_dev = 0
    
    for k, v in counts.items():
        if current_train + v <= n_train:
            train.append(k)
            current_train += v
        elif current_dev + v <= n_dev:
            dev.append(k)
            current_dev += v
        else:
            test.append(k)
    
    return train, dev, test


def over_sample(df, col_name):
    labels, values = zip(*Counter(df[col_name].values).items())  
    seed = int(np.max(values) / np.min(values))
    oversample = df.loc[df[col_name] == labels[values.index(np.min(values))]]
    for i in range(seed - 1):
        df = df.append(oversample)
    # shuffle
    df = df.sample(frac=1)
    return df


def read_meta(argv=None):
    review_clean = clean('/home/ydu/BERT/DATA/metacritic/reviews.json', 'meta')
    
    df = getDF(review_clean)
    
    df = df[['title','text','score']]
    df.score = df.score.astype(int)
    
    df['senti'] = -1
    df['senti'][df.score >= 7] = 1
    df['senti'][df.score <= 4] = 0
    df = df.loc[df['senti'] != -1]
    df = df.drop(columns=['score'])

    counts = dict(Counter(df.title.values))
    train_labels, dev_labels, test_labels = train_test_split(counts)
    
    # oversample in training
    train = df.loc[df['title'].isin(train_labels)]
    train = over_sample(train, 'senti')

    # oversample in dev
    dev = df.loc[df['title'].isin(dev_labels)]
    dev = over_sample(dev, 'senti')

    test = df.loc[df['title'].isin(test_labels)]

    train = train.drop(columns=['title'])
    dev = dev.drop(columns=['title'])
    test = test.drop(columns=['title'])
    df = df.drop(columns=['title'])


    train.to_csv('/home/ydu/BERT/DATA/metacritic/train.tsv', index=False, sep='\t')
    dev.to_csv('/home/ydu/BERT/DATA/metacritic/dev.tsv', index=False, sep='\t')
    test.to_csv('/home/ydu/BERT/DATA/metacritic/test.tsv', index=False, sep='\t')
    df.to_csv('/home/ydu/BERT/DATA/metacritic/all.tsv', index=False, sep='\t')


def read_reddit(argv=None):
    df = pd.read_csv('/home/ydu/BERT/DATA/reddit/posts_with_ids.csv')
    df = df.dropna(subset=['text'])
    df['clean_text'] = df['text'].apply(lambda x: ' '.join(x.split()))
    df['visited'] = False
    df.set_index('post_id', inplace=True)
    df = df.drop(columns=['text'])
    
    df = df.to_dict(orient='index')
    
    nlp = spacy.load('en_core_web_sm')
 
    start_time = time.time()

    f = open('pretrain_data/txt/pretrain_texttree.txt', 'w')

    for k, _ in df.items():
        if not df[k]['visited']:
            doc = nlp(str(df[k]['clean_text']))
            for sent in doc.sents:
                f.write(str(sent).rstrip())
                f.write('\n')
            df[k]['visited'] = True
            get_child(df[k]['children_ids'], df, nlp, f)

    f.close()

    print("--- %s sec ---" % (time.time() - start_time))


def read_amazon(argv=None):
    review_clean = clean('/home/ydu/BERT/DATA/amazon/aggressive_dedup_video_games.json', 'amazon')  
    df = getDF(review_clean)
    df = df[['asin','reviewText','overall']]
    df.overall = df.overall.astype(int)
    df['senti'] = -1
    df['senti'][df.overall <= 2] = 0
    df['senti'][df.overall >= 4] = 1
    df = df.loc[df['senti'] != -1]
    df = df.drop(columns=['overall'])

    df = df.rename(columns={"reviewText": "text"})
    
    counts = dict(Counter(df.asin.values))
    train_labels, dev_labels, test_labels = train_test_split(counts)
    
    # oversample in training
    train = df.loc[df['asin'].isin(train_labels)]
    train = over_sample(train, 'senti')
    # oversample in dev
    dev = df.loc[df['asin'].isin(dev_labels)]
    dev = over_sample(dev, 'senti')

    test = df.loc[df['asin'].isin(test_labels)]

    train = train.drop(columns=['asin'])
    dev = dev.drop(columns=['asin'])
    test = test.drop(columns=['asin'])
    df = df.drop(columns=['asin'])

    train.to_csv('/home/ydu/BERT/DATA/amazon/train.tsv', index=False, sep='\t')
    dev.to_csv('/home/ydu/BERT/DATA/amazon/dev.tsv', index=False, sep='\t')
    test.to_csv('/home/ydu/BERT/DATA/amazon/test.tsv', index=False, sep='\t')
    df.to_csv('/home/ydu/BERT/DATA/amazon/all.tsv', index=False, sep='\t')


# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
  data = {}
  data["text"] = []
  for file_path in os.listdir(directory):
    with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      data["text"].append(f.read())
  return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
  pos_df = load_directory_data(os.path.join(directory, "pos"))
  neg_df = load_directory_data(os.path.join(directory, "neg"))
  pos_df["senti"] = 1
  neg_df["senti"] = 0
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
  dataset = tf.keras.utils.get_file(
      fname="aclImdb.tar.gz", 
      origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
      extract=True)
  
  train_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                       "aclImdb", "train"))
  test_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                      "aclImdb", "test"))
  df = pd.concat([train_df,test_df]).sample(frac=1).reset_index(drop=True)
  
  return train_df, test_df, df


def read_imdb():    
    train, dev, df = download_and_load_datasets()
    train.to_csv('/home/ydu/BERT/DATA/imdb/train.tsv', index=False, sep='\t')
    dev.to_csv('/home/ydu/BERT/DATA/imdb/dev.tsv', index=False, sep='\t')
    df.to_csv('/home/ydu/BERT/DATA/imdb/all.tsv', index=False, sep='\t')


def read_amazon_xml(argv=None):
    filepath = '/home/ydu/BERT/bengio/sorted_data/'
    save_to = '/home/ydu/BERT/bengio/data_all/'

    f = open(save_to+'all_text/amazon_pretrain_text.txt', 'w')
    
    # get all review text for pre-training
    # num_l = 0
    # start_time = time.time()

    for folder in os.listdir(filepath):
        path = filepath + folder
        fn = path+'/all.review'
        print(fn)
        
        if os.path.exists(fn):
            with codecs.open(fn,'r+',encoding='utf-8', errors='ignore') as ff:
                test_data = ff.readlines()
                    
            test_data = [line.rstrip() for line in test_data]  # All lines including the blank ones
            test_data = [line for line in test_data if line]  # Non-blank lines

            count = 0
            i=0
            while i < len(test_data):
                # start_time = time.time()
                # num_l += 1
                line = test_data[i]
                j = i+1
                i+=1
                if line == '<review_text>':
                    nextline = test_data[j]
                    while nextline != '</review_text>':
                        f.write(nextline)
                        f.write('\n')
                        j+=1
                        nextline = test_data[j]
                        count+=1
                    f.write('\n')
                    i += count
                # if num_l % 10000 == 0:
                #     print("--- %s sec ---" % (time.time() - start_time))
                #     start_time = time.time()
    
    f.close()

    # raise SystemExit

    # get training data for classification using benchmark dataset
    category = ['books/','kitchen/', 'electronics/','dvd/']
    filename = ['negative.review','positive.review']
    senti = [0, 1]

    for c in category:
        path = filepath + c
        count = 0
        train = pd.DataFrame()
        # dev = pd.DataFrame()
        # test = pd.DataFrame()

        for i in range(len(filename)):
            fn = path + filename[i]
            print(fn)
            text = defaultdict(list)
            asin = []
            label = []
            
            with codecs.open(fn,'r+',encoding='utf-8', errors='ignore') as f:
                test_data = f.readlines()
                            
            while test_data:
                line = test_data.pop(0).strip()

                if line == '<asin>':
                    nextline = test_data.pop(0).strip()
                    while nextline != '</asin>':
                        asin.append(nextline)
                        nextline = test_data.pop(0).strip()
                if line == '<review_text>':
                    nextline = test_data.pop(0).strip()
                    while nextline != '</review_text>':
                        text[count].append(nextline)
                        nextline = test_data.pop(0).strip()
                    label.append(senti[i])
                    count+=1

            for k, _ in text.items():
                text[k] = ''.join(text[k])
            df = pd.DataFrame.from_dict(text, orient='index')
            df = df.rename(columns={0: "text"})
            df['asin'] = asin
            df['senti'] = label
            df = df[['asin','text','senti']]

            # counts = dict(Counter(df.asin.values))
            # train_labels, dev_labels, test_labels = train_test_split(counts)
            
            # train = pd.concat([train, df.loc[df['asin'].isin(train_labels)]], ignore_index=True)
            # dev = pd.concat([dev, df.loc[df['asin'].isin(dev_labels)]], ignore_index=True)
            # test = pd.concat([test, df.loc[df['asin'].isin(test_labels)]], ignore_index=True)

            train = pd.concat([train, df], ignore_index=True)

        if not os.path.exists(save_to+c):
            os.makedirs(save_to+c)
        
        train.to_csv(save_to+c+'train.tsv', index=False, sep='\t')
        
        # train.to_csv(save_to+c+'train.tsv', index=False, sep='\t')
        # dev.to_csv(save_to+c+'dev.tsv', index=False, sep='\t')
        # test.to_csv(save_to+c+'test.tsv', index=False, sep='\t')

                        
def read_all_pretrain():
    dataset = ['amazon/', 'metacritic/','imdb/']
    filename = ['train.tsv','dev.tsv','test.tsv']
    path = '/home/ydu/BERT/DATA/'
    text = []

    for d in dataset:
        folder = path + d
        for f in filename:
            fn = folder + f 
            if os.path.exists(fn):
                print(fn)
                df = pd.read_csv(fn, sep='\t')
                text.append(df[df.columns[0]].tolist())
    text = [t for sublist in text for t in sublist]
    
    nlp = spacy.load('en_core_web_sm')

    f = open('/home/ydu/BERT/DATA/all4data/all4data.txt', 'w')

    count=0
    start_time = time.time()
    for t in text:
        count+=1
        doc = nlp(str(t))
        for sent in doc.sents:
            f.write(str(sent).rstrip())
            f.write('\n')
        f.write('\n')
        if count%10000==0:
            print("--- %s sec ---" % (time.time() - start_time))
            start_time = time.time()

    f.close()

    # reddit data has tree structure,  
    # append from pretrain_data/txt/pretrain_texttree.txt
    f = open('/home/ydu/BERT/DATA/all4data/all4data.txt', 'a')
    ff = open('/home/ydu/BERT/bert_mgpu/pretrain_data/txt/pretrain_texttree.txt', 'r')

    for line in ff:
        f.write(line)

    f.close()
    ff.close()


def read_ami_train():
    dataset = ['amazon/', 'metacritic/','imdb/']
    filename = ['train.tsv','dev.tsv','test.tsv']
    path = '/home/ydu/BERT/DATA/'
    
    for f in filename:
        df = pd.DataFrame()
        for d in dataset:
            fn = path + d + f 
            if os.path.exists(fn):
                print(fn)
                df = pd.concat([df, pd.read_csv(fn, sep='\t')]).sample(frac=1).reset_index(drop=True)
            df.to_csv(path+'all4data/'+f, index=False, sep='\t')


def split_txt():
    lines_per_file = 3000000
    smallfile = None
    
    count=0
    start_time = time.time()
    
    with open('/home/ydu/BERT/DATA/all4data/all4data.txt') as bigfile:
        for lineno, line in enumerate(bigfile):
            if lineno % lines_per_file == 0:
                if smallfile:
                    smallfile.close()
                small_filename = '/home/ydu/BERT/DATA/all4data/all4data_{}.txt'.format(lineno + lines_per_file)
                smallfile = open(small_filename, "w")
            smallfile.write(line)
            
            count+=1
            if count % 1000000 == 0:
                print("--- %s sec ---" % (time.time() - start_time))
                start_time = time.time()
        
        if smallfile:
            smallfile.close()


if __name__ == '__main__':
    read_meta()
    # read_amazon()
    # read_imdb()
    # read_reddit()
    
    # read_all_pretrain()  # read text from all 4 dataset as pre-train (no senti label)
    # split_txt()  # naive split all4data.txt into small chunks

    # read_ami_train()

    # read_amazon_xml()  # bengio's experiments
