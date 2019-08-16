import tensorflow as tf
import csv
from bert_serving.client import BertClient
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer
import umap
from collections import defaultdict
import pickle
import time
from warnings import filterwarnings
filterwarnings('ignore')
import pandas as pd
import glob, os



def pause():
    int(input("enter a num to cont..."))


def read_tsv(input_file):
    lines = []
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t")      
      for line in reader:
        text = line[0]      
        new = str(" ".join(text.split()))
        new = new.rstrip()       
        if new != "" or new:
            lines.append(new)
        # lines.append(text)
    return lines[1:]


def train_svm(X, Y, key, results):    
    c = np.logspace(0.00001, 1.0, num=10)
    
    # Randomly sample 1000 data points from each
    # repeat for 100 times
    for i in range(100):
        np.random.seed(int(round(time.time() * 1000)) % (2**32 - 1))
        index = np.random.randint(0, len(X), 1000)

        x = [X[j] for j in index]
        y = [Y[j] for j in index]
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
                
        for k in range(len(c)):
            clf = LinearSVC(random_state=42, C=c[k],max_iter=100)
            clf.fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
            err_rate = 1-acc
            a_dist = 2 * (1 - 2 * err_rate)
            results[key].append(a_dist) 
    
    return results


def save_emb():

    common = [
        '-model_dir', '/home/ydu/BERT/uncased_L-12_H-768_A-12/',
        '-num_worker', '2',
        '-port', '5555',
        '-port_out', '5556',
        '-max_seq_len', '128',
        '-max_batch_size', '256',
        # '-tuned_model_dir', '/home/ydu/BERT/bert_mgpu/pretrain_output/10k-32b-all4data/',
        # '-ckpt_name', 'model.ckpt-2500',
    ]

    args = get_args_parser().parse_args(common)

    # folder = ['books', 'dvd', 'electronics', 'kitchen']
    data_path = '/home/ydu/BERT/DATA/'
    data_folder = ['metacritic', 'imdb', 'amazon', 'reddit']
    
    # model_path = 'home/ydu/BERT/bert_mgpu/results/'
    # model_folder = 'amazon-balanced/'
    # model_type = 'bert-tune'
    data = {}

    # setattr(args, 'tuned_model_dir', '/home/ydu/BERT/bert_mgpu/pretrain_output/reddit-pretrain')
    # setattr(args, 'ckpt_name', 'model.ckpt-2500')
    setattr(args, 'tuned_model_dir', '/home/ydu/BERT/bert_mgpu/pretrain_output/10k-32b-all4data')
    setattr(args, 'ckpt_name', 'model.ckpt-2500')
   
    for d in data_folder:
        fn = data_path + d + '/all.tsv'
        print("===========",fn,"================")
        text = read_tsv(fn)       
        server = BertServer(args)
        server.start()
        print('wait until server is ready...')
        time.sleep(20)
        print('encoding...')
        bc = BertClient()
        data[d] = bc.encode(text)
        bc.close()
        server.close()

    pickle_name = data_path+'EMB/allpre_emb.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return pickle_name


def load_emb(pickle_name):    
    with open(pickle_name, 'rb') as handle:
        emb = pickle.load(handle)

    data = ['metacritic', 'imdb', 'amazon', 'reddit']
    # model = ['meta', 'amazon','imdb', 'bert']
    results = defaultdict(list)

    for i in range(len(data)):
        emb1 = emb[data[i]]
        for j in range(len(data)):
            if j == i:
                continue
            emb2 = emb[data[j]]
            
            key = str(data[i]) + '_' + str(data[j])
            print(key)
            
            # dummy label
            label1 = np.array([0 for _ in emb1])
            label2 = np.array([1 for _ in emb2])

            X = np.concatenate((emb1, emb2), axis=0)
            Y = np.concatenate((label1, label2), axis=0)

            # Randomly shuffle data
            np.random.seed(int(round(time.time() * 1000)) % (2**32 - 1))
            shuffle_indices = np.random.permutation(np.arange(len(Y)))
            x_shuffled = X[shuffle_indices]
            y_shuffled = Y[shuffle_indices]

            results = train_svm(x_shuffled, y_shuffled, key, results)           
    
    result2 = defaultdict(list)
    
    for k, i in results.items():
        avg = np.mean(i)
        std = np.std(i)
        result2[k] = [avg, std]
        df = pd.DataFrame.from_dict(result2, orient='index', columns=['A-distance', 'Stdev'])
    print(df)
    tsv_name = pickle_name.replace('_emb.pickle', '_adist.tsv')
    print(tsv_name)
    df.to_csv(tsv_name, index=True, sep='\t')

    # print("model: {}\ndata_pair: {}, avg: {}, stdev: {}".format(str(pickle_name), str(k), np.mean(i), np.std(i)))


def load_baseline_emb(pickle_list):
    emb_dict = defaultdict(dict)
    keys = []

    for p in pickle_list:
        with open(p, 'rb') as handle:
            key = p.replace('_emb.pickle','')
            emb_dict[key] = pickle.load(handle)
            keys.append(key)
        
    data = ['metacritic', 'amazon', 'imdb']
    # model = ['meta', 'amazon','imdb', 'bert']
    results = defaultdict(list)

    base_model = emb_dict['metatune']
    trans_model = emb_dict['imdbtune']
    
    emb1 = base_model['metacritic']
    emb2 = trans_model['imdb']
            
    key = str(data[0]) + '-base_' + str(data[2]+'-trans')
    print(key)
    
    # dummy label
    label1 = np.array([0 for _ in emb1])
    label2 = np.array([1 for _ in emb2])

    X = np.concatenate((emb1, emb2), axis=0)
    Y = np.concatenate((label1, label2), axis=0)

    # Randomly shuffle data
    np.random.seed(int(round(time.time() * 1000)) % (2**32 - 1))
    shuffle_indices = np.random.permutation(np.arange(len(Y)))
    x_shuffled = X[shuffle_indices]
    y_shuffled = Y[shuffle_indices]

    results = train_svm(x_shuffled, y_shuffled, key, results)           
    
    result2 = defaultdict(list)
    
    for k, i in results.items():
        avg = np.mean(i)
        std = np.std(i)
        result2[k] = [avg, std]
        df = pd.DataFrame.from_dict(result2, orient='index', columns=['A-distance', 'Stdev'])
    print(df)
    # tsv_name = pickle_base.replace('_emb.pickle', '-base_adist.tsv')
    # print(tsv_name)
    # df.to_csv(tsv_name, index=True, sep='\t')


if __name__ == '__main__':
    
    pickle_name = save_emb()
    load_emb(pickle_name)
    
    # os.chdir("/home/ydu/BERT/DATA/EMB/")
    # pickle_list = []
    # for f in glob.glob("*tune*.pickle"):
    #     pickle_list.append(f)
    # print(pickle_list)
    # load_baseline_emb(pickle_list)
    