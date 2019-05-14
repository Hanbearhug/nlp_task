import pandas as pd
from tqdm import tqdm
import jieba
from collections import defaultdict, Counter
from itertools import chain
import  json
tqdm.pandas()


train = pd.read_table('cnews/cnews.train.txt', header=None, names=['type', 'content'])
test = pd.read_table('cnews/cnews.test.txt', header=None, names=['type', 'content'])

def read_stopwords():
    with open('cnews/stopwords.txt', encoding='utf-8') as f:
        stopwords = [i.strip() for i in f.readlines()]
    return stopwords

def delete_stopwords(df, stopwords):
    def delete_single_stopwords(sent, stopwords=stopwords):
        for stopword in stopwords:
            sent = sent.replace(stopword, '')
        return sent

    df['content'] = df['content'].progress_apply(delete_single_stopwords)
    print('已完成去除停用词！')
    return df

def test_delete_stop_words(df, stopwords):
    for stopword in stopwords:
        if df['content'].str.contains(stopword).all():
            print(f'{stopword}')
            break

def partition_df(df, type):
    df['content'] = df['content'].apply(lambda r: jieba.cut(r, type=type))
    return df

def partition_by_num(df, num):
    def partition(sent, num=num):
        sent_gram = [sent[i:i+num] for i in range(0, len(sent)-num)]
        return sent_gram
    df['content'] = df['content'].apply(partition)
    return df


def build_vocab(df, name, freq_cutoff):
    def bulid_corpus(df):
        corpus = []
        for sent in df['content']:
            corpus+=sent
        return corpus
    corpus = bulid_corpus(df)
    word_freq = Counter(chain(*corpus))
    vocab = [w for w, v in word_freq.items() if v >= freq_cutoff]
    json.dump(vocab, f'vocab/{name.json}')
    return vocab



if __name__=='__main__':
    stopwords = read_stopwords()
    train = delete_stopwords(train, stopwords)
    test = delete_stopwords(test, stopwords)
    test_delete_stop_words(train, ["，", "。"])
    test_delete_stop_words(test, ["，", "。"])
    print('pass test!')

