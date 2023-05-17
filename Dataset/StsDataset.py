from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from string import punctuation
import re

import nltk
nltk.download('stopwords')


def _on_bad_line(values):
    return values[:7]


def preprocess(sentence, stop_words):
    result = re.sub(f'[{punctuation}]', ' ', sentence).lower()
    result = re.sub('\W', ' ', result).split()
    return [w for w in result if w not in stop_words]


class StsDataset(object):

    def __init__(self) -> None:
        super(StsDataset, self).__init__()
        self.paths = {
            'train': './Dataset/sts-train.csv',
            'val': './Dataset/sts-dev.csv',
            'test': './Dataset/sts-test.csv'
        }
        self.columns_mapping = {
            0: 'genre',
            1: 'filename',
            2: 'year',
            3: 'index',
            4: 'score',
            5: 'sentence1',
            6: 'sentence2'
        }
        self.stop_words = stopwords.words('english')

    def load_csv(self, target='train'):
        return pd.read_csv(
            self.path[target],
            sep="\t",
            on_bad_lines=_on_bad_line,
            engine='python',
            header=None,
            encoding='utf-8',
            quoting=3
        ).rename(columns=self.columns_mapping)

    def get_raw_sentences(self, df: pd.DataFrame):
        sentences1 = df["sentence1"]
        sentences2 = df["sentence2"]
        temp = np.concatenate((sentences1, sentences2))
        return np.unique(temp)

    def get_tokenized_sentences(self, sentences):
        return [preprocess(s, self.stop_words) for s in sentences]