import os
import json_lines
import zipfile
import h5py
import numpy as np
import re

from torch.utils.data import Dataset


class MultiNLIDataset(Dataset):
    def __init__(self,
                 dataset: dict):
        self.s1 = dataset['s1']
        self.s2 = dataset['s2']
        self.label = dataset['label']
        assert len(self.s1) == len(self.s2) == len(self.label)

    def __getitem__(self, index):
        return self.s1[index], self.s2[index], self.label[index]

    def __len__(self):
        return len(self.s1)


def tokenize(string):
    string = re.sub(r'\(|\)', '', string)
    return string.split()


def get_multinli(data_path: str,
                 prefix: str,
                 suffix: str,
                 dataset: str,
                 genres: list = None) -> dict:
    path = os.path.join(data_path, prefix + dataset + suffix)

    labels = {'entailment': 0,
              'neutral': 1,
              'contradiction': 2}

    with open(path) as f:
        data = [item for item in json_lines.reader(f)]

    s1, s2, label = [], [], []

    for entry in data:
        if genres is None or entry['genre'] in genres:
            if entry['gold_label'] in labels:
                s1.append(entry['sentence1'])
                s2.append(entry['sentence2'])
                label.append(labels[entry['gold_label']])

    return {'s1': s1,
            's2': s2,
            'label': label}


def download_and_unzip(root: str,
                       urls: list,
                       dir: str,
                       name: str):
    for url in urls:
        filename = os.path.basename(url)
        out_image = os.path.join(os.path.join(root, name), filename)
        path = os.path.join(root, dir)
        if not os.path.exists(os.path.join(root, name)):
            os.makedirs(os.path.join(root, name))
        if not os.path.isfile(out_image):
            os.system("wget -O {0} {1}".format(out_image, url))

            with zipfile.ZipFile(out_image, "r") as zip_ref:
                zip_ref.extractall(path)


def build_vocab(sentences: list,
                max_len: int = None) -> (dict, int):
    vocab = {}
    i = 0
    l = 0
    if max_len is None:
        max_len = 1000
    for sentence in sentences:
        tokens = tokenize(sentence)
        if l < len(tokens):
            l = min(len(tokens), max_len)
        for j, word in enumerate(tokens):
            if word not in vocab and j < max_len:
                vocab[word] = i
                i += 1

    vocab['<s>'] = j
    vocab['</s>'] = j + 1
    vocab['<p>'] = j + 2
    return vocab, l


def load_glove(glove_path: str) -> dict:
    glove = {}
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            glove[word] = np.array(list(map(float, vec.split())))
    return glove


def build_vocab_vectors_with_glove(vocab: dict,
                                   glove: dict) -> np.ndarray:
    weights_matrix = np.random.normal(scale=0.5, size=(len(vocab), 300))

    for word, i in vocab.items():
        if word in glove:
            weights_matrix[i] = glove[word]
    return weights_matrix


def format_dataset(dataset: dict,
                   vocab: dict,
                   max_len: int) -> dict:
    for keys in ['s1', 's2']:
        sentences = []
        for sentence in dataset[keys]:
            cur_sentence = [vocab['<s>']]
            word_list = tokenize(sentence)
            cur_sentence += [vocab[word_list[i]] for i in range(min(len(word_list), max_len-2))]
            cur_sentence += [vocab['</s>']]
            if len(cur_sentence) < max_len:
                cur_sentence += [vocab['<p>']] * (max_len - len(cur_sentence))
            sentences.append(cur_sentence)
        dataset[keys] = np.array(sentences)

    return dataset


def prepare_glove(glove_path: str,
                  vocab: dict):
    if os.path.isfile('datasets/GloVe/weight_matrix.h5'):
        with h5py.File('datasets/GloVe/weight_matrix.h5', 'r') as hf:
            weight_matrix = hf['weight_matrix'][:]
    else:
        glove = load_glove(glove_path=glove_path)
        weight_matrix = build_vocab_vectors_with_glove(vocab=vocab, glove=glove)

        with h5py.File('datasets/GloVe/weight_matrix.h5', 'w') as hf:
            hf.create_dataset("weight_matrix", data=weight_matrix)
    return weight_matrix


def prepare_mnli(root: str,
                 urls: list,
                 dir: str,
                 name: str,
                 data_path: str,
                 max_len: int = None) -> (dict, dict, dict):

    prefix = 'multinli_1.0_'
    suffix = '.jsonl'
    genres = ['fiction',
              'government',
              'slate',
              'telephone',
              'travel']

    download_and_unzip(root=root,
                       urls=urls,
                       dir=dir,
                       name=name)

    train = get_multinli(data_path=data_path,
                         prefix=prefix,
                         suffix=suffix,
                         dataset='train',
                         genres=genres)

    dev_matched = get_multinli(data_path=data_path,
                               prefix=prefix,
                               suffix=suffix,
                               dataset='dev_matched',
                               genres=genres)

    datasets = train['s1'] + train['s2'] + dev_matched['s1'] + dev_matched['s2']

    vocab, max_len = build_vocab(sentences=datasets,
                                 max_len=max_len)

    train = format_dataset(dataset=train,
                           vocab=vocab,
                           max_len=max_len)

    dev_matched = format_dataset(dataset=dev_matched,
                                 vocab=vocab,
                                 max_len=max_len)

    return train, dev_matched, vocab


def prepare_mnli_split(root: str,
                       urls: list,
                       dir: str,
                       name: str,
                       data_path: str,
                       train_genres: list,
                       test_genres: list,
                       max_len: int = None) -> (list, list, list, list, dict):

    prefix = 'multinli_1.0_'
    suffix = '.jsonl'

    download_and_unzip(root=root,
                       urls=urls,
                       dir=dir,
                       name=name)

    train = []
    dev_matched_train = []
    for genres in train_genres:
        train.append(get_multinli(data_path=data_path,
                                  prefix=prefix,
                                  suffix=suffix,
                                  dataset='train',
                                  genres=genres))

        dev_matched_train.append(get_multinli(data_path=data_path,
                                              prefix=prefix,
                                              suffix=suffix,
                                              dataset='dev_matched',
                                              genres=genres))
    test = []
    dev_matched_test = []
    for genres in test_genres:
        test.append(get_multinli(data_path=data_path,
                                 prefix=prefix,
                                 suffix=suffix,
                                 dataset='train',
                                 genres=genres))

        dev_matched_test.append(get_multinli(data_path=data_path,
                                             prefix=prefix,
                                             suffix=suffix,
                                             dataset='dev_matched',
                                             genres=genres))

    datasets = []
    for t in train + dev_matched_train + test + dev_matched_test:
        datasets += t['s1'] + t['s2']

    vocab, max_len = build_vocab(sentences=datasets,
                                 max_len=max_len)

    train = [format_dataset(dataset=dataset,
                            vocab=vocab,
                            max_len=max_len) for dataset in train]

    dev_matched_train = [format_dataset(dataset=dataset,
                                        vocab=vocab,
                                        max_len=max_len) for dataset in dev_matched_train]

    test = [format_dataset(dataset=dataset,
                           vocab=vocab,
                           max_len=max_len) for dataset in test]

    dev_matched_test = [format_dataset(dataset=dataset,
                                       vocab=vocab,
                                       max_len=max_len) for dataset in dev_matched_test]

    return train, dev_matched_train, test, dev_matched_test, vocab
