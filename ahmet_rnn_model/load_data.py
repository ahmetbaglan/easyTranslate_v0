# _*_ coding: utf-8 _*_

import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
import spacy

spacy_en = spacy.load('en')

def load_dataset(test_sen=None):
    print("in load_dataset")

    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.
                 
    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.
                  
    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    
    """
    #
    def tokenizer(text):  # create a tokenizer function
        return [tok.text for tok in spacy_en.tokenizer(text)]

    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=200)
    LABEL = data.Field(tensor_type=torch.FloatTensor, sequential=False)

    # tokenize = lambda x: x.split()
    # TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)
    # LABEL = data.LabelField(tensor_type=torch.FloatTensor)
    # train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    # print('data loaded')
    train_data, valid_data, test_data = data.TabularDataset.splits(
        path='../Github/Data/author_identification/', train='train.csv',
        validation='val.csv', test='test.csv', format='csv',
        fields=[('id', None), ('text', TEXT), ('author', LABEL)])
    #
    # TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    # LABEL.build_vocab(train_data)
    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))

    from torchtext.data import Iterator, BucketIterator

    # train_iter, valid_iter = BucketIterator.splits(
    #     (train_data, valid_data),  # we pass in the datasets we want the iterator to draw data from
    #     batch_size=64,
    #     device=-1,  # if you want to use the GPU, specify the GPU number here
    #     sort_key=lambda x: len(x.text),
    #     # the BucketIterator needs to be told what function it should use to group the data.
    #     sort_within_batch=False,
    #     repeat=False  # we pass repeat=False because we want to wrap this Iterator layer.
    # )
    # test_iter = Iterator(test_data, batch_size=64, device=-1, sort=False, sort_within_batch=False, repeat=False)
    # train_data, valid_data = train_data.split() # Further splitting of training_data to create new training_data & validation_data

    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)

    '''Alternatively we can also use the default configurations'''
    # train_iter, test_iter = datasets.IMDB.iters(batch_size=32)

    vocab_size = len(TEXT.vocab)

    return TEXT, LABEL, vocab_size, word_embeddings, train_iter, valid_iter, test_iter
