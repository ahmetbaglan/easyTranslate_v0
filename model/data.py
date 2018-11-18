from torchtext import data
import torch
import spacy
from torchtext.vocab import Vectors


class Articles:
    """
    Article
    """

    def __init__(self, batch_size=100):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.spacy_en = spacy.load('en')

        self.text = data.Field(sequential=True, tokenize=self.tokenizer, lower=True)
        self.author = data.Field(sequential=False, use_vocab=True)

        self.train_set, self.test_set, self.validation_set = data.TabularDataset(
            path='./data/test.csv',
            format='csv',
            fields=[
                ('index', None),
                ('id', None),
                ('title', None),
                ('publication', None),
                ('author', self.author),
                ('date', None),
                ('year', None),
                ('month', None),
                ('url', None),
                ('text', self.text)
            ],
            skip_header=True,
        ).split(split_ratio=[0.7, 0.15, 0.15])

        self.train_iter, self.test_iter, self.validation_iter = data.BucketIterator.splits(
            (self.train_set, self.validation_set, self.test_set),
            batch_size=batch_size,
            device=device,
            sort_key=lambda x: len(x.author))

        url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
        self.text.build_vocab(self.train_set, max_size=None, vectors=Vectors('wiki.simple.vec', url=url))
        self.author.build_vocab(self.train_set)

    def tokenizer(self, text):  # create a tokenizer function
        return [tok.text for tok in self.spacy_en.tokenizer(text)]


if __name__ == "__main__":
    articles = Articles()
    train_iter = articles.train_iter
