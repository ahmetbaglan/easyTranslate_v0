"""
Model file
"""
import torch
import torch.nn as nn
import torchtext
from torch.nn import Linear
from torch.nn.functional import softmax


class SimpleNet(nn.Module):
    """
    Simple model for article-author paring
    """

    def __init__(self, article_vectors, num_authors, l1=50, l2=50, p1=0.3, p2=0.3, p3=0.3):
        """

        :param article_vectors: Vector dictionary for the article texts
        :type article_vectors: dict
        :param num_authors: Number of authors
        :type num_authors: int
        :param l1: Number of hidden units in the 1st layer
        :param l2: Number of hidden units in the 2nd layer
        :param p1: Dropout probability for the 1st layer
        :param p2: Dropout probability for the 2nd layer
        :param p3: Dropout probability in the output layer
        """
        super(SimpleNet, self).__init__()
        num_embeddings = article_vectors.size()[0]
        embedding_dim = article_vectors.size()[1]

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.copy_(article_vectors)

        self.l_1 = nn.Sequential(
            nn.Dropout(p1),
            Linear(in_features=embedding_dim,
                   out_features=l1,
                   bias=True),
            nn.ReLU(),
        )

        self.l_2 = nn.Sequential(
            nn.Dropout(p2),
            Linear(in_features=l1,
                   out_features=l2,
                   bias=True),
            nn.ReLU(),
        )

        self.l_out = nn.Sequential(
            nn.Dropout(p3),
            Linear(in_features=l1,
                   out_features=num_authors,
                   bias=True),
        )

    def forward(self, x):
        x = self.embeddings(x.text)
        x = torch.mean(x, dim=0)

        x = self.l_1(x)
        # x = self.l_2(x)

        out = softmax(self.l_out(x), dim=1)
        return out


class CollaborativeFilteringNet(nn.Module):
    """
    Colaboratie filtering model for article-author paring
    """

    def __init__(self, article_field, author_field, author_dim=10, l1=50, l2=50, p1=0.3, p2=0.3, p3=0.3):
        """

        :param article_field: Field for the article texts
        :type article_field: torchtext.data.Field
        :param author_field: Field for authors
        :type author_field: torchtext.data.Field
        :param author_dim: Dimensionality of the author embedding
        :param l1: Number of hidden units in the 1st layer
        :param l2: Number of hidden units in the 2nd layer
        :param p1: Dropout probability for the 1st layer
        :param p2: Dropout probability for the 2nd layer
        :param p3: Dropout probability in the output layer
        """
        super(CollaborativeFilteringNet, self).__init__()

        article_vectors = article_field.vocab.vectors
        num_embeddings = article_vectors.size()[0]
        embedding_dim = article_vectors.size()[1]

        self.article_embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.article_embeddings.weight.data.copy_(article_vectors)

        num_author = len(author_field.vocab.freqs)
        self.author_embedding = nn.Embedding(num_author, author_dim)
        self.author_embedding.weight.data.uniform_(0, 200)

        self.l_1 = nn.Sequential(
            nn.Dropout(p1),
            Linear(in_features=(embedding_dim + author_dim),
                   out_features=l1,
                   bias=True),
            nn.ReLU(),
        )

        self.l_2 = nn.Sequential(
            nn.Dropout(p2),
            Linear(in_features=l1,
                   out_features=l2,
                   bias=True),
            nn.ReLU(),
        )

        self.l_out = nn.Sequential(
            nn.Dropout(p3),
            Linear(in_features=l2,
                   out_features=1,
                   bias=True),
        )

    def forward(self, x):
        author = self.author_embedding(x.author)
        text = torch.mean(self.article_embeddings(x.text), dim=0)
        x = torch.cat((author, text), 1)

        x = self.l_1(x)
        x = self.l_2(x)

        out = torch.sigmoid(self.l_out(x))
        return out


class LstmNet(nn.Module):
    def __init__(self, article_field, author_field, author_dim=10, hidden_dim=50, num_layers=5):
        super(LstmNet, self).__init__()
        article_vectors = article_field.vocab.vectors
        num_embeddings = article_vectors.size()[0]
        embedding_dim = article_vectors.size()[1]

        self.article_embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.article_embeddings.weight.data.copy_(article_vectors)

        num_author = len(author_field.vocab.freqs)
        self.author_embedding = nn.Embedding(num_author, author_dim)
        self.author_embedding.weight.data.uniform_(0, 200)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers)

        self.linear = nn.Sequential(
            Linear(in_features=(author_dim + hidden_dim),
                   out_features=1,
                   bias=True),
            nn.ReLU(),
        )

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda(),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda())

    def forward(self, x):
        batch_size = len(x.author)
        author = self.author_embedding(x.author)
        text = self.article_embeddings(x.text)

        hidden = self.init_hidden(batch_size)

        _, (lstm_hidden, lstm_state) = self.lstm(text, hidden)

        x = torch.cat((author, lstm_state[-1]), 1).cuda()
        x = self.linear(x)
        out = torch.sigmoid(x)

        return out
