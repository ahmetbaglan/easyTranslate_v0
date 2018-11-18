"""
Model file
"""
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn.functional import softmax, relu


class Net(nn.Module):
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
        super(Net, self).__init__()
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
            Linear(in_features=l2,
                   out_features=num_authors,
                   bias=True),
        )

    def forward(self, x):
        x = self.embeddings(x.text)
        x = torch.mean(x, dim=0)

        x = self.l_1(x)
        x = self.l_2(x)

        out = softmax(self.l_out(x), dim=1)
        return out
