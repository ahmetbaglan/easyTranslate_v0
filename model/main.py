"""
Main
"""

from torch import optim, nn
from model import Net
from data import Articles
from train import train

articles = Articles(batch_size=100)

train_iter = articles.train_iter
test_iter = articles.test_iter
validation_iter = articles.validation_iter

authors = articles.author
num_authors = len(authors.vocab.stoi)
texts = articles.text
text_vector = texts.vocab.vectors

net = Net(article_vectors=text_vector, num_authors=num_authors).cuda()
opt = optim.Adam(net.parameters(), 1e-4, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

train(train_iter=train_iter, test_iter=test_iter, val_iter=validation_iter,
      net=net, optimizer=opt, criterion=criterion, num_epochs=50)
