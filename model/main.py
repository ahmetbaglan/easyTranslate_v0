"""
Main
"""

from torch import optim, nn
from model import SimpleNet, CollaborativeFilteringNet, LstmNet
from data import Articles
from train import train_collaborative_f_model, train_simple_model

articles = Articles(batch_size=20)
authors = articles.author
text = articles.text

train_iter = articles.train_iter
test_iter = articles.test_iter
validation_iter = articles.validation_iter

num_authors = len(authors.vocab.itos)
text_vector = text.vocab.vectors

print('Data loaded.')

# simple_net = SimpleNet(article_vectors=text_vector, num_authors=num_authors).cuda()
# simple_net_opt = optim.Adam(simple_net.parameters(), 1e-4, weight_decay=1e-5)
# simple_net_criterion = nn.CrossEntropyLoss()
#
# train_simple_model(train_iter=train_iter, test_iter=test_iter, val_iter=validation_iter,
#                    net=simple_net, optimizer=simple_net_opt, criterion=simple_net_criterion, num_epochs=50)

##################################################################################

# col_filt = CollaborativeFilteringNet(article_field=text, author_field=authors, author_dim=10).cuda()
# col_filt_opt = optim.Adam(col_filt.parameters(), 1e-4, weight_decay=1e-5)
# col_filt_criterion = nn.MSELoss()
#
# train_collaborative_f_model(train_iter=train_iter, test_iter=test_iter, val_iter=validation_iter, num_auth=num_authors,
#                             net=col_filt, optimizer=col_filt_opt, criterion=col_filt_criterion, num_epochs=50)

lstm = LstmNet(article_field=text, author_field=authors, author_dim=10).cuda()
col_filt_opt = optim.Adam(lstm.parameters(), 1e-4, weight_decay=1e-5)
col_filt_criterion = nn.BCELoss()

train_collaborative_f_model(train_iter=train_iter, test_iter=test_iter, val_iter=validation_iter, num_auth=num_authors,
                            net=lstm, optimizer=col_filt_opt, criterion=col_filt_criterion, num_epochs=100)
