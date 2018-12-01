from types import SimpleNamespace
from random import randint
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


def train_simple_model(train_iter, val_iter, net, test_iter, optimizer, criterion, num_epochs=5):
    net.train()
    prev_epoch = 0
    train_loss, train_accs, train_length = [0, 0, 0]
    train_res = []
    val_res = []

    for batch in train_iter:

        if (train_iter.epoch != prev_epoch) & (train_iter.epoch % 5 != 0):
            train_loss /= train_length
            train_accs /= train_length
            train_res.append(train_accs)
            val_res.append(None)
            train_loss, train_accs, train_length = [0, 0, 0]

        if (train_iter.epoch != prev_epoch) & (train_iter.epoch % 5 == 0):
            net.eval()
            val_loss, val_accs, val_length = [0, 0, 0]

            for val_batch in val_iter:
                val_output = net(val_batch)
                val_target = Variable(val_batch.author)
                val_loss += criterion(val_output, val_target) * val_batch.batch_size
                val_accs += accuracy_one_hot(val_output, val_target) * val_batch.batch_size
                val_length += val_batch.batch_size

            val_loss /= val_length
            val_accs /= val_length
            val_res.append(val_accs)

            train_loss /= train_length
            train_accs /= train_length
            train_res.append(train_accs)
            print("Epoch {}: Train loss: {:.2f}, Train acc: {:.2f} Validation loss: {:.2f}, Validation acc: {:.2f}"
                  .format(train_iter.epoch, train_loss, train_accs, val_loss, val_accs))

            # plot_res(train_res, val_res, train_iter.epoch)

            net.train()

        net.train()
        output = net(batch)
        target = Variable(batch.author)
        batch_loss = criterion(output, target)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        train_loss += criterion(output, target) * batch.batch_size
        train_accs += accuracy_one_hot(output, target) * batch.batch_size
        train_length += batch.batch_size

        prev_epoch = train_iter.epoch
        if train_iter.epoch == num_epochs:
            break


def train_collaborative_f_model(train_iter, val_iter, net, test_iter, optimizer, criterion, num_auth, num_epochs=5):
    net.train()
    prev_epoch = 0
    train_loss, train_accs, train_length = [0, 0, 0]
    train_res = []
    val_res = []
    for batch in train_iter:
        if (train_iter.epoch != prev_epoch) & (train_iter.epoch % 5 != 0):
            train_loss /= train_length
            train_accs /= train_length
            train_res.append(train_accs)
            val_res.append(None)
            train_loss, train_accs, train_length = [0, 0, 0]

        if (train_iter.epoch != prev_epoch) & (train_iter.epoch % 5 == 0):
            net.eval()
            val_loss, val_accs, val_length = [0, 0, 0]

            for val_batch in val_iter:
                val_output = net(val_batch).reshape(-1)
                val_target = Variable(torch.tensor([1 for _ in range(len(val_batch))]).float().cuda())
                val_loss += criterion(val_output, val_target) * val_batch.batch_size
                val_accs += accuracy_sigmoid(val_output, val_target) * val_batch.batch_size
                val_length += val_batch.batch_size

            val_loss /= val_length
            val_accs /= val_length
            val_res.append(val_accs)

            train_loss /= train_length
            train_accs /= train_length
            train_res.append(train_accs)
            print(
                "Epoch {}: Train loss: {:.2f}, Train avg error: {:.2f} Validation loss: {:.2f}, Validation avg error: {:.2f}"
                    .format(train_iter.epoch, train_loss, train_accs, val_loss, val_accs))
            # plot_res(train_res, val_res, train_iter.epoch)

            net.train()

        net.train()

        pos_and_neg_batch = negative_sampling(batch, num_auth)
        target_pos = Variable(torch.tensor([1 for _ in range(len(batch))]).cuda())
        target_neg = Variable(torch.tensor([0 for _ in range(len(batch))]).cuda())

        output = net(pos_and_neg_batch).reshape(-1)
        target = torch.cat((target_pos, target_neg), 0).float().cuda()
        batch_loss = criterion(output, target)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # print(
        #     "Train loss: {:.3f}, Train avg error: {:.3f}"
        #         .format(criterion(output, target), accuracy_sigmoid(output, target)))

        train_loss += criterion(output, target) * batch.batch_size * 2
        train_accs += accuracy_sigmoid(output, target) * batch.batch_size * 2
        train_length += batch.batch_size * 2

        prev_epoch = train_iter.epoch
        if train_iter.epoch == num_epochs:
            break


def negative_sampling(batch, num_authors):
    random_authors = torch.tensor(
        [randint(0, num_authors) for _ in range(len(batch))]
    ).cuda()

    author = torch.cat((batch.author, random_authors), 0).cuda()
    text = torch.cat((batch.text, batch.text), 1).cuda()

    batch_with_negative_sampling = {'author': author, 'text': text}
    return SimpleNamespace(**batch_with_negative_sampling)


def plot_res(train_res, val_res, num_res):
    x_vals = np.arange(num_res)
    plt.figure()
    plt.plot(x_vals, train_res, 'r', x_vals, val_res, 'b')
    plt.legend(['Train Accucary', 'Validation Accuracy'])
    plt.xlabel('Updates'), plt.ylabel('Acc')


def accuracy_one_hot(output, target):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    correct_prediction = torch.eq(torch.max(output, 1)[1], target)
    # averaging the one-hot encoded vector
    return torch.mean(correct_prediction.float())


def accuracy_sigmoid(output, target):
    return torch.mean(torch.abs(output - target).float())


def print_params(net):
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name, param.data)
