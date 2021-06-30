import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets, models
import torchvision.transforms as transforms

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from time import time


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.4)

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.drop(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = self.pool(x)
        x = self.drop(x)

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.log_softmax(self.fc3(x), dim=1)

        return x


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)


def train(device, train_loader, val_loader):
    model = Net()

    model.parameters()

    model.apply(init_weights).to(device)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    no_epochs = 1 + 10

    val_loss_min = np.Inf

    for epoch in range(1, no_epochs):

        start = time()

        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        model.train()
        for idx, (data, target) in tqdm(enumerate(train_loader)):
            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = criterion(output, target)

            optimizer.zero_grad()

            loss.backward()

            # with amp.scale_loss(loss,optimizer) as scaled_loss:
            #    scaled_loss.backward()

            optimizer.step()

            train_loss += loss.item()

            _, pred = torch.max(output, dim=1)

            equals = pred == target.view(*pred.shape)

            train_acc += torch.mean(equals.type(torch.FloatTensor))

        model.eval()
        for idx, (data, target) in tqdm(enumerate(val_loader)):
            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = criterion(output, target)

            val_loss += loss.item()

            _, pred = torch.max(output, dim=1)

            equals = pred == target.view(*pred.shape)

            val_acc += torch.mean(equals.type(torch.FloatTensor))

        train_loss = train_loss / len(train_loader)
        train_acc = train_acc * 100 / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_acc = val_acc * 100 / len(val_loader)

        end = time()
        taken = end - start

        print(
            'Epoch: {} \tTime: {:.3f} \nTraining Loss: {:.6f} \tTraining Acc: {:.2f} \tValidation Loss: {:.6f} \tValidation Acc: {:.2f}'.format(
                epoch, taken, train_loss, train_acc, val_loss, val_acc))

        if val_loss <= val_loss_min:
            print(
                'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(val_loss_min, val_loss))
            torch.save(model.state_dict(), 'model_cifar.pth')
            val_loss_min = val_loss


def pytorch_test(device, test_loader, b_epoch):
    model_test = Net()
    model_test.load_state_dict(torch.load('model_cifar.pth'))
    model_test = model_test.to(device)
    model_test.eval()

    test_acc = 0
    start = time()
    for i in range(b_epoch):

        for idx, (data, target) in tqdm(enumerate(test_loader)):
            data, target = data.to(device), target.to(device)

            # optimizer.zero_grad()

            output = model_test(data)

            _, pred = torch.max(output, dim=1)

            equal = pred == target.view(*pred.shape)

            test_acc += torch.mean(equal.type(torch.FloatTensor))

    test_acc /= b_epoch
    taken = time() - start
    print("Accuracy is: {:.2f}%".format(test_acc * 100 / len(test_loader)))
    print("Time taken: {:.2f}s".format(taken))

    acc = test_acc * 100 / len(test_loader)

    return model_test, taken, acc


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomRotation(20),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = datasets.CIFAR10(
        root='data_cifar', train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10(
        root='data_cifar', train=False, download=True, transform=test_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    # train/val split
    val_size = 0.1
    split = int(np.floor((val_size * num_train)))
    train_idx, val_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    batch_size = 100

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    no_classes = len(classes)

    # visualization
    # data_iter = iter(train_loader)
    # images, labels = data_iter.next()
    # images = images.numpy()

    # fig = plt.figure(figsize=(9, 9))
    # fig.suptitle('CIFAR10', fontsize=18)

    # for im in np.arange(batch_size):
    #     fig.tight_layout()  #
    #     ax = fig.add_subplot(10, int(batch_size / 10), im + 1, xticks=[], yticks=[])
    #     ax.set_title(classes[labels[im]], fontsize=8)
    #     # plt.rcParams.update({'axes.titlesize': 'small'})
    #     images[im] = images[im] / 2 + 0.5
    #     plt.imshow(np.transpose(images[im], (1, 2, 0)))

    # plt.show()

    # train
    train(device=device, train_loader=train_loader, val_loader=val_loader)

    # pytorch native inference
    b_epoch = 10
    model_test, taken_pytorch, acc_pytorch = pytorch_test(
        device=device, test_loader=test_loader, b_epoch=b_epoch)

    print("Accuracy PyTorch is: {:.2f}%".format(acc_pytorch))
    print("Time taken PyTorch: {:.2f}s".format(taken_pytorch))
