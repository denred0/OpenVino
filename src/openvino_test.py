import numpy as np
import timm
import torch
import os
import random
import torchvision.transforms as transforms
import torch.nn as nn

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from time import time

from torch.utils.data import WeightedRandomSampler, DataLoader
from torch.utils.data import Subset
import torchvision.datasets as datasets


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    # datasets = {}
    # datasets['train'] = Subset(dataset, train_idx)
    # datasets['val'] = Subset(dataset, val_idx)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def main(data_dir, transforms):
    seed_everything(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load image dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transforms)

    # split on train and val + test
    train_dataset, val_test_dataset = train_val_dataset(dataset, val_split=0.3)

    # calcuate class_weights
    class_weights = []
    count_all_files = 0
    for root, subdir, files in os.walk(data_dir):
        if len(files) > 0:
            class_weights.append(len(files))
            count_all_files += len(files)

    class_weights = [x / count_all_files for x in class_weights]
    print('class_weights', class_weights)

    # get train WeightedRandomSamplers
    sample_weights = [0] * len(train_dataset)
    for idx, (data, label) in enumerate(train_dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # split val and test
    val_dataset, test_dataset = train_val_dataset(val_test_dataset, val_split=0.5)

    # create loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # start training
    model_type = 'inception_v4'
    b_epoch = 5
    num_classes = len(dataset.classes)
    train(model_type, num_classes, device, train_loader, val_loader, no_epochs=b_epoch)

    # test
    model_test, taken_pytorch, acc_pytorch = pytorch_test(model_type=model_type, num_classes=num_classes,
                                                          device=device, test_loader=test_loader, b_epoch=b_epoch)

    print("\nAccuracy PyTorch is: {:.2f}%".format(acc_pytorch))
    print("Time taken PyTorch: {:.2f}s".format(taken_pytorch))


def train(model_type, num_classes, device, train_loader, val_loader, no_epochs, class_weights=None):
    model = timm.create_model(model_type, pretrained=True)
    in_features = model.last_linear.in_features
    # model.last_linear = nn.Linear(in_features, num_classes)
    model.classifier = nn.Linear(in_features, num_classes)
    model.to(device)
    print(model)

    if class_weights:
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    val_loss_min = np.Inf

    for epoch in range(1, no_epochs + 1):

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
            '\nEpoch: {} \tTime: {:.3f} \nTraining Loss: {:.6f} \tTraining Acc: {:.2f} \tValidation Loss: {:.6f} \tValidation Acc: {:.2f}'.format(
                epoch, taken, train_loss, train_acc, val_loss, val_acc))

        if val_loss <= val_loss_min:
            print(
                'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(val_loss_min, val_loss))
            torch.save(model.state_dict(), 'model_inceptionv4.pth')
            val_loss_min = val_loss


def pytorch_test(model_type, num_classes, device, test_loader, b_epoch):
    model_test = timm.create_model(model_type, pretrained=True)
    in_features = model_test.last_linear.in_features
    # model_test.last_linear = nn.Linear(in_features, num_classes)
    model_test.classifier = nn.Linear(in_features, num_classes)
    model_test.load_state_dict(torch.load('model_inceptionv4.pth'))
    model_test.to(device)
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
    my_transforms = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ]
    )

    data_dir = 'data'
    main(data_dir=data_dir, transforms=my_transforms)
