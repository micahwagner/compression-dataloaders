import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timeit import default_timer as timer
from dataloader import DataLoader
from matplotlib import pyplot as plt

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def main():
    
    jpeg = True

    training_sets = []
    for x in range(1,6):
        with open('cifar/data/data_batch_'+f'{x}', 'rb') as file:
            dict = pickle.load(file, encoding='bytes')
        training_sets.append(dict)
        with open('cifar/data/test_batch', 'rb') as file:
            dict = pickle.load(file, encoding='bytes')
        test_set = dict
    # keys in dictionary are byte strings, do this:
    # print(test_set[b'data'])
    # b'labels', b'data' 
    # data is 10,000 x 3,072. 10,000 images, all 32x32. 3,072 / 3 = 1,024 corresponding to RGB channels.

    x_train = []
    y_train = []

    for batch in training_sets:
        x_train.extend(batch[b'data'])
        y_train.extend(batch[b'labels'])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_valid = np.array(test_set[b'data'])
    y_valid = np.array(test_set[b'labels'])

    dataloader = DataLoader(x_train, y_train, batch_size=128, shuffle=True, jpeg=jpeg)
    validloader = DataLoader(x_valid, y_valid, batch_size=10000, shuffle=True, jpeg=jpeg)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
    model = ResNet(ResidualBlock, [2, 2, 2, 2], 10)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    epochs = 25
    losses = []
    valid_losses = []
    min_loss = float('inf')
    best_epoch = 0
    train_accuracy = 0
    test_accuracy = 0

    print(f"Training for {epochs} epochs...")
    start1 = timer() * 1000000
    # training and validation loop
    for epoch in range(1,epochs+1):
        # training loop
        loss_total = 0.0
        train_count = 0
        for id_batch, (x_batch, y_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model.forward(x_batch)
            loss = criterion(y_pred, y_batch)
            loss_total += loss.cpu().detach().numpy()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_count += 1
        if epoch % 5 == 0:
            print("Epoch " + str(epoch) + " training loss: " + str(loss_total/train_count))
        losses.append(loss_total/train_count)

        # validation loop
        with torch.no_grad():
            loss_total = 0.0
            valid_count = 0
            for id_batch, (x_batch, y_batch) in enumerate(validloader):
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                loss = loss.cpu().detach().numpy()
                loss_total += loss
                valid_count += 1
            curr_loss = loss_total/valid_count
            if epoch % 5 == 0:
                print("Epoch " + str(epoch) + " validation loss: " + str(curr_loss))
            valid_losses.append(curr_loss)
            if curr_loss < min_loss:
                min_loss = curr_loss
                best_epoch = epoch
                torch.save(model.state_dict(), 'cifar/best_model')
    
    end1 = timer() * 1000000
    training_time_taken = end1 - start1

    model.load_state_dict(torch.load('cifar/best_model', weights_only=True))
    with torch.no_grad():
        correct = 0
        total = 0
        loss_total = 0.0
        train_count = 0
        for id_batch, (x_batch, y_batch) in enumerate(dataloader):
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss = loss.cpu().detach().numpy()
            loss_total += loss
            _, predicted = torch.max(y_pred, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            train_count += 1
        train_accuracy = 100 * correct / total
        train_loss = loss_total / train_count

        correct = 0
        total = 0
        loss_total = 0.0
        test_count = 0
        for id_batch, (x_batch, y_batch) in enumerate(validloader):
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss = loss.cpu().detach().numpy()
            loss_total += loss
            _, predicted = torch.max(y_pred, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            test_count += 1
            test_accuracy = 100 * correct / total
        test_loss = loss_total / test_count
    
    print("----------------------------\n")
    print("Best Epoch: " + str(best_epoch))
    print("Training Loss: " + str(train_loss))
    print("Testing Loss: " + str(test_loss))
    print("Training Accuracy: " + str(train_accuracy) + " %" )
    print("Testing Accuracy: " + str(test_accuracy) + " %")
    print("Time Spent Training: " + str(training_time_taken))
    print("Time Per Epoch: " + str(training_time_taken/epochs))

    file_name = "results/cifar"
    if jpeg:
        file_name += "_jpeg"
    
    with open(file_name + ".txt", 'w') as file:
        file.write("Best Epoch: " + str(best_epoch) + "\n")
        file.write("Training Loss: " + str(train_loss) + "\n")
        file.write("Testing Loss: " + str(test_loss) + "\n")
        file.write("Training Accuracy: " + str(train_accuracy) + " %" + "\n")
        file.write("Testing Accuracy: " + str(test_accuracy) + " %" + "\n")
        file.write("Time Spent Training: " + str(training_time_taken) + "\n")
        file.write("Time Per Epoch: " + str(training_time_taken/epochs))

    plt.plot(range(1,epochs+1), losses, linestyle = '--', label='Training Loss', color = 'darkslateblue')
    plt.plot(range(1,epochs+1), valid_losses, label='Validation Loss', color='darkorange')
    legend = plt.legend(fancybox=False, edgecolor="black")
    legend.get_frame().set_linewidth(0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy')
    plt.title(f'ResNet18 on CIFAR-10 w/ JPEG={jpeg}')
    plt.savefig(file_name + ".pdf")

if __name__ =='__main__':
    main()