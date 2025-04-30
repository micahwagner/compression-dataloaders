import torch
torch.cuda.set_per_process_memory_fraction(0.38, 0)
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer as timer
from lazy_dataloader import Dataset, DataLoader
from matplotlib import pyplot as plt
import json
import os

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
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
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
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def main():

    jpeg = False
    offline = True

    base_path = 'imagenet/data'
    train_path = base_path + '/train/'
    valid_path = base_path + '/valid/'

    if offline:
        base_path = 'imagenet/jpeg_data'
        train_path = base_path + '/train/'
        valid_path = base_path + '/valid/'

    # create mapping between id and labels
    mapping = {}
    with open(base_path + '/Labels.json', 'r') as file:
        labels = json.load(file)
        for index, line in enumerate(labels):
            mapping[line] = int(index)

    y_train = []
    train_order = []
    for folder in os.listdir(train_path):
        for file in os.listdir(train_path+folder):
            y_train.append(mapping[folder])
            train_order.append(folder+"/"+file)

    y_valid = []
    valid_order = []
    for folder in os.listdir(valid_path):
        for file in os.listdir(valid_path+folder):
            y_valid.append(mapping[folder])
            valid_order.append(folder+"/"+file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = Dataset(train_order, train_path, y_train, jpeg=jpeg)
    valid_dataset = Dataset(valid_order, valid_path, y_valid, jpeg=jpeg)
    dataloader = DataLoader(train_dataset, batch_size=250, num_workers=12, shuffle=True)
    validloader = DataLoader(valid_dataset, batch_size=250, num_workers=12, shuffle=True)

    model = ResNet(ResidualBlock, [2, 2, 2, 2], 1000)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 15
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
            train_count += 1
            #print(id_batch)
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
            print("Epoch " + str(epoch) + " validation loss: " + str(curr_loss))
            valid_losses.append(curr_loss)
            if curr_loss < min_loss:
                min_loss = curr_loss
                best_epoch = epoch
                torch.save(model.state_dict(), 'imagenet/best_model')
    
    end1 = timer() * 1000000
    training_time_taken = end1 - start1

    model.load_state_dict(torch.load('imagenet/best_model', weights_only=True))
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

    file_name = "results/imagenet"
    if offline:
        file_name += "_offline"
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
    plt.title(f'ResNet18 on ImageNet100 w/ JPEG={jpeg} & Offline={offline}')
    plt.savefig(file_name + ".pdf")

if __name__ =='__main__':
    main()