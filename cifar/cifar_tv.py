import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from timeit import default_timer as timer
from dataloader import DataLoader
from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet18

def main():
	jpeg = True

	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
	])

	transform_test = transforms.Compose([
    	transforms.ToTensor(),
    	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
	])
	train_ds = CIFAR10(root='./torch_data', train=True, download=True, transform=transform_train)
	test_ds = CIFAR10(root='./torch_data', train=False, download=True, transform=transform_test)

	dataloader = DataLoader(train_ds.data, train_ds.targets, batch_size=128, shuffle=True, jpeg=jpeg, Q=10, subsampling="4:2:2")
	validloader = DataLoader(test_ds.data, test_ds.targets, batch_size=10000, shuffle=True, jpeg=jpeg, Q=10, subsampling="4:2:2")

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = resnet18(weights=None)
	#cifar10 only classifies 10 things, get rid of early downsampling
	model.maxpool = nn.Identity()
	model.fc = nn.Linear(model.fc.in_features, 10)
	model = model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
	epochs = 200
	losses = []
	valid_losses = []
	min_loss = float('inf')
	best_epoch = 0

	print(f"Training for {epochs} epochs...")
	start1 = timer() * 1000000
	for epoch in range(1, epochs+1):
		model.train()
		running_loss, steps = 0.0, 0

		for imgs, labels in dataloader:
			optimizer.zero_grad()
			loss = criterion(model(imgs), labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			steps += 1
		scheduler.step()

		if epoch % 5 == 0:
			print("Epoch " + str(epoch) + " training loss: " + str(running_loss/steps))
		losses.append(running_loss/steps)

		model.eval()
		loss_total, valid_count = 0.0, 0
		with torch.no_grad():
			for imgs, labels in validloader:
				y_pred = model(imgs)
				loss = criterion(y_pred, labels).cpu().detach().item()
				loss_total += loss
				valid_count += 1
			
			curr_loss = loss_total / valid_count
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
	model.eval()
	loss_total = 0.0
	correct = total = train_count = 0
	with torch.no_grad():
		for imgs, labels in dataloader:
			y_pred = model(imgs)
			loss = criterion(y_pred, labels).cpu().detach().item()
			loss_total += loss
			_, predicted = torch.max(y_pred, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			train_count += 1
	train_accuracy = 100 * correct / total
	train_loss     = loss_total / train_count

	loss_total = 0.0
	correct = total = test_count = 0
	with torch.no_grad():
		for imgs, labels in validloader:
			y_pred = model(imgs)
			loss = criterion(y_pred, labels).cpu().detach().item()
			loss_total += loss
			_, predicted = torch.max(y_pred, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			test_count += 1
	test_accuracy = 100 * correct / total
	test_loss = loss_total / test_count

	print("----------------------------\n")
	print(f"Best Epoch: {best_epoch}")
	print(f"Training Loss: {train_loss:.4f}")
	print(f"Testing  Loss: {test_loss:.4f}")
	print(f"Training Accuracy: {train_accuracy:.2f}%")
	print(f"Testing  Accuracy: {test_accuracy:.2f}%")
	print(f"Time Spent Training: {training_time_taken:.0f} μs")
	print(f"Time Per Epoch: {training_time_taken/epochs:.0f} μs")

	file_name = "results/cifar" + (f"_jpeg_{dataloader.Q}_{dataloader.subsampling}" if jpeg else "")
	with open(file_name + ".txt", 'w') as f:
		f.write(f"Best Epoch: {best_epoch}\n")
		f.write(f"Training Loss: {train_loss:.4f}\n")
		f.write(f"Testing  Loss: {test_loss:.4f}\n")
		f.write(f"Training Accuracy: {train_accuracy:.2f}%\n")
		f.write(f"Testing  Accuracy: {test_accuracy:.2f}%\n")
		f.write(f"Time Spent Training: {training_time_taken:.0f} μs\n")
		f.write(f"Time Per Epoch: {training_time_taken/epochs:.0f} μs")

	plt.plot(range(1, epochs+1), losses,linestyle='--', label='Training Loss')
	plt.plot(range(1, epochs+1), valid_losses,label='Validation Loss')
	legend = plt.legend(fancybox=False, edgecolor="black")
	legend.get_frame().set_linewidth(0.5)
	plt.xlabel('Epoch')
	plt.ylabel('Cross Entropy')
	plt.title(f'ResNet18 on CIFAR-10 w/ JPEG={jpeg}')
	plt.savefig(file_name + ".pdf")

if __name__ == "__main__":
	main()
