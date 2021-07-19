import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
from torchvision import datasets
from data_loader import speckle_dataset
from model import LeNet
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim



net = LeNet().to(device)
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(net.parameters(), 0.0001, (0.9,0.9), 1e-4)
# Specify path, transform and batchsize for obtaining the data 
transform = T.Compose([
    T.ToTensor()
])
root_dir = '../data'
batch_size = 4
#Load data from the class data_loader
train, test = speckle_dataset(batch_size=batch_size, root_dir=root_dir, transform=transform)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



def evaluation(dataloader):
        total, correct = 0, 0
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        return 100 * correct / total


def run(opt = optim.Adam(net.parameters(), 0.0001, (0.9,0.9), 1e-4)):

    # %%time
    max_epochs = 50
    loss_epochs_arr = []
    loss_arr = []
    for epoch in range(max_epochs):
        print(epoch)
        for i, data in enumerate(train, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            opt.zero_grad()

            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()

            opt.step()
            loss_arr.append(loss.item())
        
    loss_epochs_arr.append(loss.item())
    print('Epochs: %d/%d, Train acc: %0.2f, Test acc: %0.2f' % (epoch, max_epochs, evaluation(train), evaluation(test)))
    return net

def __main__():
    net = run()
    correct = 0 # initialize the number of correct predictions to zero
    total = 0 # initialize the total number of predictions to zero
    prediction = []
    labels_arr = [] 
    for i, data in enumerate(test, 0): # loop over all the batches in the test data
        inputs, labels = data 
        inputs, labels = inputs.to(device), labels.to(device)
        opt.zero_grad()
        outputs = net(inputs)
        total += labels.size(0) # increment the total number of predictions by the number of labels in the batch
        _, pred = torch.max(outputs.data, 1)
        # print(_, pred)
        correct += (pred == labels).sum() # compare the predicted labels with the actual labels, and increment the correct number of predictions by the number of matches
        pred = pred.numpy()
        labels = labels.numpy()
        prediction.extend(pred)
        labels_arr.extend(labels)
    print('Accuracy of the network on the 32 test images: %d %%' % (100 * correct / total)) # print the accuracy\ь
    cm = confusion_matrix(labels_arr, prediction)
    print(cm)
    print(classification_report(labels_arr, prediction))
    print(accuracy_score(labels_arr, prediction))


