import glob
import javaobj # For Reading Serialized Java Files
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from PIL import Image
import keyboard
import random

device = torch.device("cpu")
if torch.cuda.is_available():
    print(f'Switched to Cuda {torch.cuda_version}')
    device = torch.device("cuda")

train_data_start = 0; train_data_end = 25
test_data_start = 25; test_data_end = 30;



class CustomDatasetTrain(Dataset):

    def __init__(self):
        self.file_location = '../Java/Results50Moves/*.ser*'
        # Glob Batches into batch_locations
        self.batch_locations = []
        self.glob_batches()

        # TEMP FOR TESTING...
        self.batch_locations = self.batch_locations[train_data_start:train_data_end]

        print(f'Number Batches = {len(self.batch_locations)}')
        # Unserialize Java Objects
        self.unserialized_games = []
        self.populate_unserialized_games()
        # Organize Data
        self.master_data = []
        self.organize_data_into_tensors()
        # Get Normlaize Value
        self.get_mean_std()
        # Transforms
        self.transformations = transforms.Compose([
            # Need to Normalize -> Otherwise predictions are stuck at 0s and 1s
            transforms.Normalize((self.mean, ), (self.std, ))
        ])

    def __len__(self):
        return len(self.master_data)

    def __getitem__(self, idx):
        tensor, label = self.master_data[idx]
        label_tensor = torch.tensor([label], dtype=torch.int64)
        tensor = tensor.view(-1, 4, 3) # Set Dimensions to be 2d
        tensor = self.transformations(tensor)
        tensor = tensor.view(-1, 12) # Added to make work with solely linear net
        # print(tensor.shape)
        return tensor, label_tensor

    def glob_batches(self):
        fls = glob.glob(self.file_location)
        for file_name in fls:
            self.batch_locations.append(file_name)

    def load_data(self, location):
        with open(location, "rb") as fd:
            jobj = fd.read()
        return javaobj.loads(jobj)

    def populate_unserialized_games(self):
        batches_done = 0
        for batch_location in self.batch_locations:
            loaded_batch = self.load_data(batch_location)
            for i in range(len(loaded_batch)):
                self.unserialized_games.append(loaded_batch[i])
            batches_done += 1
            print(f"Finished Unserializing Batch {batches_done}")

    def organize_data_into_tensors(self):
        for i in range(len(self.unserialized_games)): # Iterating through Games
            for j in range(len(self.unserialized_games[i])): # Iterating Through Moves / result
                datapoint = []

                for k in range(len(self.unserialized_games[i][j])):
                    # Every once in while -> Error because len(bitboard array) == 1????? -> Need to figure out why
                    if len(self.unserialized_games[i][j][k])==12:
                        # Bitboards for Move
                        # Turn to Python Integers
                        python_integer_board = []
                        for number in self.unserialized_games[i][j][k]:
                            python_integer_board.append(int(number)) # Was int
                        # Tensor Bitboard
                        datapoint.append(torch.from_numpy(np.array(python_integer_board).astype("float64")))
                        # Label Integer
                        datapoint.append(int(self.unserialized_games[i][1][0][0]))
                        self.master_data.append((datapoint[0], datapoint[1]))
                        datapoint.clear()
            if i%250==0:
                print(f"Unpacking Game {i} to Tensor Train Data")
                print(f"Len Master Data = {len(self.master_data)}")
        print(f"Len Master Data = {len(self.master_data)}")

    def get_mean_std(self):

        # Use Starting Value for Chess Board -> May have to make this variable per individual trial which would be possible, but this could also mess things up...
        tensor, label = self.master_data[0]
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

class CustomDatasetTest(Dataset):

    def __init__(self):
        self.file_location = '../Java/Results50Moves/*.ser*'
        # Glob Batches into batch_locations
        self.batch_locations = []
        self.glob_batches()

        # TEMP FOR TESTING...
        self.batch_locations = self.batch_locations[test_data_start:test_data_end]

        print(f'Number Batches = {len(self.batch_locations)}')
        # Unserialize Java Objects
        self.unserialized_games = []
        self.populate_unserialized_games()
        # Organize Data
        self.master_data = []
        self.organize_data_into_tensors()
        # Get Normlaize Value
        self.get_mean_std()
        # Transforms
        self.transformations = transforms.Compose([
            # Need to Normalize -> Otherwise predictions are stuck at 0s and 1s
            transforms.Normalize((self.mean, ), (self.std, ))
        ])

    def __len__(self):
        return len(self.master_data)

    def __getitem__(self, idx):
        tensor, label = self.master_data[idx]
        label_tensor = torch.tensor([label], dtype=torch.int64)
        tensor = tensor.view(-1, 4, 3) # Set Dimensions to be 2d
        tensor = self.transformations(tensor)
        tensor = tensor.view(-1, 12) # Added to make work with solely linear net
        # print(tensor.shape)
        return tensor, label_tensor

    def glob_batches(self):
        fls = glob.glob(self.file_location)
        for file_name in fls:
            self.batch_locations.append(file_name)

    def load_data(self, location):
        with open(location, "rb") as fd:
            jobj = fd.read()
        return javaobj.loads(jobj)

    def populate_unserialized_games(self):
        batches_done = 0
        for batch_location in self.batch_locations:
            loaded_batch = self.load_data(batch_location)
            for i in range(len(loaded_batch)):
                self.unserialized_games.append(loaded_batch[i])
            batches_done += 1
            print(f"Finished Unserializing Batch {batches_done}")

    def organize_data_into_tensors(self):
        for i in range(len(self.unserialized_games)): # Iterating through Games
            for j in range(len(self.unserialized_games[i])): # Iterating Through Moves / result
                datapoint = []

                for k in range(len(self.unserialized_games[i][j])):
                    # Every once in while -> Error because len(bitboard array) == 1????? -> Need to figure out why
                    if len(self.unserialized_games[i][j][k])==12:
                        # Bitboards for Move
                        # Turn to Python Integers
                        python_integer_board = []
                        for number in self.unserialized_games[i][j][k]:
                            python_integer_board.append(int(number)) # Was int
                        # Tensor Bitboard
                        datapoint.append(torch.from_numpy(np.array(python_integer_board).astype("float64")))
                        # Label Integer
                        datapoint.append(int(self.unserialized_games[i][1][0][0]))
                        self.master_data.append((datapoint[0], datapoint[1]))
                        datapoint.clear()
            if i%250==0:
                print(f"Unpacking Game {i} to Tensor Train Data")
                print(f"Len Master Data = {len(self.master_data)}")
        print(f"Len Master Data = {len(self.master_data)}")

    def get_mean_std(self):

        # Use Starting Value for Chess Board -> May have to make this variable per individual trial which would be possible, but this could also mess things up...
        tensor, label = self.master_data[0]
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)


def print_data(data):
    for i in range(len(data)):
        print(f"========================== Game {i} ==============================")
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                print(f"Move {k}")
                print(data[i][j][k])
                # print(type(data[i][j][k]))
            print(f"Result = {data[i][1][0][0]}")


# class JudgementCNN(nn.Module):
#
#     def __init__(self, num_classes=3):
#         super(JudgementCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=1, padding=0)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=1, padding=0)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=1, padding=0)
#         # self.average_pool = nn.AvgPool2d((2, 2))
#         self.lin1 = nn.Linear(128*4*3, 128)
#         self.lin2 = nn.Linear(128, 64)
#         self.lin3 = nn.Linear(64, num_classes)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu((self.conv3(x)))
#         # x = self.average_pool(x)
#         # print(x.shape)
#         x = x.view(-1, 128*4*3)
#         x = F.relu(self.lin1(x))
#         x = F.relu(self.lin2(x))
#         x = self.lin3(x)
#         return F.softmax(x)


class JudgementCNN(nn.Module):

    def __init__(self):
        super(JudgementCNN, self).__init__()
        self.lin1 = nn.Linear(12, 144)
        self.lin2 = nn.Linear(144, 1728)
        self.lin3 = nn.Linear(1728, 10000)
        self.lin4 = nn.Linear(10000, 1728)
        self.lin5 = nn.Linear(1728, 3)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = F.relu(self.lin5(x))
        return F.softmax(x)


# Hyperparameters
epochs = 100
# learning_rate = 0.000001
learning_rate = 0.01
# batch_size = 128*6  # May actually want this to be something like 600 -> As one game is 300 moves right now...
batch_size = 36



# model = torchvision.models.resnet50(pretrained=False).to(device)
model = JudgementCNN()
model = model.to(device)
model.to(dtype=torch.float64)
loss_function = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# custom_data_train = CustomDatasetTrain()
# train_data = DataLoader(dataset=custom_data_train, batch_size=batch_size, shuffle=True)


custom_data_test = CustomDatasetTest()
test_data = DataLoader(dataset=custom_data_test, batch_size=batch_size, shuffle=True)


def train_judgement_cnn():
    for epoch in range(epochs):
        model.train()
        for batch_index, (data, label) in enumerate(train_data):
            data = data.to(device)
            label = label.to(device)

            # print(data)
            # quit()

            label = label.view(-1)

            data = data.view(-1, 12)
            # print(data.shape)

            result = model(data)

            # print(result)
            # print(label)

            loss = loss_function(result, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(result)
            # print(label)
            # quit()

            if batch_index % 50 == 0:
                print(f'Epoch = {epoch}/{epochs} Batch Index = {batch_index}/{round(len(custom_data_train)/batch_size)} Loss = {loss.item()}')
                # print(result)
                # print(label)
            if batch_index % 500 == 0:
                print(result)
                print(label)
        test()
        scheduler.step()

def test():
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_data:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # labels = labels.view(-1)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()
            print(correct)

    print('Accuracy of the network: %d %%' % (
            100 * correct / total))



if __name__ == '__main__':

    # test()

    # train_judgement_cnn()

    tensor = torch.tensor([[0.3, 0.4, 0.2],
                          [0.5, 0.1, 0.2]])
    labels = torch.tensor([2, 0])

    correct = (tensor == labels).sum().item()
    print(correct)

    # temp_tensor = torch.randn((1, 1, 4, 3))+100000
    # trans = transforms.Compose([
    #     transforms.Normalize((torch.mean(temp_tensor)), (torch.std(temp_tensor)))
    # ])
    # temp_tensor = trans(temp_tensor)
    # temp_tensor = temp_tensor.view(-1, 12)
    # print(temp_tensor)
    # print(model(temp_tensor))





    # temp_tensor = torch.randn((1, 1, 4, 3))+100000
    # trans = transforms.Compose([
    #     transforms.Normalize((torch.mean(temp_tensor)), (torch.std(temp_tensor)))
    # ])
    # print(temp_tensor)
    # temp_tensor = trans(temp_tensor)
    # print(temp_tensor)
    # print(model(temp_tensor))



