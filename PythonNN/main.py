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
    device = torch.device("cuda")


class CustomDataset(Dataset):

    def __init__(self):
        self.file_location = '../Java/Results0-2/*.ser*'
        # Glob Batches into batch_locations
        self.batch_locations = []
        self.glob_batches()

        # TEMP FOR TESTING...
        self.batch_locations = self.batch_locations[0:]

        print(f'Number Batches = {len(self.batch_locations)}')
        # Unserialize Java Objects
        self.unserialized_games = []
        self.populate_unserialized_games()
        # Organize Data
        self.master_data = []
        self.organize_data_into_tensors()

    def __len__(self):
        return len(self.master_data)

    def __getitem__(self, idx):
        tensor, label = self.master_data[idx]
        label_tensor = torch.tensor([label])
        # print(tensor.shape)
        # print(label_tensor.shape)
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
                    if len(self.unserialized_games[i][j][k])==12: # Every once in while -> Error because len(bitboard array) == 1????? -> Need to figure out why
                        # Bitboards for Move
                        # Turn to Python Integers
                        python_integer_board = []
                        for number in self.unserialized_games[i][j][k]:
                            python_integer_board.append(int(number))
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




def print_data(data):
    for i in range(len(data)):
        print(f"========================== Game {i} ==============================")
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                print(f"Move {k}")
                print(data[i][j][k])
                # print(type(data[i][j][k]))
            print(f"Result = {data[i][1][0][0]}")


class JudgementCNN(nn.Module):

    def __init__(self):
        super(JudgementCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.lin1 = nn.Linear(in_features=512, out_features=128)
        self.lin2 = nn.Linear(in_features=128, out_features=3) # Cap, lc, stalemate
        self.dropoutConv = nn.Dropout(0.5)
        self.dropoutLin = nn.Dropout(0.4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # print(x.shape)
        x = x.view(-1, 512)
        x = F.relu(self.lin1(x))
        x = self.dropoutLin(x)
        x = F.relu(self.lin2(x))
        # print(x.shape)
        return x


# Hyperparameters
epochs = 10
# learning_rate = 0.000001
learning_rate = 0.00001
batch_size = 128*6  # May actually want this to be something like 600 -> As one game is 300 moves right now...

model = JudgementCNN()
model = model.to(device)
model.to(dtype=torch.float64)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

custom_data = CustomDataset()
train_data = DataLoader(dataset=custom_data, batch_size=batch_size, shuffle=True)

def train_judgement_cnn():
    for epoch in range(epochs):
        for batch_index, (data, label) in enumerate(train_data):
            data = data.to(device)
            label = label.to(device)
            label = label.view(-1)

            data.unsqueeze_(1)

            result = model(data)

            # print(result)
            # print(label)

            loss = loss_function(result, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 10 == 0:
                print(f'Epoch = {epoch}/{epochs} Batch Index = {batch_index}/{round(len(custom_data)/batch_size)} Loss = {loss.item()}')

if __name__ == '__main__':
    train_judgement_cnn()
    temp_tensor = torch.randn((3, 1, 12))



