import pickle
import requests
import torch.utils.data as data
import torch
from model import Net
from torchvision import datasets, transforms
import json
import numpy as np
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import base64
import websockets
import asyncio
import argparse
import socket
import select
import os
import threading
import time
import io
import random


HOSTS = ["127.0.0.1", "127.0.0.1", "127.0.0.1"]
PORT = 8000
# 65432 

# server_address = "localhost"
# server_port = 8000
lock = threading.Lock()

#Helper Function
def chunks(data, size=1024):
    """Yield successive chunks of the given size from the data."""
    for i in range(0, len(data), size):
        yield data[i:i + size]


class Client:
    def __init__(self,client_id):
        self.primary_server_id = 0
        self.client_id = client_id
        self.trainset = datasets.MNIST('data', download=True, train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
        self.testset = datasets.MNIST('data', download=True, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
        # Shuffle the data
        self.indices = np.arange(len(self.trainset))
        np.random.shuffle(self.indices)
        self.trainset.data = self.trainset.data[self.indices]
        self.trainset.targets = self.trainset.targets[self.indices]
        self.trainng_round = 0
        # Divide the data into n subsets
        self.batch_size = 32
        self.client_thresh = 5
        self.local_train_round = 0
        self.n = 5
        self.num_clients_connected = 0
        self.subset_size = len(self.trainset) // self.n
        self.trainsets = [torch.utils.data.Subset(self.trainset, range(i *self.subset_size, (i + 1) * self.subset_size)) for i in range(self.n)]

        # Reserve a portion of the data as a test set
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client_id = 1
    def get_trainset(self):    
        # Get the corresponding trainset
        self.trainset = self.trainsets[self.client_id]
        # Convert the trainset to a DataLoader
        trainloader = data_utils.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

        return trainloader

    def initialize_socket(self, tries=3):
        # If we've tried all three servers, return an error
        if tries == 0:
            return -1
        self.socket.close()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # If the current primary server doesn't work, try the next server (wrapping back around)
        try:
            self.socket.connect((HOSTS[self.primary_server_id], PORT + self.primary_server_id))
        except:
            self.primary_server_id = (self.primary_server_id + 1) % 3
            return self.initialize_socket(tries - 1)
        # connection_message = (CLIENT).to_bytes(1, byteorder = 'big')
        # self.socket.sendall(connection_message)
        time.sleep(0.05)
        return 0

    def send_heartbeat(self):
        while True:
            # Send a heartbeat message to the server
            time.sleep(1)
            try:
                self.sock.sendall('c'.encode())
                time.sleep(2)
                self.sock.sendall('heartbeat'.encode())
                data = str(self.sock.recv(1024).decode())
                with lock:
                    self.num_clients_connected = int(data[0])
                    self.trainng_round = int(data[1])
                    # print()
                    # print(self.num_clients_connected,self.trainng_round)
                # print(num_clients_connected,trainng_round)
            except:
                # try:
                    # 
                # except:
                self.initialize_socket()
                self.send_heartbeat()
                # continue


    def initialize_socket(self, tries=3):
        # If we've tried all three servers, return an error
        if tries == 0:
            return -1
        self.sock.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # If the current primary server doesn't work, try the next server (wrapping back around)
        try:
            self.sock.connect((HOSTS[self.primary_server_id], PORT + self.primary_server_id))
        except:
            self.primary_server_id = (self.primary_server_id + 1) % 3
            return self.initialize_socket(tries - 1)
        # connection_message = (CLIENT).to_bytes(1, byteorder = 'big')
        # self.socket.sendall(connection_message)
        # time.sleep(0.05)
        # return 0


    def send_weights(self):
        # client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # client_socket.connect(('localhost', 8000))
        # Send a message to request the weights
        
        self.sock.sendall('c'.encode())
        time.sleep(1)
        self.sock.sendall('here_are_weights'.encode())
        file_name = 'local_model_client_{}.pt'.format(self.client_id)
        # client_socket.sendall(f'local_model_client_{client_id}.pt'.encode())
        filesize = os.path.getsize(file_name)
        global_weights = torch.load(file_name)
        buffer = io.BytesIO()
        torch.save(global_weights, buffer)
        model_bytes = buffer.getvalue()
        with tqdm(total=filesize, unit='B', unit_scale=True, desc=f"Sending {'local_model'}") as pbar:
        # Send the model in chunks
            for chunk in chunks(model_bytes):
                self.sock.sendall(chunk)
                pbar.update(len(chunk))

        status = self.sock.recv(1024).decode()
        #Retry to send
        if status == "Error":
            self.send_weights()
        # client_socket.close()

    def get_weights(self):
        self.sock.sendall('c'.encode())
        time.sleep(1)
        self.sock.sendall('need_weights_pls'.encode())
        # rand_time = random.randint(30,60)
        # time.sleep(rand_time)
        # Open a file to write the weights to
        with open("local_model_client_{}.pt".format(self.client_id), 'wb') as f:
            # Receive the model in chunks and write them to the file
            while True:
                try:
                    chunk = self.sock.recv(1024)
                    if not chunk:
                        # Exit the loop when there are no more bytes to receive
                        break
                    f.write(chunk)
                except:
                    break
                try:
                    ready = select.select([self.sock], [], [], 0.1)
                    if not ready[0]:
                        # No bytes received within the timeout period, assume transfer is complete
                        break
                except:
                    break
        try:
            torch.load("local_model_client_{}.pt".format(self.client_id))
        except:
            self.get_weights()
        # Close the socket
        print("Loaded Weights")




    def train(self,round):
        local_model = Net()
        if round > 0:
            local_model.load_state_dict(torch.load('local_model_client_{}.pt'.format(self.client_id)))
        checkpoint_path = "local_model_client_{}.pt".format(self.client_id) 
        
        train_data = self.get_trainset()
        #Define Hyperparameters and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.5)
        epochs = 1
        for epoch in range(epochs):
            train_loss = 0.0
            with tqdm(train_data, unit="batch") as tepoch:
                for data, target in tepoch:
                    optimizer.zero_grad()
                    output = local_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()*data.size(0)
                    tepoch.set_postfix(loss=loss.item())
                    torch.save(local_model.state_dict(), checkpoint_path)
        
            train_loss = train_loss/len(train_data.dataset)
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))

    def training_loop(self):
        self.local_train_round = self.trainng_round
        while True:
            if self.local_train_round < self.trainng_round:
                self.local_train_round += 1
                print("Waiting for clients to connect...")
                while True:
                    print(f"Number of clients connected: {int(self.num_clients_connected)}")
                    if int(self.num_clients_connected) >= self.client_thresh:
                        break
                    time.sleep(4)
                print("Training round",self.trainng_round)
                print("Loading Weights")
                try:
                    self.get_weights()
                except:
                    try:
                        self.get_weights()
                    except:
                        self.initialize_socket()
                        self.get_weights()
                print("Starting Training")
                self.train(round=0)
                print("Sending Weights")

                try:
                    self.send_weights()
                except:
                    try:
                        self.send_weights()
                    except:
                        self.initialize_socket()
                        self.send_weights()
                print("Wating for Round to Finish")
                time.sleep(5)

    def main(self):
        # Start the send_message coroutine and the other_task coroutine concurrently
        print("Connecting to Server")
        # sock = connect()
        self.initialize_socket()
        print("Connected!")
        t1 = threading.Thread(target=self.training_loop)
        t2 = threading.Thread(target=self.send_heartbeat)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument('--id',type=int, help='client_id')
    args = parser.parse_args()

    c = Client(client_id = args.id)
    c.main()




















