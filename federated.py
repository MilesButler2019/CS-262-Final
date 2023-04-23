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

# Define the URL of the Flask server
server_url = 'http://34.229.220.141:5000'





trainset = datasets.MNIST('data', download=True, train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))
testset = datasets.MNIST('data', download=True, train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))

# Shuffle the data
indices = np.arange(len(trainset))
np.random.shuffle(indices)
trainset.data = trainset.data[indices]
trainset.targets = trainset.targets[indices]
trainng_round = 0
# Divide the data into n subsets
batch_size = 32
client_thresh = 5
local_train_round = 0
n = 5
num_clients_connected = 0
subset_size = len(trainset) // n
trainsets = [torch.utils.data.Subset(trainset, range(i * subset_size, (i + 1) * subset_size)) for i in range(n)]

# Reserve a portion of the data as a test set
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
# client_id = 1
def get_trainset(client_id):    

    # Get the corresponding trainset
    trainset = trainsets[client_id]

    # Convert the trainset to a DataLoader
    trainloader = data_utils.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    return trainloader






server_address = "localhost"
server_port = 8000


def connect():
            # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the server's address and port
    sock.connect(('localhost', 8000))
    return sock

lock = threading.Lock()

def send_heartbeat(sock):

    global num_clients_connected,trainng_round

    while True:
        # Send a heartbeat message to the server
        time.sleep(3)
        try:
            sock.sendall('heartbeat'.encode())
            data = str(sock.recv(1024).decode())

            with lock:
                num_clients_connected = int(data[0])
                trainng_round = int(data[1])
            # print(num_clients_connected,trainng_round)
        except:
            continue



def chunks(data, size=1024):
    """Yield successive chunks of the given size from the data."""
    for i in range(0, len(data), size):
        yield data[i:i + size]

def get_num_clients(client_socket):
    client_socket.sendall('num_clients_connected'.encode())
    num_clients = client_socket.recv(1024).decode()
    return num_clients


def send_weights(client_id):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 8000))
    # Send a message to request the weights
    client_socket.sendall('here_are_weights'.encode())
    file_name = 'local_model_client_{}.pt'.format(client_id)
    # client_socket.sendall(f'local_model_client_{client_id}.pt'.encode())
    filesize = os.path.getsize(file_name)
    global_weights = torch.load(file_name)
    buffer = io.BytesIO()
    torch.save(global_weights, buffer)
    model_bytes = buffer.getvalue()
    with tqdm(total=filesize, unit='B', unit_scale=True, desc=f"Sending {'local_model'}") as pbar:
    # Send the model in chunks
        for chunk in chunks(model_bytes):
            client_socket.sendall(chunk)
            pbar.update(len(chunk))

    status = client_socket.recv(1024).decode()
    #Retry to send
    if status == "Error":
        send_weights(client_id=client_id)
    client_socket.close()

def get_weights(client_id,client_socket):


    client_socket.sendall('need_weights_pls'.encode())
    # rand_time = random.randint(30,60)
    # time.sleep(rand_time)
    # Open a file to write the weights to
    with open("local_model_client_{}.pt".format(client_id), 'wb') as f:
        # Receive the model in chunks and write them to the file
        while True:
            chunk = client_socket.recv(1024)
            if not chunk:
                # Exit the loop when there are no more bytes to receive
                break
            f.write(chunk)
            ready = select.select([client_socket], [], [], 0.1)
            if not ready[0]:
                # No bytes received within the timeout period, assume transfer is complete
                break
    try:
        torch.load("local_model_client_{}.pt".format(client_id))
    except:
        get_weights(client_id,client_socket)
    # Close the socket
    print("Loaded Weights")




def train(client_id,round):
    local_model = Net()
    if round > 0:
        local_model.load_state_dict(torch.load('local_model_client_{}.pt'.format(client_id)))
    checkpoint_path = "local_model_client_{}.pt".format(client_id) 
    
    train_data = get_trainset(client_id=client_id)
    
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




def training_loop(client_id,sock):
    global local_train_round
    local_train_round = trainng_round
    while True:
        if local_train_round < trainng_round:
            print("Waiting for clients to connect...")
            while True:
                print(f"Number of clients connected: {int(num_clients_connected)}")
                if int(num_clients_connected) >= client_thresh - 1:
                    break
                time.sleep(10)
            print("Training round",trainng_round)
            print("Loading Weights")
            get_weights(client_id,sock)
            print("Starting Training")
            train(client_id = client_id,round=0)
            print("Sending Weights")
            send_weights(client_id)
            print("Wating for Round to Finish")
            local_train_round += 1
            time.sleep(10)



def main(client_id):
    # Start the send_message coroutine and the other_task coroutine concurrently
    print("Connecting to Server")
    sock = connect()
    print("Connected!")
    

    t1 = threading.Thread(target=training_loop, args=(client_id, sock))
    t2 = threading.Thread(target=send_heartbeat, args=(sock,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument('--id',type=int, help='client_id')
    args = parser.parse_args()
    main(client_id = args.id)
    # asyncio.run(main(client_id = args.id))
# asyncio.run(send_message())

















