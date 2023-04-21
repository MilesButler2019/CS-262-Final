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
import time
import io

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
n = 5
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

def get_global_model():
    try:
        response = requests.get('http://34.229.220.141:5000/get_global_model')
    except:
        print("cant load model")
    global_model_params = response.json()

    global_model_params = {k.replace('.', '_'): v for k, v in global_model_params.items()}

    for name, param in global_model_params.items():
        setattr(local_model, name, nn.Parameter(torch.tensor(param)))

    # # Print the model parameters to verify that they have been set correctly
    for name, param in local_model.named_parameters():
        print(name, param.shape)










server_address = "localhost"
server_port = 8000



async def send_heartbeat():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the server's address and port
    sock.connect(('localhost', 8000))

    while True:
        # Send a heartbeat message to the server
        time.sleep(1)
        # print("hi")
        sock.sendall('heartbeat'.encode())

        sock.settimeout(1)  # Set a timeout of 1 second
        try:
            data = sock.recv(1024)
        except socket.timeout:
            # If no data is received within the timeout period, continue with the next iteration
            continue
        trainng_round = data.decode()
        print(trainng_round)
        print("doing_work")


async def chunks(data, size=1024):
    """Yield successive chunks of the given size from the data."""
    for i in range(0, len(data), size):
        yield data[i:i + size]

async def send_weights(client_id):
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
        async for chunk in chunks(model_bytes):
            client_socket.sendall(chunk)
            pbar.update(len(chunk))

async def get_weights(client_id):
    # Create a TCP socket and connect to the server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 8000))
    # Send a message to request the weights
    client_socket.sendall('need_weights_pls'.encode())
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
    # Close the socket
    print("Loaded Weights")
    client_socket.close()




async def train(client_id,round):
    local_model = Net()
    if round > 0:
        local_model.load_state_dict(torch.load('local_model_client_{}.pt'.format(client_id)))
    checkpoint_path = "local_model_client_{}.pt".format(client_id) 
    
    train_data = get_trainset(client_id=client_id)
    
    #Define Hyperparameters and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.5)
    epochs = 5
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




async def training_loop(client_id):
    print("Training round",trainng_round)
    print("Loading Weights")
    await get_weights(client_id)
    print("Starting Training")
    await train(client_id = client_id,round=1)
    print("Sending Weights")
    await send_weights(client_id=client_id)
    print("Wating for Round to Finish")





async def main(client_id):
    # Start the send_message coroutine and the other_task coroutine concurrently
    # await training_loop(client_id=client_id)
    training_loop_l = asyncio.create_task(training_loop(client_id))
    heartbeat_task = asyncio.create_task(send_heartbeat())
    # await send_heartbeat()
    # Wait for both tasks to complete
    await asyncio.gather(training_loop_l,heartbeat_task)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id',type=int, help='client_id')
    args = parser.parse_args()
    asyncio.run(main(client_id = args.id))
# asyncio.run(send_message())

















