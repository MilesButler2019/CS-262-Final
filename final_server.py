import socket
import os
import io
import select
import threading
import time
import asyncio
from tqdm import tqdm
import torch
import traceback
import struct

# Define constants
HEARTBEAT_INTERVAL = 30

training_round = 1
lock = threading.Lock()

# Initialize clients dictionary
clients = {}
client_connections = {}
num_weights_recieved = 0
tracker = 0
# Define helper function to send data in chunks
def chunks(data):
    chunk_size = 1024
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]

# Define server function to handle incoming client connections
def server():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Bind the socket to a specific address and port
    server_address = ('localhost', 8000)
    sock.bind(server_address)
    # Listen for incoming connections
    sock.listen(5)

    while True:
        # Wait for a connection
        print('waiting for a connection')
        connection, client_address = sock.accept()
        print('connection from', client_address)
        clients[client_address] = time.time()
        client_connections[client_address] = connection
        # Start a new thread to handle the client connection
        threading.Thread(target=handle_client_connection, args=(connection, client_address)).start()




# Define function to handle incoming messages from clients
def handle_client_connection(connection, client_address):


    try:
        # Receive data from the client
        data = b''
        while True:
            chunk = connection.recv(1024)
            # print(chunk)
            # break
            if not chunk:
                break
            data += chunk

            # Handle incoming messages from clients
            if chunk.decode(errors='ignore') == 'need_weights_pls':
                print("serving weights")
                    # Replace with generic filename
                # Replace with generic filename
                filesize = os.path.getsize('new_global_model.pt')
                global_weights = torch.load('new_global_model.pt')
                buffer = io.BytesIO()
                torch.save(global_weights, buffer)
                model_bytes = buffer.getvalue()
                with tqdm(total=filesize, unit='B', unit_scale=True, desc=f"Sending {'global_model'} to  {client_address}") as pbar:                        # Send the model in chunks
                    for chunk in chunks(model_bytes):
                        connection.sendall(chunk)
                        pbar.update(len(chunk))

            elif chunk.decode(errors='ignore') == 'here_are_weights':
                print("Working")
                global num_weights_recieved
                with lock:
                    num_weights_recieved += 1
                # Open a file to write the weights to
                with open(f"clinet_{client_address[1]}_round_{training_round}.pt", 'wb') as f:
                    # Receive the model in chunks and write them to the file
                    while True:
                        chunk = connection.recv(1024)
                        if not chunk:
                            # Exit the loop when there are no more bytes to receive
                            break
                        f.write(chunk)
                        # print(chunk)
                        ready = select.select([connection], [], [], 0.1)
                        if not ready[0]:
                            # No bytes received within the timeout period, assume transfer is complete
                            break
                # Close the socket
                #Verify weights can be read

                try:
                    torch.load(f"clinet_{client_address[1]}_round_{training_round}.pt")
                    connection.sendall("Recieved".encode())
                    print("sucsess")
                except:
                    connection.sendall("Error".encode())

                print("Loaded Weights")
                connection.close()

            elif chunk.decode(errors='ignore') == 'heartbeat':
                # Update the heartbeat for this client
                clients[client_address] = time.time()
                # Send a response to the client
                connection.sendall(str(len(clients)).encode())
                connection.sendall(str(training_round).encode())

    except Exception as e:
        traceback.print_exc()
    # except:
        # Clean up the connection
        print('connection closed')
        connection.close()


def aggergate_weights():
    model_files = []
    avg_model = None
    dir_path = "/Users/mbutler/Documents/Winter_2023/CS262/CS626-Final"
    for f in os.listdir(dir_path):
        if f.endswith('.pt') and f.startswith('clinet'):
            model_files.append(f)
    # print(model_files)
    models = []
    # Load each model file and append it to the list of models
    for model_file in model_files:
        model = torch.load(os.path.join(model_file), map_location=torch.device('cpu'))
        models.append(model)

    # Get the number of models loaded
    num_models = len(models)
    # Iterate over each loaded model and average its weights with the existing average model
    # main = {}
    for model in models[1:]:
        for key in model:
            models[0][key] += model[key]
    for key in models[0]:
        models[0][key] /= num_models
    # print(models[0][key])
    torch.save(models[0],'new_global_model_1.pt')



def check_heartbeats():
    while True:
        # Check the heartbeat time for each client
        now = time.time()
        for client_address in clients.copy():
            last_heartbeat_time = clients[client_address]
            if now - last_heartbeat_time > HEARTBEAT_INTERVAL:
                # print(now - last_heartbeat_time)
                # The client has not sent a heartbeat message recently, remove it from the list
                client_connections[client_address].close()
                del clients[client_address]
                del client_connections[client_address]
                print(f"Removed inactive client: {client_address}")

        # Wait for the next check interval
        time.sleep(HEARTBEAT_INTERVAL)


def update_round():
    #Give clients a chance to connect
    time.sleep(10)
    while True:
        if num_weights_recieved >= len(clients):
            global training_round
            print("Averaging weights")
            aggergate_weights()
            with lock:
                training_round += 1
            break
        time.sleep(10)

if __name__ == '__main__':
     # Start the server in the main thread
    threading.Thread(target=server).start()
    # # Start hearbeat check in its own thread
    threading.Thread(target=check_heartbeats).start()
     # Start update round in its own thread
    threading.Thread(target=update_round).start()
   
