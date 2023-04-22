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

# Define constants
HEARTBEAT_INTERVAL = 5.0

training_round = 1

# Initialize clients dictionary
clients = {}

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
        # Start a new thread to handle the client connection
        threading.Thread(target=handle_client_connection, args=(connection, client_address)).start()

# Define function to handle incoming messages from clients
def handle_client_connection(connection, client_address):
    try:
        # Receive data from the client
        data = b''
        while True:
            chunk = connection.recv(1024)
            if not chunk:
                break
            data += chunk


            if chunk.decode() == 'num_clients_connected':
                time.sleep(2)
                connection.sendall(str(len(clients)).encode())
            # print(data.decode())
            # Handle incoming messages from clients
            if data.decode() == 'need_weights_pls':
                    # Replace with generic filename
                # Replace with generic filename
                filesize = os.path.getsize('global_model.pt')
                global_weights = torch.load('global_model.pt')
                buffer = io.BytesIO()
                torch.save(global_weights, buffer)
                model_bytes = buffer.getvalue()
                with tqdm(total=filesize, unit='B', unit_scale=True, desc=f"Sending {'global_model'} to  {client_address}") as pbar:                        # Send the model in chunks
                    for chunk in chunks(model_bytes):
                        connection.sendall(chunk)
                        pbar.update(len(chunk))

            elif data.decode() == 'here_are_weights':
                # file_name = connection.recv(1024)
                # print(file_name)
                # Open a file to write the weights to
                with open("clinet_n_round_n.pt", 'wb') as f:
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
                print("Loaded Weights")
                # connection.close()

            elif chunk.decode() == 'heartbeat':
                # Update the heartbeat for this client
                clients[client_address] = time.time()
                print(time.time())
                # Send a response to the client
                connection.sendall(str(training_round).encode())
                # print("sendning")
                # print(clients)

    except Exception as e:
        traceback.print_exc()
    # except:
        # Clean up the connection
        print('connection closed')
        connection.close()



def check_heartbeats():
    while True:
        # Check the heartbeat time for each client
        now = time.time()
        for client_address in clients.copy():
            last_heartbeat_time = clients[client_address]
            print(last_heartbeat_time)
            if now - last_heartbeat_time > HEARTBEAT_INTERVAL:
                # print(now - last_heartbeat_time)
                # The client has not sent a heartbeat message recently, remove it from the list
                # connection.close()
                del clients[client_address]
                print(f"Removed inactive client: {client_address}")

        # Wait for the next check interval
        time.sleep(HEARTBEAT_INTERVAL)

if __name__ == '__main__':
     # Start the server in the main thread
    threading.Thread(target=server).start()

    # Start hearbeat check in its own thread
    threading.Thread(target=check_heartbeats).start()
   
