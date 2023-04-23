import socket
import os
import io
import select
import threading
import time
from tqdm import tqdm
import torch
import traceback
import argparse

# Define constants
HEARTBEAT_INTERVAL = 30

training_round = 1
lock = threading.Lock()

# Initialize clients dictionary
clients = {}
client_connections = {}
num_weights_recieved = 0

LEADER = 0
FOLLOWER = 1
tracker = 0
# HOSTS = {8000: "127.0.0.1", 8001: "127.0.0.1", 8002: "127.0.0.1"}
HOSTS = {65432: "127.0.0.1", 65433: "127.0.0.1", 65434: "127.0.0.1"}
# Port used is PORT + ID
PORT = 65432
# Define helper function to send data in chunks
def chunks(data):
    chunk_size = 1024
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]

# Define server function to handle incoming client connections
class Server:
    def __init__(self,id):
        self.id = id
        self.master_id = 0
        self.port = PORT + id
        self.host = HOSTS[self.port]
        self.HEARTBEAT_INTERVAL = 30
        self.training_round = 1
        self.lock = threading.Lock()

        # Initialize clients dictionary
        self.clients = {}
        self.client_connections = {}
        self.num_weights_recieved = 0
        self.tracker = 0
        self.other_servers = {}

        for i in range(self.id):
            self.other_servers[self.master_id + PORT + i] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
       
    def start(self):
        # Bind the socket to a specific address and port
        self.server_address = ('localhost', self.port)
        self.sock.bind(self.server_address)
        # Listen for incoming connections
        self.sock.listen(5)
        print(self.other_servers)
        self.connect_to_servers()
        while True:
            # Wait for a connection
            print('waiting for a connection')
            self.connection, self.client_address = self.sock.accept()
            print('connection from', self.client_address)
            clients[self.client_address] = time.time()
            client_connections[self.client_address] = self.connection
            # Start a new thread to handle the client connection
            threading.Thread(target=self.handle_connection, args=(self.connection, self.client_address)).start()



    def connect_to_servers(self):
        for port in self.other_servers:
            self.other_servers[port].connect((HOSTS[port], port))
            threading.Thread(target=self.handle_server_connection, args=(self.other_servers[port], port))
            # self.other_servers[port].sendall("stest".encode())

    def handle_connection(self,conn,client_address):
        data = conn.recv(1024).decode()
        server_or_client = str(data)[0]
        if server_or_client == 's':
            # port = int.from_bytes(data[1:5],)
            self.handle_server_connection(conn, data[1:])
        elif server_or_client == 'c':
            print ("connected to client")
            self.handle_client_connection(conn,client_address)


    def handle_server_connection(self,connection,port):
        print("Connected to " + str(port))
        self.other_servers[port] = connection
        data = connection.recv(1024).decode()
        if data[0] == LEADER:
            self.master_id = int(str(data[1:5])) - PORT    
      
                


    def server_hearbeat(self):
        #Check if master
        time.sleep(5)
        while True:
            print(self.id,self.master_id)
            message = str((LEADER if self.id == self.master_id else FOLLOWER))

                    # Close the socket
                
            for port in self.other_servers:
                
                try:
                    self.other_servers[port].sendall(f's{port}'.encode())
                    time.sleep(1)
                    self.other_servers[port].sendall(message.encode())
                   
              
                except:

                    # If the server is down then close the connection
                    self.other_servers[port].close()
                    self.other_servers[port] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    # If the master has gone down, update the master id

                    if int(port) - int(PORT) == self.master_id:
                        self.master_id = (self.master_id + 1) % 3
                    # Keep trying to reconnect, this allows for down servers to come back up seamlessly
                    try:
                        self.other_servers[port].connect((HOSTS[port], port))
                        threading.Thread(target=self.handle_server_connection, args=(self.other_servers[port], port)).start()
                        # connection_message = (SERVER).to_bytes(1, byteorder = 'big') + (self.port).to_bytes(4, byteorder = 'big')
                        self.other_servers[port].sendall(f's{port}'.encode())
                    except:
                        pass
            time.sleep(5)

# Define function to handle incoming messages from clients
    def handle_client_connection(self,connection, client_address):

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
                    filesize = os.path.getsize('new_global_model_1.pt')
                    global_weights = torch.load('new_global_model_1.pt')
                    buffer = io.BytesIO()
                    torch.save(global_weights, buffer)
                    model_bytes = buffer.getvalue()
                    with tqdm(total=filesize, unit='B', unit_scale=True, desc=f"Sending {'global_model'} to  {client_address}") as pbar:                        # Send the model in chunks
                        for chunk in chunks(model_bytes):
                            connection.sendall(chunk)
                            pbar.update(len(chunk))

                elif chunk.decode(errors='ignore') == 'here_are_weights':
                    print("Working")
                    # globaself.num_weights_recieved;
                    with lock:
                        self.num_weights_recieved += 1
                    # Open a file to write the weights to
                    with open(f"clinet_{client_address[1]}_round_{self.training_round}.pt", 'wb') as f:
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
                        torch.load(f"clinet_{client_address[1]}_round_{self.training_round}.pt")
                        connection.sendall("Recieved".encode())
                        print("sucsess")
                    except:
                        connection.sendall("Error".encode())

                    print("Loaded Weights")
                    connection.close()

                elif chunk.decode(errors='ignore') == 'heartbeat':
                    # Update the heartbeat for this client
                    self.clients[client_address] = time.time()
                    # Send a response to the client
                    connection.sendall(str(len(self.clients)).encode())
                    connection.sendall(str(self.training_round).encode())

        except Exception as e:
            traceback.print_exc()
        # except:
            # Clean up the connection
            print('connection closed')
            connection.close()


    def aggergate_weights(self):
        model_files = []
        avg_model = None
        dir_path = "/Users/mbutler/Documents/Winter_2023/CS262/CS626-Final"
        for f in os.listdir(dir_path):
            if f.endswith(f'_{self.training_round}.pt') and f.startswith('clinet'):
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



    def check_heartbeats(self):
        while True:
            # Check the heartbeat time for each client
            now = time.time()
            for client_address in self.clients.copy():
                last_heartbeat_time = self.clients[client_address]
                if now - last_heartbeat_time > HEARTBEAT_INTERVAL:
                    # print(now - last_heartbeat_time)
                    # The client has not sent a heartbeat message recently, remove it from the list
                    self.client_connections[client_address].close()
                    del self.clients[client_address]
                    del self.client_connections[client_address]
                    print(f"Removed inactive client: {client_address}")

            # Wait for the next check interval
            time.sleep(HEARTBEAT_INTERVAL)

    def update_round(self):
        #Give clients a chance to connect
        time.sleep(10)
        while True:
            if self.num_weights_recieved >= len(self.clients):
                # global self.training_round
                print("Averaging weights")
                self.aggergate_weights()
                with lock:
                    self.training_round += 1
                break
            time.sleep(10)

    # def run(self):


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id',type=int, help='server id')
    args = parser.parse_args()
     # Start the server in the main thread
    s = Server(args.id)
    threading.Thread(target=s.start,args=()).start()
    # # Start hearbeat check in its own thread
    threading.Thread(target=s.check_heartbeats).start()
    threading.Thread(target=s.server_hearbeat).start()
     # Start update round in its own thread
    # threading.Thread(target=s.update_round).start()
