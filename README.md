# CS-262-Final - Distributed Federated Learning


<img width="547" alt="Screen Shot 2023-04-25 at 9 54 03 AM" src="https://user-images.githubusercontent.com/47306315/234298977-7372f606-f714-41c0-96ea-50ae9130aba1.png">



### How to use:

### First run
```
pip install -r requirements.txt
```

### In seperate terminals or machines run
```
python3 final_server.py --id 0 
python3 final_server.py --id 1   
python3 final_server.py --id 2   
```

### Then run the client machines (only 5 for our code)

```
python3 federated.py --id 0
python3 federated.py --id 1
python3 federated.py --id 2
python3 federated.py --id 3
python3 federated.py --id 4
```



If you would like the replicated servers to run on different machines, you must change the IP addresses in the global variable HOSTS in both federated.py and final_server.py, to match the IP addresses of the three servers.



### To run the tests 

```
python3 tests.py
```



### How our code works: 
#### Server 
- Server - Server Heartbeat function that transmits status and weights to slaves from master, this runs independently in its own thread. 
- Server - Client Heartbeat function that transmits the round number and number of connected clients to the clients periodically  in its own thread
- Fault tolerance - our server is replicated between 3 different machines utilizing the master - slave replication. Our implementation guides all traffic to the current master and has the slaves on standby in case the master fails.
#### Client  

Connects to server and wait for a certain number of clients to start training round, server then sends updated training round and client then download model weights in the form of a .pt file as this is the form that Pytoch uses. The client then checks if the weights have been download successfully (we found that by sending them over the wire can cause them to be corrupted to counteract this we do a retry until they can be loaded correctly). Once the client has the weights, it then begins training on its own local data for a set number of epochs (we choose 1 as this was quicker to test). Once the client has finished training it then sends its newly trained weights tot the server and waits for the round to increment.







Engineering Notebook:

4/22


Todo for Nibras:
- Add 2n + 1 fault tolerance to server to ensure it doesn't go down (3 different machines) and handle server data (1 file) and some variables like clients and training round [X]


Todo for Miles: 
- Aggergate Weights from servers end [X]
- Client thread to check what round and loops through rounds [X, X] 
- Server side check for n clients before starting rounds [X]
- Determine what to do when client drops [X]
- Evaluation on Server side of model [X]


Todo Together:
- Move some servers to AWS + we can combination of local and AWS or just on both of our local machines [X]
- Presentation [X]

4/21

Today we brainstormed some ways to hangled dropping Clients

- Handling the dropout of clients during federated learning is an important aspect of the design of a federated learning system. Here are a few approaches that can be used to handle client dropout:

- Replicate training data: One approach is to replicate the data of the dropped client to other clients. This can be done by either transferring the data to another client, or by increasing the weight of the data of the remaining clients in the aggregation.

- Resuming training from the last model state: Another approach is to save the last state of the global model before the dropout and resume training from that point when a new client joins. This approach is commonly used in federated learning systems that employ centralized parameter servers.

- Adaptive learning rate: Another approach is to use an adaptive learning rate that can adjust the model weights according to the number of active clients. In this approach, when a client drops out, the learning rate is adjusted to account for the reduction in the number of clients.

- Threshold for client dropout: Finally, a threshold can be set for the minimum number of active clients required for training to continue. If the number of active clients falls below this threshold, training can be paused until new clients join the system.
(We used this)


4/20

We worked on debugging some of the send weights functions as they are often corrupted, we decided to add a check on both ends and then to retry to send recuirsivley until they are recieved. This is not perfect although it is a simple fix for now. A perfect fix may be to retransmit the certain bits. We also worked on alot of the client end atleast getting one client to work. We also put together some test cases to verify this doesn't happend again!



4/19

Today we worked on the fault torerance of of backed this involed heartbeats from server this proved to be more difficult than expected as we had to use a different thread for this and had to lock the variables which was a pain to debug. 



4/18

Today we worked from switching the client heartbeat and the client in general from asyncio to threads as they seemed easier to work with and debug we also converted the client and server to be classes as they are much more robust for replication. 


4/15

Today we got a simple model training on the data a CNN on one client node, we realized that sending weights over the network will be an absolute pain although we think that by chunking the file up and sending parts of it will make it easier for us.







