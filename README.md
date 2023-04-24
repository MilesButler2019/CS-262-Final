# CS-262-Final


How to use:

In seperate terminals or machines run
`
python3 final_server.py --id 0
python3 final_server.py --id 1
python3 final_server.py --id 2
`

Then run the client machines (only 5 for our code)

`
python3 federated.py --id 0
python3 federated.py --id 1
python3 federated.py --id 2
python3 federated.py --id 3
python3 federated.py --id 4
`



If you would like the replicated servers to run on different machines, you must change the IP addresses in the global variable HOSTS in both federated.py and final_server.py, to match the IP addresses of the three servers.



To run the tests 

`
python3 tests.py
`




Engineering Notebook:

4/22


Todo for Nibras:
- Add 2n + 1 fault tolerance to server to ensure it doesn't go down (3 different machines) and handle server data (1 file) and some variables like clients and training round [X]


Todo for Miles: 
- Aggergate Weights from servers end [X]
- Client thread to check what round and loops through rounds [X, X] 
- Server side check for n clients before starting rounds [X]
- Determine what to do when client drops []
- Evaluation on Server side of model


Todo Together:
- Move some servers to AWS + we can combination of local and AWS or just on both of our local machines
- Presentation

4/21









