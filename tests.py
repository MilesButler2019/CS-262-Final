from final_server import Server
from federated import Client
import os
def unit_tests():
    server = Server(0).main()
    c1 = Client(0).main()
    c2 = Client(1).main()
    c3 = Client(2).main()
    c4 = Client(3).main()
    c5 = Client(4).main()



    # Test: Successfully download weights
    current_directory = os.getcwd()
    c1.get_weights()
    c2.get_weights()
    c3.get_weights()
    c4.get_weights()
    c5.get_weights()

    assert('local_model_clinet_0.pt' in current_directory)
    assert('local_model_clinet_1.pt' in current_directory)
    assert('local_model_clinet_2.pt' in current_directory)
    assert('local_model_clinet_3.pt' in current_directory)
    assert('local_model_clinet_4.pt' in current_directory)

    #Test Send weights
    
    c1.send_weights()
    c2.send_weights()
    c3.send_weights()
    c4.send_weights()
    c5.send_weights()

    files = []
    for f in os.listdir(current_directory):
        if f.endswith(f'_1.pt') and f.startswith('clinet'):
            files.append(f)

    assert(len(files) == 5)


    #Test aggergate weights
    server.average_weights()
    assert('new_global_model_1.pt' in current_directory)





    # assert(server.users["yejoo"].username == "yejoo")
    # assert(server.users["yejoo"].password == "0104")

    # # Test: Username must be unique
    # server.create_account("yejoo", "0123")
    # assert(server.users["yejoo"].username == "yejoo")
    # assert(server.users["yejoo"].password == "0104")

    # # Test: Listing Accounts
    # server.create_account("idk", "sth")
    # server.create_account("yej", "password")
    # server.create_account("middle", "mid")
    # assert(set(server.list_accounts("*")) == set(["yejoo", "yej", "idk", "middle"]))
    # assert(set(server.list_accounts("ye*")) == set(["yejoo", "yej"]))
    # assert(set(server.list_accounts("*oo")) == set(["yejoo"]))
    # assert(set(server.list_accounts("*d*")) == set(["idk", "middle"]))

    # # Test: Login only logs in users with correct passwords
    # assert(server.login("yejoo", "0123") == 1)
    # assert(server.login("yejoo", "0104") == 0)
    # assert(server.login("idk", "sth") == 0)
    # assert(server.login("dklfjsk;", "sdl k") == 2)
    # assert(server.login("middle", "idk") == 1)

    # # Test: sending message queues message
    # server.send_message("yejoo", "idk", "secrete")
    # server.send_message("yejoo", "idk", "dfjopadd")
    # server.send_message("idk", "yejoo", "dofjsoi")
    # sent_messages_idk = server.users["idk"].messages
    # assert(len(sent_messages_idk) == 2)
    # assert(sent_messages_idk[0].sender == "yejoo")
    # assert(sent_messages_idk[0].receiver == "idk")
    # assert(sent_messages_idk[0].message == "secrete")
    # assert(sent_messages_idk[1].sender == "yejoo")
    # assert(sent_messages_idk[1].receiver == "idk")
    # assert(sent_messages_idk[1].message == "dfjopadd")
    # sent_messages_yejoo = server.users["yejoo"].messages
    # assert(len(sent_messages_yejoo) == 1)
    # assert(sent_messages_yejoo[0].sender == "idk")
    # assert(sent_messages_yejoo[0].receiver == "yejoo")
    # assert(sent_messages_yejoo[0].message == "dofjsoi")

    # # Test: receiving message looks at queued message
    # assert(server.receive_messages("yejoo") == "From idk: dofjsoi\n")
    # assert(server.receive_messages("idk") == "From yejoo: secrete\nFrom yejoo: dfjopadd\n")

    # # Test: messages are received just once.
    # assert(server.receive_messages("yejoo") == "")
    # assert(server.receive_messages("idk") == "")

    # # Test: deleted account returns messages and gets rid of user
    # server.send_message("yejoo", "idk", "more")
    # server.send_message("yejoo", "idk", "more2")
    # server.delete_account("idk")
    # assert("idk" not in server.users)
    # assert(server.receive_messages("yejoo") == "")
# torch.load('local_model_client_4.pt')
# # print("done")
# def aggergate_weights():
#     model_files = []
#     avg_model = None
#     dir_path = "/Users/mbutler/Documents/Winter_2023/CS262/CS626-Final"
#     for f in os.listdir(dir_path):
#         if f.endswith('.pt') and f.startswith('clinet'):
#             model_files.append(f)
#     # print(model_files)
#     models = []
#     # Load each model file and append it to the list of models
#     for model_file in model_files:
#         model = torch.load(os.path.join(model_file), map_location=torch.device('cpu'))
#         models.append(model)
    # print(models)
    
#     # Get the number of models loaded
#     num_models = len(models)
#     # Iterate over each loaded model and average its weights with the existing average model
#     # main = {}
#     print(models[0]['fc2.bias'])
#     for model in models[1:]:
#         for key in model:
#             models[0][key] += model[key]
#     for key in models[0]:
#         models[0][key] /= num_models
#     # print(models[0][key])
#     torch.save(models[0],'new_global_model.pt')

#         # print(model.keys())
#         # Net().load_state_dict(state_dict = model)
    



# import socket
# HOSTS = ["127.0.0.1", "54.147.154.94", "127.0.0.1"]
# PORT = 5004 
# primary_server_id = 1
# # host = socket.gethostbyaddr('aws.ec2.public.ip')[0]
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         # If the current primary server doesn't work, try the next server (wrapping back around)

# print(HOSTS[primary_server_id])
# print(PORT)
# sock.connect((HOSTS[primary_server_id], PORT+primary_server_id))
# aggergate_weights()


# def initialize_socket(tries=3):
#         # If we've tried all three servers, return an error
#         # if tries == 0:
#             # return -1
#         # self.sock.close()
#         self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         # If the current primary server doesn't work, try the next server (wrapping back around)
#         try:
#             self.sock.connect((HOSTS[self.primary_server_id], PORT + self.primary_server_id))
#         except:
#             self.primary_server_id = (self.primary_server_id + 1) % 3
#             return self.initialize_socket(tries - 1)