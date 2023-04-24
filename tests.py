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




