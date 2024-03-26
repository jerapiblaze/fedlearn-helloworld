import flwr as fl
from client import client_fn, N_CLIENTS, CLIENT_RESOURCES

# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=10,  # Never sample less than 10 clients for training
    min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    min_available_clients=10,  # Wait until all 10 clients are available
)

############################
# MAIN
############################
def Main():
    # fl.simulation.start_simulation(
    #     client_fn=client_fn,
    #     num_clients=N_CLIENTS,
    #     config=fl.server.ServerConfig(num_rounds=5),
    #     strategy=strategy,
    #     client_resources=CLIENT_RESOURCES,
    # )
    fl.server.start_server(server_address="localhost:8080", config=fl.server.ServerConfig(num_rounds=3))

if __name__=="__main__":
    Main()