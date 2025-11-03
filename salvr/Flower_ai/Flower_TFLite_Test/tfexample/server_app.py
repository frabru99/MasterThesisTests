"""tftest: A flower / tensorflowlite app"""

from typing import List, Tuple
from flwr.app import ArrayRecord, MetricRecord, ConfigRecord, Context
from flwr.serverapp import ServerApp, Grid
from flwr.serverapp.strategy import FedAvg

# Lets notice that server doesn't load any data
from tfexample.task import load_model

server_app = ServerApp()

@server_app.main()
def server_main_fn(grid: Grid, context: Context) -> None:

    """
    Main entry point for the ServerApp
    """

    # Load config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]

    # Load initial model
    model = load_model()
    arrays = ArrayRecord(model.get_weights())

    # Define the strategy
    strategy = FedAvg(
        fraction_train=fraction_train    
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,    
    )

    # Save the final model
    ndarrays = result.arrays.to_numpy_ndarrays()
    final_model_name = "final_model.keras"
    print(f"Saving final model to disk as: {final_model_name}...")
    model.set_weights(ndarrays)
    model.save(final_model_name)

