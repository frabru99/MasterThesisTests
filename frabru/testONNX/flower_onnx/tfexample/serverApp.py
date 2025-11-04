from flwr.app import Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from onnxruntime.training.api import CheckpointState
from flwr.app import ArrayRecord, MetricRecord, ConfigRecord, RecordDict


server_app = ServerApp()


"""
Here, or in other occurrences we've to export the model creating the .onnx file. 
After, we've to create artifacts in order to load the checkpoint file ans save the state_dict()
useful for array variable. 
"""

"""
Config and Metrics structure.
"""
BATCH_SIZE = 4 #Batch Size
NUM_CLASSES = 10 #Same as Classes of the Dataset

config = ConfigRecord(
        {"batch_size": BATCH_SIZE, "use_augmentation": True, "data-path": "./data", "lr": 0.1}
)

#metrics = MetricRecord({"accuracy": 0.9, "losses": [0.1, 0.001], "perplexity": 2.31})


@server_app.main()
def main(grid: Grid, context: Context) -> None:
    #Loading the context

    num_rounds = context.run_config["num-server-rounds"]
    fractionTrain = context.run_config["fraction-train"]
        

    #loading the state
    state = CheckpointState.load_checkpoint("/home/frabru99/Sync/MasterThesisTests/frabru/testONNX/flower_onnx/tfexample/training_artifacts/checkpoint")
    parameters = dict(state.parameters)
    parameters_dict = {name: parameters[name].data for name in list(parameters.keys())} 

    arrays = ArrayRecord(parameters)

    #Initialize FedAvg Strategy
    strategy = FedAvg(fraction_train=fractionTrain)
    

    result = strategy.start(
        grid=grid, 
        initial_arrays=arrays,
        train_config=config, 
        num_rounds=num_rounds
    )


"""
TODO: FIND THE RIGHT WAY TO SAVE THE CHECKPOINT IN .ONNX FILE
"""
 #print("\nSaving final model to disk...")
#state_dict = result.arrays
#torch.save(state_dict, "final_model.pt")
