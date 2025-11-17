"""tftest: A flower / tensorflowlite app"""

import keras
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from tfexample.task import load_data, load_model

# Flower Client app
client_app = ClientApp()

@client_app.train()
def my_local_training_fn(msg: Message, context: Context):

    """
    Traing the model on local data (is a simulation)
    """

    # Reset local Tensorflow state
    keras.backend.clear_session()

    # Load the data
    partition_id = context.node_config["partition-id"] # Partition ID sent by server to the client
    num_partitions = context.node_config["num-partitions"]
    x_train, y_train, _, _ = load_data(partition_id, num_partitions)

    # Load the model
    learning_rate = context.run_config["learning-rate"]
    model = load_model(learning_rate)
    
    # Initialize local model with central server weights
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    # Train the model
    history = model.fit(
        x_train,
        y_train,
        epochs = epochs,
        batch_size = batch_size,
        verbose = verbose,    
    )

    # Get training metrics
    train_loss = history.history["loss"][-1] if "loss" in history.history else None
    train_acc = (
        history.history["accuracy"][-1] if "accuracy" in history.history else None   
    )

    # Send metrics to server
    metrics = {"num-examples": len(x_train)}
    if train_loss is not None:
        metrics["train_loss"] = train_loss
    if train_acc is not None:
        metrics["train_acc"] = train_acc
    content = RecordDict({"arrays": ArrayRecord(model.get_weights()), "metrics": MetricRecord(metrics)})

    return Message(content = content, reply_to = msg)

@client_app.evaluate()
def my_local_evaluate_fn(msg: Message, context: Context):

    """
    Evaluate model on local data
    """

    # Reset local Tensorflow state
    keras.backend.clear_session()

    # Load data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, _, x_test, y_test = load_data(partition_id, num_partitions)

    # Load the model
    lr = context.run_config["learning-rate"]
    model = load_model(lr)

    # Initialize the model and evaluate
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())
    eval_loss, eval_acc = model.evaluate(x_test, y_test, verbose=0)

    # Pack the metrics
    metrics = {
        "eval_loss": eval_loss, 
        "eval_acc": eval_acc,
        "num-examples": len(x_test),     
    }

    content = RecordDict({"metrics": MetricRecord(metrics)})

    return Message(content=content, reply_to=msg)
