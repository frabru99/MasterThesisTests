from flwr.app import Context, Message
from flwr.clientapp import ClientApp
import torchvision.transforms as transforms
from flwr.app import ArrayRecord, MetricRecord, ConfigRecord, RecordDict

client_app = ClientApp()


BATCH_SIZE = None #Batch Size
NUM_CLASSES = 10 #Same as Classes of the Dataset
EPOCHS = None #Number of epochs, TODO: add the EPOCHS in code




device_type=None
transform=train_dataset=train_loader=None
state=model=optimizer=None



def deviceChoice():
    global device_type

    print("Device Choice...")

    #or rocm and ROCMExecutionProvider for AMD
    device_type= "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    print(f"Using device: {device_type}")



def dataLoad(): # TODO: LOAD TRAINING DATA 
    global transform, train_dataset, train_loader, BATCH_SIZE

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    # Download the Dataset 
    train_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                                train=True,
                                                download=True, 
                                                transform=transform)

    # Data Loader for batch generation
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)


def loadArtifacts():
    global state, model, optimizer
    print("Loading ONNX artifacts...")

    try:
        state = onnxtraining.CheckpointState.load_checkpoint(f"{ARTIFACT_DIR}/checkpoint")
        model = Module(f"{ARTIFACT_DIR}/training_model.onnx", state, device=device_type)
        optimizer = Optimizer(f"{ARTIFACT_DIR}/optimizer_model.onnx", model)
    except Exception as e:
        print(f"Error in loading artifacts from {ARTIFACT_DIR}.")
        print("Make sure to extract the correct model with right number of classes in FC layer.")
        print(f"Error Details: {e}")
        exit()


def trainingFn():

    global train_loader, model, optimizer

    loss_accum = 0
    # Training Loops 
    for i, (images_tensor, labels_tensor) in enumerate(train_loader):

        # Images in numpy array
        images = images_tensor.numpy()
      
        # Labels in numpy array, in int64 type
        labels = labels_tensor.numpy().astype(np.int64)

        # TRAINING STEP: Forward, Loss and Backword propagation
        loss = model(images, labels)

        loss_accum += loss

        # Applies the gradients on weights and biases
        optimizer.step()

        # Gradients reset
        model.lazy_reset_grad()

    
        if i % 10 == 0: # Every 10 batches
            print(f"Step {i}, Loss: {loss.item():.4f}")

        

    return loss_accum/len(train_loader)
    

@client_app.train()
def train(msg: Message, context: Context):
    global state, train_dataset, BATCH_SIZE, EPOCHS

    print(dict(msg.content))

    recordDictReceived =  msg.content
    
    #---Configuration---#

    BATCH_SIZE = context.run_config["batch-size"]
    EPOCHS = context.run_config["local-epochs"]
        
    
    ##----DATA LOAD----##
    deviceChoice()
    dataLoad()
    loadArtifacts()

    ##----TRAINING----##
    train_loss = trainingFn()
    

    ##----RESPONSE----##
    trained_parameters = dict(state.parameters)
    trained_parameters_dict = {name: trained_parameters[name].data for name in list(trained_parameters.keys())}

    model_record = ArrayRecord(trained_parameters_dict)
    metrics = {
        "train_loss": train_loss, 
        "num-examples": len(train_dataset) # Or something else?
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)

