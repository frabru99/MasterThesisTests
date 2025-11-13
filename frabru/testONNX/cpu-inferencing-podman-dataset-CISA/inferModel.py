import numpy as np
import onnxruntime as ort
from tqdm import tqdm


def inferenceModel(model_name, test_dataset, test_loader, device):
    # Enable profiling here
    #sess_options = ort.SessionOptions()
    #sess_options.enable_profiling = True  

    print("Creating ONNX Runtime session with profiling enabled...")

    sess = ort.InferenceSession(f"./artifacts/{model_name}.onnx", 
                                #sess_options=sess_options,
                                providers=ort.get_available_providers()
                                )


    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    print(f"Model Input Name: {input_name}")
    print(f"Model Output Name: {label_name}")


    correct = 0
    total = 0
    print("Executing inference...")

    pbar = tqdm(test_loader, desc='test')

    for images, labels in pbar:
        image, label = images.cpu().numpy().astype(np.float32), labels.cpu().numpy()


        print(image.min(), image.max(), image.mean())


        pred_onx = sess.run(
            [label_name], 
            {input_name: image} 
        )[0]

        curr_pred = np.argmax(pred_onx, axis=1)

        print(f"PREDICTION: {curr_pred}")

        correct += (curr_pred == label).sum()
        total +=  len(label)

        pbar.set_postfix({
            'acc': f'{100.*correct/total:.2f}%'
        })

    final_acc = 100. * correct / total
    print(f"\nFinal Test Accuracy: {final_acc:.2f}%")
    return final_acc

def balancingDataset(dataset):
    from collections import defaultdict 
    from torch.utils.data import Subset 
    class_indices = defaultdict(list) 
    
    for i, (_, label) in enumerate(dataset): 
        class_indices[label].append(i) 
        # Prendi max 20 per classe, ad esempio 
    
    balanced_idx = [i for lbls in class_indices.values() for i in lbls[:200]] 
    balanced = Subset(dataset, balanced_idx)

    return balanced

