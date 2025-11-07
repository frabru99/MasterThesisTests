import numpy
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
        image, label = images.cpu().numpy(), labels.cpu().numpy()

        pred_onx = sess.run(
            [label_name], 
            {input_name: image} 
        )[0]

        curr_pred = numpy.argmax(pred_onx, axis=1)

        print(f"PREDICTION: {curr_pred}")

        correct += (curr_pred == label).sum()
        total +=  len(label)

        pbar.set_postfix({
            'acc': f'{100.*correct/total:.2f}%'
        })

    final_acc = 100. * correct / total
    print(f"\nFinal Test Accuracy: {final_acc:.2f}%")
    return final_acc
