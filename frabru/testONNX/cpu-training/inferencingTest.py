import numpy
import onnxruntime as ort


BATCH_SIZE = 1
CHANNELS = 3
HEIGHT = 224
WIDTH = 224

X_test = numpy.random.rand(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH).astype(numpy.float32)

sess_options = ort.SessionOptions()
sess_options.enable_profiling = True  # Enable profiling here


print("Creating ONNX Runtime session with profiling enabled...")
sess = ort.InferenceSession("./training_artifacts/resnet.onnx", 
                             sess_options=sess_options,
                             providers=ort.get_available_providers())

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
print(f"Model Input Name: {input_name}")
print(f"Model Output Name: {label_name}")


print("Executing inference...")
pred_onx = sess.run(
    [label_name], 
    {input_name: X_test} # X_test is already numpy.float32, so no need to cast again
)[0]

pred_onx = sess.run(
    [label_name], 
    {input_name: X_test} # X_test is already numpy.float32, so no need to cast again
)[0]


profiler_file = sess.end_profiling()
print(f"\nInference completed. Profile trace saved to: {profiler_file}")
print("\nModel Output (First 5 Predictions):")
print(pred_onx[:5])
