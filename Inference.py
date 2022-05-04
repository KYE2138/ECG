import time
import random
import numpy as np
from tflite_runtime.interpreter import Interpreter

# load input data
test_x = np.load('test_x.npy')
test_y = np.load('test_y.npy')
print (f'test_x shape:{test_x.shape}')
print (f'test_y shape:{test_y.shape}')
r = random.randrange(test_x.shape[0])
# (1,4096,1)
input_data = test_x[r:r+1]
print(input_data.shape)
# np float64 to float32
input_data = np.float32(input_data)

# load tflite model to interpreter
model_path = '2022_05_04_16_10_04.tflite'
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()  # Needed before execution!
input_details = interpreter.get_input_details()  # Model has single input.
output_details = interpreter.get_output_details()  # Model has single output.
print(input_details,'\n')
print(output_details)

# set input data
interpreter.set_tensor(input_details[0]['index'], input_data)
# invoke
start_invoke_time = time.time()
interpreter.invoke()
finish_invoke_time = time.time()
# get output
output_data = interpreter.get_tensor(output_details[0]['index'])
output_data = output_data[0]
invoke_time = round(finish_invoke_time - start_invoke_time, 2)
print (f'output data:{output_data}')
# tf.keras.metrics.binary_accuracy(    y_true, y_pred, threshold=0.5)
threshold = 0.5
prediction = output_data
prediction[prediction>threshold] = 1
prediction[prediction<=threshold] = 0
print (f'prediction:{prediction}')
print (f'real label:{test_y[r:r+1]}')
print (f'invoke time:{invoke_time}s')

