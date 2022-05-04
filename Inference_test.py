import time
import random
import numpy as np
from tflite_runtime.interpreter import Interpreter

# load input data
test_x = np.load('test_lead2_x.npy')
test_y = np.load('test_y.npy')
print (f'test_x shape:{test_x.shape}')
print (f'test_y shape:{test_y.shape}')

# (n,4096,1)
input_data = test_x[0:63]
print(input_data.shape)
# np float64 to float32
input_data = np.float32(input_data)

# load tflite model to interpreter
m#odel_path = '2022_05_04_16_10_04.tflite'
model_path = 'backup_model_best.tflite'
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()  # Needed before execution!
input_details = interpreter.get_input_details()  # Model has single input.
output_details = interpreter.get_output_details()  # Model has single output.
print(input_details,'\n')
print(output_details)

#test with tflite model
i = 0
true_conut = 0 
all_invoke_time = 0 
for idx, ecg_sample in enumerate(input_data, start=-1):
    input_data = ecg_sample
    label = np.float32(test_y[idx])
    #print(input_data.shape)
    input_data = np.reshape(np.float32(input_data),(1,input_data.shape[0],input_data.shape[1]))
    #print(input_data.shape)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    start_invoke_time = time.time()
    interpreter.invoke()
    finish_invoke_time = time.time()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = output_data[0]
    invoke_time = round(finish_invoke_time - start_invoke_time, 5)
    #print (f'output data:{output_data}')
    # tf.keras.metrics.binary_accuracy(    y_true, y_pred, threshold=0.5)
    threshold = 0.5
    prediction = output_data
    prediction[prediction>threshold] = 1
    prediction[prediction<=threshold] = 0

    # acc
    if np.array_equal(prediction, label, equal_nan=False):
        true_conut += 1 
        #print ('True')
    else:
        print (f'id:{idx} is False')
        print (f'prediction:{prediction}')
        print (f'real label:{label}')
        print (f'invoke time:{invoke_time}s')
    # time
    all_invoke_time = all_invoke_time + invoke_time
    # num
    i += 1
acc = true_conut/i
print (f'acc:{acc}')
average_time = round(all_invoke_time/i, 5)
print (f'average time:{average_time}s')