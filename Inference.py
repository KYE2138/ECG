import time
import random
import numpy as np
import matplotlib.pyplot as plt
from tflite_runtime.interpreter import Interpreter

# load input data
test_x = np.load('test_lead2_x.npy')
test_y = np.load('test_y.npy')
#print (f'test_x shape:{test_x.shape}')
#print (f'test_y shape:{test_y.shape}')
r = random.randrange(test_x.shape[0])
# (1,4096,1)
input_data = test_x[r:r+1]
#print(input_data.shape)
# np float64 to float32
input_data = np.float32(input_data)

# normolize to -1 , 1
#signal (4096, lead_num)
signal_T = input_data.T
#print (signal_T.shape)
#signal_T (lead_num, 4096)
for signal_T_idx, signal_T_signal in enumerate (signal_T):
    # signal_T_signal (4096)
    #print (signal_T_signal.shape)
    pos_max = max(signal_T_signal)
    neg_max = abs(min(signal_T_signal))
    if max(pos_max, neg_max)==0:
        print (idx,signal_T_idx)
    #print (max(pos_max, neg_max))
    signal_T_signal = signal_T_signal/max(pos_max, neg_max)
    signal_T[signal_T_idx] = signal_T_signal
input_data = signal_T.T



# load tflite model to interpreter
#model_path = '2022_05_04_16_10_04.tflite'
model_path = 'backup_model_best.tflite'
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()  # Needed before execution!
input_details = interpreter.get_input_details()  # Model has single input.
output_details = interpreter.get_output_details()  # Model has single output.
#print(input_details,'\n')
#print(output_details)

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
print (f'real label:{test_y[r:r+1][0]}')
print (f'invoke time:{invoke_time}s')

plt_x = np.reshape(input_data,(4096,))
plt.plot(plt_x)
plt.show()
