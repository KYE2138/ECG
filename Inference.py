import time
import random
import numpy as np
import matplotlib.pyplot as plt
from tflite_runtime.interpreter import Interpreter
# GUI
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *
import tkinter.font as tkFont
# Sensor
import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

ecg_single = []
pred_result = []

def ECG_single(freq):
    i2c = busio.I2C(board.SCL, board.SDA)
    ads = ADS.ADS1115(i2c)
    chan = AnalogIn(ads, ADS.P1)
    idx = []
    ecgs_signals = []
    for i in range(4096):
        ecgs_signals.append(chan.voltage)
        idx.append(i)
        time.sleep(1.0/freq)
    return ecgs_signals

def t_data():
    # load input data
    test_x = np.load('test_lead1_x.npy')
    test_y = np.load('test_y.npy')
    #print (f'test_x shape:{test_x.shape}')
    #print (f'test_y shape:{test_y.shape}')
    r = random.randrange(test_x.shape[0])
    # (1,4096,1)
    input_data = test_x[r:r+1]
    #print(input_data.shape)
    # np float64 to float32
    input_data = np.float32(input_data)
    return input_data

def inference(input_data, model_path):

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
    #model_path = 'Arch_2022_05_05_03_27_58.tflite'
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

    #plt_x = np.reshape(input_data,(4096,))
    #plt.plot(plt_x)
    #plt.show()
    return [prediction, invoke_time]

#inference(ECG_single(400), )
#inference(t_data(), )


# GUI
root = Tk()
root.title("ECG Automatic diagnosis")

f = Figure(figsize=(6, 5), dpi=100)
f_plot = f.add_subplot(111)

def draw_record():
    f_plot.clear()
    ecg_single = ECG_single(400)
    x = np.reshape(ecg_single,(4096,))
    y = np.arange(11)
    f_plot.plot(x, y)
    canvs.draw()
    return ecg_single

def draw_t_record():
    f_plot.clear()
    ecg_single = t_data()
    x = np.reshape(ecg_single,(4096,))
    y = np.arange(11)
    f_plot.plot(x, y)
    canvs.draw()
    return ecg_single

def inference_record(ecg_single, model_path):
    result = inference(ecg_single, model_path)
    return result

canvs = FigureCanvasTkAgg(f, root)
canvs.get_tk_widget().pack(side=LEFT, fill=BOTH, expand=5)
Button(root, width=20, command=lambda:ecg_single==draw_t_record(), text='Start record').pack(padx=10,pady=10,ipady=30)
Button(root, width=20, command=lambda:pred_result==inference_record(ecg_single,'Arch_2022_05_05_03_27_58.tflite'), text='Automatic diagnosis').pack(padx=10,pady=10,ipady=30)

Label(root, font=("Times", 20, "italic"), text="", fg="black").pack(padx=10,pady=10,ipady=30)

root.mainloop()