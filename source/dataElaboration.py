import matplotlib.pyplot as plt
import csv
import numpy as np


MODEL_320_PATH = "results/InceptionV3_320x320/results.csv"
MODEL_416_PATH = "results/InceptionV3_416x416/results.csv"
MODEL_640_PATH = "results/InceptionV3_640x640/results.csv"

MODEL_320_LOCAL_PATH = "results/Local/InceptionV3_320x320/results.csv"
MODEL_416_LOCAL_PATH = "results/Local/InceptionV3_416x416/results.csv"
#MODEL_640_LOCAL_PATH= "results/Local/InceptionV3_640x640/results.csv"


MODEL_LIST = [MODEL_320_PATH,MODEL_416_PATH,MODEL_640_PATH,MODEL_320_LOCAL_PATH,MODEL_416_LOCAL_PATH]

mean_time_big = []
mean_time_small= []

for MODEL_PATH_RESULT in MODEL_LIST:
    time = []
    with open(MODEL_PATH_RESULT,'r') as file:
        reader = csv.reader(file)
        header = next(reader)

        for row in reader:
            time.append(row[5])

    time_np = np.array(time).astype(float)
    small_net = np.mean(time_np[:30]).item()
    mean_time_small.append(small_net)
    big_net = np.mean(time_np[30:]).item()
    mean_time_big.append(big_net)


mean_time_gpu_big = mean_time_big[:3]
mean_time_cpu_big = mean_time_big[3:]
mean_time_gpu_small = mean_time_small[:3]
mean_time_cpu_small = mean_time_small[3:]

mean_time_cpu_small.append(0)
mean_time_cpu_big.append(0)

X = np.arange(3)
pos = np.arange(3)

labels=["GPU SMALL","GPU BIG","CPU SMALL","CPU BIG"]


plt.bar(X,mean_time_gpu_small,width=0.2,color=('#9E3A3A'),label='GPU SMALL')
plt.bar(X+0.2,mean_time_gpu_big,width=0.2,color=('#EF2951'),label='GPU BIG')
plt.bar(X+0.4,mean_time_cpu_small,width=0.2,color=('#6B8E23'),label='CPU SMALL')
plt.bar(X+0.6,mean_time_cpu_big,width=0.2,color=('#556B2F'),label='CPU BIG')
plt.xlabel('Model Size')
plt.ylabel('Time')
plt.legend()
plt.xticks(pos+0.3,['320','416','640'])
plt.savefig("results/trainingTime.png")
plt.show()




