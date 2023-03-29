import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from chain import *
from utils import *
import os
import gc
import pickle

JOB_ID = os.environ["SLURM_ARRAY_JOB_ID"]
JOB_TASK_ID = os.environ["SLURM_ARRAY_JOB_ID"] + "-" + os.environ["SLURM_ARRAY_TASK_ID"]

filename = 'data_ih.pkl'
with open('../Data/' + filename, "rb") as fp:
    pkl = pickle.load(fp)
    
data, intervals = pkl[0], pkl[1]
start_idx, end_idx = 0, 1000
data = data[start_idx: end_idx]
intervals = intervals[start_idx: end_idx]
print('%s %d: %d' %(filename, start_idx, end_idx))
print('data shape:' + str(data.shape))
print()

def loss_fun2(ihchain, data, intervals):
    """
    compute the negative log likelihood
    likelihood = product of P(X(Tk)=Ck)
    """
    logprob = 0 
    for idx in range(len(data)):
        Tk, Ck = data[idx]
        endpoints = intervals[idx]
        logprob -= np.log(ihchain.ExitProb_logdp(Tk, Ck, endpoints))
        if np.isinf(logprob):
            raise ValueError('Infty detected, computation is stopped.')
    return logprob / len(data)

# True parameters
# mu1 = 0.5, mu2 = -0.3, sigma = 1, a = 4.0, z = 1.5/4.0

mu1_list = np.linspace(-1, 1, 21)
mu2_list = np.linspace(-1, 1, 21)
loss_list = np.zeros((len(mu1_list), len(mu2_list)))
for i in range(len(mu1_list)):
    for j in range(len(mu2_list)):
        chain = ihmc(mu1=mu1_list[i], mu2=mu2_list[j], sigma=1, a=4, z=1.5/4.0, dt=0.002, Nx=50)
        loss_list[i,j] = loss_fun2(chain, data, intervals)
        print("%.2f, %.2f: %.5f" %(mu1_list[i], mu2_list[j], loss_list[i,j]))
        gc.collect()

np.savetxt('../Results/probs-' + JOB_TASK_ID + '.txt', np.array(loss_list))
print("results saved!")
indices = np.unravel_index(loss_list.argmin(), loss_list.shape)
print(indices)
print("argmin: %.2f, %.2f" %(mu1_list[indices[0]], mu2_list[indices[1]]))