import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import pickle

import sys
sys.path.append("..")

from chain import *
from utils import *

JOB_ID = os.environ["SLURM_ARRAY_JOB_ID"]
JOB_TASK_ID = os.environ["SLURM_ARRAY_JOB_ID"] + "-" + os.environ["SLURM_ARRAY_TASK_ID"]

filename = 'data_ih2.pkl'
with open('../../Data/' + filename, "rb") as fp:
    pkl = pickle.load(fp)
    
data, _ = pkl[0], pkl[1]
start_idx, end_idx = 0, 200
data = data[start_idx: end_idx]
print('%s %d: %d' %(filename, start_idx, end_idx))
print('data shape:' + str(data.shape))
print()

def loss_fun3(chain, data):
    """
    compute the negative log likelihood
    likelihood = product of P(X(Tk)=Ck)
    """
    logprob = 0 
    for idx in range(len(data)):
        Tk, Ck = data[idx]
        Ck *= chain.a
        logprob -= np.log(chain.TotalExitProb(Tk, Ck))
        if np.isinf(logprob):
            raise ValueError('Infty detected, computation is stopped.')
    return logprob / len(data)

# True parameters
# mu1 = -1, mu2 = 1, sigma = 1, lbda = 2, a = 1 (upper bdy 1, lower bdy -1), z = 0.5


lbda_list = np.linspace(1.5, 2.5, 41)
loss_list = np.zeros(len(lbda_list))
for i in range(len(lbda_list)):
    chain = MC_PDDM(mu1=-1, mu2=1, sigma=1, lbda=lbda_list[i], a=1, z=0.5, t_nd=0.0, dt=0.0003, Nx=100)
    loss_list[i] = loss_fun3(chain, data)
    print("%.4f: %.5f" %(lbda_list[i], loss_list[i]))
    gc.collect()

print("argmin: %.2f" %lbda_list[np.argmin(loss_list)])
np.savetxt('../../Results/probs-' + JOB_TASK_ID + '.txt', np.array(loss_list))
print("results saved!")
