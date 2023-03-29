import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import sys

sys.path.append("..")
from chain import approx_hmc
from utils import *


# True parameters: mu=0.3, z=1.5/4.0
JOB_ID = os.environ["SLURM_ARRAY_JOB_ID"]
JOB_TASK_ID = os.environ["SLURM_ARRAY_JOB_ID"] + "-" + os.environ["SLURM_ARRAY_TASK_ID"]

filename = "data_seed10.npy"
data = np.load("../../Data/" + filename)  # data_num x 2 data
start_idx, end_idx = 0, 1000
data = data[start_idx:end_idx]
print("%s %d: %d" % (filename, start_idx, end_idx))
print("data shape:" + str(data.shape))
print()

# mu_list = np.arange(0, 0.5, 0.01)
# loss_list = []
# for mu in mu_list:
#     chain = approx_hmc(mu=mu, sigma=1, a=4, z=1.5/4, dt=0.001, Nx=100, verbose=False)
#     loss = loss_fun(chain, data)
#     print("%.2f: %.5f" %(mu, loss))
#     loss_list.append(loss)
#     gc.collect()


# np.savetxt('../../Results/probs-' + JOB_TASK_ID + '.txt', np.array(loss_list))
# print("results saved!")
# print("argmin: %.2f" %mu_list[np.argmin(loss_list)])


mu_list = np.linspace(-1, 1, 21)
z_list = np.linspace(1 / 16, 15 / 16, 15)
loss_list = np.zeros((len(mu_list), len(z_list)))
for i in range(len(mu_list)):
    for j in range(len(z_list)):
        chain = approx_hmc(
            mu=mu_list[i], sigma=1, a=4, z=z_list[j], dt=0.001, Nx=100, verbose=False
        )
        loss_list[i, j] = loss_fun(chain, data)
        print("%.2f, %.2f: %.5f" % (mu_list[i], z_list[j], loss_list[i, j]))
        gc.collect()

np.savetxt('../../Results/probs-' + JOB_TASK_ID + '.txt', np.array(loss_list))
print("results saved!")

idx = np.unravel_index(np.argmin(loss_list), loss_list.shape)
print("argmin: %.2f, %.2f" % (mu_list[idx[0]], z_list[idx[1]]))
