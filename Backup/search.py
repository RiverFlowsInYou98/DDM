import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from chain import approx_hmc
from utils import *
import os
import gc

# True parameters: mu=0.2, z=1.5
JOB_ID = os.environ["SLURM_ARRAY_JOB_ID"]
JOB_TASK_ID = os.environ["SLURM_ARRAY_JOB_ID"] + "-" + os.environ["SLURM_ARRAY_TASK_ID"]

filename = 'data_seed0.npy'
data = np.load('../Data/' + filename) # 100 x 2 data
start_idx, end_idx = 0, 100
data = data[start_idx: end_idx]
print('%s %d: %d' %(filename, start_idx, end_idx))
print('data shape:' + str(data.shape))
print()

mu_list = np.arange(0, 0.5, 0.01)
probs_list = []
for mu in mu_list:
    chain = approx_hmc(mu=mu, sigma=1, a=4, z=1.5, dt=0.001, Nx=100, verbose=False)
    loss = LogLikelihood(chain, data)
    print("%.2f, %.5f" %(mu, loss))
    probs_list.append(loss)
    gc.collect()


np.savetxt('../Results/probs' + JOB_TASK_ID + '.txt', np.array(probs_list))
print("argmin: %.2f" %mu_list[np.argmin(probs_list)])
# plt.plot(mu_list, probs_list)
# plt.savefig('probs1d.png', bbox_inches='tight')



# mu_list = np.linspace(-1, 1, 20)
# z_list = np.linspace(0.5, 3.5, 20)
# probs_list = np.zeros((len(mu_list), len(z_list)))
# for i in range(len(mu_list)):
#     for j in range(len(z_list)):
#         chain = approx_hmc(mu=mu_list[i], sigma=1, a=4, z=z_list[j], dt=0.001, Nx=100, verbose=False)
#         probs_list[i,j] = LogLikelihood(chain, data)

# np.savetxt('results/probs2d.txt', np.array(probs_list))

# hm = sns.heatmap(probs_list, cmap='hot')
# hm.set_xlabel('$z$', fontsize=10)
# hm.set_ylabel('$\mu$', fontsize=10)
# hm.set_xticklabels(z_list)
# hm.set_yticklabels(mu_list)
# plt.savefig('probs2d.png', bbox_inches='tight')