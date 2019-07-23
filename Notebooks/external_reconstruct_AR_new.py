#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import mrcfile
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import collections
import tensorflow as tf
from tensorflow import spectral
from ClassFiles.relion_fixed_it import load_star
from ClassFiles.AdversarialRegularizer import AdversarialRegulariser
from ClassFiles.ut import l2
from ClassFiles.Utilities import registration
from ClassFiles.ut import locate_gt, rfft, irfft, unify_form, Rescaler


# In[2]:


# saves_path = '/local/scratch/public/sl767/SPA/Saves/Adversarial_Regulariser/AllData/AllAugmentation'
saves_path = '/local/scratch/public/sl767/SPA/Saves/Adversarial_Regulariser/Cutoff_20/Translation_Augmentation'
regularizer = AdversarialRegulariser(saves_path)


# In[3]:


Registrator = registration.LocalRegistrator()


# In[4]:


def l2_gt(x):
    image=unify_form(np.copy(x))
    r=Rescaler(image)
    r.normalize(image)
    reg=Registrator.register(image=image, reference=ground_truth)
    return l2(reg-ground_truth)


# In[5]:


def vis(data, fourier=True):
    if fourier:
        data = irfft(data)
    plt.imshow(data.squeeze()[...,45])
#     plt.imshow(np.mean(data.squeeze(), axis=-1))


# In[17]:


NOISE_LEVEL = '016'
ITERATION = '03'
METHODE = 'EM'
PDB_ID = '5A0M'


# In[18]:


base_path = '/local/scratch/public/sl767/MRC_Data/Data/Data_0{n}_10k/eval/{m}/{PDB}/'
path=base_path+'{PDB}_mult0{n}_it0{i}_half2_class001_external_reconstruct.star'
path = path.format(n=NOISE_LEVEL, m=METHODE, PDB = PDB_ID, i=ITERATION)


# In[19]:


file=load_star(path)


# In[20]:


with mrcfile.open(file['external_reconstruct_general']['rlnExtReconsDataReal']) as mrc:
    data_real = mrc.data.copy()
with mrcfile.open(file['external_reconstruct_general']['rlnExtReconsDataImag']) as mrc:
    data_im = mrc.data.copy()
with mrcfile.open(file['external_reconstruct_general']['rlnExtReconsWeight']) as mrc:
    kernel = mrc.data.copy()
with mrcfile.open(locate_gt(PDB_ID, full_path=False)) as mrc:
    ground_truth = mrc.data.copy()
with mrcfile.open(file['external_reconstruct_general']['rlnExtReconsResult']) as mrc:
    naive_recon = mrc.data.copy()
    
ground_truth = unify_form(ground_truth)
r_gt = Rescaler(ground_truth)
r_gt.normalize(ground_truth)

complex_data=data_real + 1j * data_im


# In[23]:


REG = 0.03
    
tikhonov_kernel = kernel + 1e6
print(tikhonov_kernel.max(),tikhonov_kernel.min())
precondioner = np.abs(np.divide(1, tikhonov_kernel))
precondioner /= precondioner.max()
print(precondioner.max()/precondioner.min())
tikhonov = np.divide(complex_data, tikhonov_kernel)
reco = np.copy(tikhonov)

# The scales produce gradients of order 1
ADVERSARIAL_SCALE=(96**(-0.5))
DATA_SCALE=1/(10*96**3)

IMAGING_SCALE=96

for k in range(50):
    STEP_SIZE=1.0 * 1 / np.sqrt(1 + k / 20)
    
    gradient = regularizer.evaluate(reco)
    g1 = REG * gradient * ADVERSARIAL_SCALE
#     print(l2(gradient))
    g2 = DATA_SCALE*(np.multiply(reco, tikhonov_kernel) - complex_data)
    
    g = g1 + g2
#     reco = reco - STEP_SIZE * 0.02 * g
    
    reco_o = np.copy(reco)
    reco = reco - STEP_SIZE * precondioner * g
    
    reco = np.fft.rfftn(np.maximum(0, np.fft.irfftn(reco)))
        
    #reco = reco - STEP_SIZE*(g1 + g2 + g3)
    if k%15==0:
        plt.figure(k, figsize=(18,3))
        plt.subplot(171)
        vis(IMAGING_SCALE*np.fft.rfftn(np.maximum(0, np.fft.irfftn(tikhonov))))
        plt.colorbar()
        plt.subplot(172)
        vis(IMAGING_SCALE*reco)
        plt.colorbar()
        plt.subplot(173)
        vis(IMAGING_SCALE*precondioner*g1)
        plt.colorbar()
        plt.subplot(174)
        vis(IMAGING_SCALE*precondioner*g2)
        plt.colorbar()
        plt.subplot(175)
        vis(IMAGING_SCALE*precondioner*(g1+g2))
        plt.colorbar()
        plt.subplot(176)
        vis(IMAGING_SCALE*(reco-reco_o))
        plt.colorbar()
        plt.subplot(177)
        vis(IMAGING_SCALE*(reco - np.fft.rfftn(np.maximum(0, np.fft.irfftn(tikhonov)))))
        plt.colorbar()
        plt.show()


# In[24]:


plt.figure(figsize=(20,20))
plt.subplot(121)
vis(np.fft.rfftn(np.maximum(0, np.fft.irfftn(tikhonov))))
plt.subplot(122)
vis(reco)


# In[38]:


def evaluate(reg):
    tikhonov_kernel = kernel + 1e6
    precondioner = np.abs(np.divide(1, tikhonov_kernel))
    precondioner /= precondioner.max()
    tikhonov = np.divide(complex_data, tikhonov_kernel)
    reco = np.copy(tikhonov)

    # The scales produce gradients of order 1
    ADVERSARIAL_SCALE=(96**(-0.5))
    DATA_SCALE=1/(10*96**3)

    IMAGING_SCALE=96

    for k in range(70):
        STEP_SIZE=1.0 * 1 / np.sqrt(1 + k / 20)

        gradient = regularizer.evaluate(reco)
        g1 = reg * gradient * ADVERSARIAL_SCALE
    #     print(l2(gradient))
        g2 = DATA_SCALE*(np.multiply(reco, tikhonov_kernel) - complex_data)

        g = g1 + g2
    #     reco = reco - STEP_SIZE * 0.02 * g
        reco = reco - STEP_SIZE * precondioner * g

        reco = np.fft.rfftn(np.maximum(0, np.fft.irfftn(reco)))
    return l2_gt(irfft(reco))


# In[39]:


parameters = [0.0, 0.01, 0.02, 0.03]
results = {}
for reg in parameters:
    results['Reg_{}'.format(reg)] = evaluate(reg = reg)


# In[41]:


print(results)


# In[ ]:





# # Optimal parameters:
# 
# - SNR .02, first iterate: .03 --> Improvement .015
# - SNR .02, 10th iterate: .05 --> Improvement .081
# - SNR .01, first iterate: .005 (reasonable till .01) --> Improvement .002
# - SNR .01, 10th: .005 (reasonable till .01) --> Improvement .002
# - SNR .012, first iterate: .01 --> Improvement .006
# - SNR .012, 10th iterate: .01 --> Improvement .006
# - SNR .016, first iterate: .02 --> Improvement .012
# - SNR .016, 10th: .03 --> Improvement .042

# In[15]:


for l, item in res1.items():
    print(l)
    print(item)


# # Results for 10th iterate, SNR .01

# In[25]:


for l, item in results.items():
    print(l)
    print(item)


# # Results for 01st iterate, SNR .02

# In[32]:


for l, item in results.items():
    print(l)
    print(item)


# # Results for 10th iterate, SNR .02

# In[42]:


for l, item in results.items():
    print(l)
    print(item)


# In[ ]:




