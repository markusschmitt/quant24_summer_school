import numpy as np
import matplotlib.pyplot as plt

# from functools import partial
import pickle

# Some helper functions

def norm_fun(v, df=lambda x: x):
    return jnp.real(jnp.conj(jnp.transpose(v)).dot(df(v)))

def save_to_disk(data, L, g, h, fn=""):
    with open(fn+"L="+str(L)+"_g="+str(g)+"_h="+str(h)+".pkl", 'wb') as f:
        pickle.dump(data, f)
        
def load_from_disk(L, g, h, fn=""):
    with open(fn+"L="+str(L)+"_g="+str(g)+"_h="+str(h)+".pkl", 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data


def plot_observables(fn=""):

    loaded_data = load_from_disk(fn)
    
    obs_data = np.array(loaded_data["observables"])
    res_data = np.array(loaded_data["residuals"])
    
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    
    ax[0].plot(obs_data[:,0], obs_data[:,2])
    ax[0].set_xlabel(r"Time $Jt$")
    ax[0].set_ylabel(r"Magnetization $\langle \hat X\rangle$")
    ax[1].semilogy(res_data[:,0], res_data[:,1])
    ax[1].set_xlabel(r"Time $Jt$")
    ax[1].set_ylabel(r"TDVP error")
    
    plt.tight_layout()
    
    
def plot_parameters(fn=""):
    
    param_data = load_from_disk(fn)["parameters"]

    n_hidden = param_data[0]["L1"]["kernel"].shape[1]
    n_hidden = 5

    fig, ax = plt.subplots(n_hidden, figsize=(10,8), sharex=True)

    for k in range(n_hidden):

        D = np.concatenate([np.reshape( np.array(p["L1"]["kernel"][:,k]), (1,-1) ) for p in param_data])

        ax[k].imshow(np.abs(np.transpose(D[::5,:])))
        ax[k].set_ylabel("Phys. site")
    
    ax[-1].set_xlabel("Time step")
    plt.tight_layout()
    
    
def plot_gradient(grads):

    grads = np.array(grads)
    
    fig, ax = plt.subplots(figsize=(12,4))

    ax.plot(np.real(grads[0,0,:]), '-o', label="real part", linewidth=0.8, markersize=4)
    ax.plot(np.imag(grads[0,0,:]), '-o', label="imaginary part", linewidth=0.8, markersize=4)
    ax.set_xlabel(r"Parameter index $k$", fontsize=20)
    ax.set_ylabel(r"$\frac{\partial}{\partial\theta_k}\log\psi_\theta(s)$", fontsize=20)

    plt.legend()
    plt.tight_layout()
    
    
def plot_coeffs(coeffs):
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

    for c in coeffs[0]:
        ax.scatter(np.imag(c),np.real(c)/np.log(10))

        
    ax.set_title("Polar plot of wave function coefficients")
    plt.show()
    

import jVMC
from jVMC.util import ground_state_search
from jVMC.operator import BranchFreeOperator
from jVMC.operator import Sx
from jVMC.operator import scal_opstr
def initialize_in_X_state(psi, L, sampler):
    
    H_init = BranchFreeOperator()

    for l in range(L):
        H_init.add( scal_opstr(-1.0, (Sx(l),)) )  # - Sx(l)

    gsEquation = jVMC.util.TDVP(sampler, pinvTol=1e-8, rhsPrefactor=1.0, makeReal='real', diagonalShift=10)

    ground_state_search(psi, H_init, gsEquation, sampler, numSteps=200)