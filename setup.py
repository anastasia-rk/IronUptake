import numpy as np
import scipy as sp
import matplotlib
import emcee
import csv
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import odeint
#
# matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = (20,10)
plt.rcParams['figure.dpi'] = 400
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.style.use("ggplot")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

# model of evolution of iron concentration in the growt matrix
def ode_iron_leaf_root(x, t, params):
    # the models takes as an input:
    # rate of uptake from matrix to roots, r_mr
    # rate of uptake from roots to leaves, r_rl
    # rate of decay, r_d
    # carrying capacity of the root matrix, c_max
    # iron concentration at the
    r_mr, r_rl, r_d, c_max, fe_matrix, dw = params
    fe_root, fe_leaf = x
    dxdt = [(r_mr*m*fe_matrix)/(c_max - fe_matrix) - r_rl * fe_root - r_d * fe_root, r_rl * fe_root - r_d * fe_leaf]
    return dxdt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Ardi and Randa')
    # Note that samples are taken every other day
    sampling_days = [0,2,4,7,9,11,14,16,18,21]
    #  arbitrary values of model parameters
    r_mr = 0.1 # rate of intake
    m = 5 # dry weight
    c_max = 50 # maximum concentration of iron in the growth matrix
    r_rl = 0.007 # rate of roots to leaves transfer
    r_d = 0.0008 # rate of decay
    l_matrix = 150
#     initial value of the concentration
    x0 = 6
    time = np.linspace(0, 30, 301)# time over which we run the ODE
    sampling_indeces = [i for i, e in enumerate(time) if e in sampling_days] # we need indeces that correspond to sampling days to select only those points for likelihood

    # The initial values of the iron concentration as a list [Iron_in_matrix, Iron_in_roots, Iron_in_leaves]
    iron0 = [320, 66]
    params = [r_mr, r_rl, r_d, c_max, l_matrix, m]
    # solve with smaller timestep to avoid numerical errors
    iron_everywhere = odeint(ode_iron_leaf_root, iron0, time, args=(params,))
    # print(iron_everywhere)
    # print modelling output
    legendLabels = ['Fe in roots','Fe in leaves']
    for i in range(2):
        plt.plot(sampling_days, iron_everywhere[sampling_indeces,i],'o',label=legendLabels[i])
    plt.legend(loc='best')
    plt.xlabel('time, days')
    plt.grid()
    plt.show()