import numpy as np
import scipy as sp
import matplotlib
from matplotlib import pyplot as plt
from scipy.integrate import odeint
# matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = (20,10)
plt.rcParams['figure.dpi'] = 200
plt.style.use("ggplot")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

# model of evolution of iron concentration in the growt matrix
def ode_iron_everywhere(x, t, r_mr, r_rl, r_d, c_max, m):
    x_m, x_r, x_l = x
    dxdt = [-(r_mr*m*x_m)/(c_max - x_m), (r_mr*m*x_m)/(c_max - x_m) - r_rl * x_r - r_d * x_r, r_rl * x_r - r_d * x_l]
    return dxdt

def ode_iron_roots(x, t, r_mr, c_max, m):
    dxdt = -(r_mr*m*x)/(c_max - x)
    return dxdt



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Ardi and Randa')
    #  arbitrary values of model parameters
    r_mr = 0.1 # rate of intake
    m = 5 # dry weight
    c_max = 50 # maximum concentration of iron in the growth matrix
    r_rl = 0.007 # rate of roots to leaves transfer
    r_d = 0.0008 # rate of decay
#     initial value of the concentration
    x0 = 6
    time = range(21) # time over which we run the ODE


    x_matrix = odeint(ode_iron_roots, x0, time, args=(r_mr, c_max, m))
    print(x_matrix)
    # The initial values of the iron concentration as a list [Iron_in_matrix, Iron_in_roots, Iron_in_leaves]
    iron0 = [200, 320, 66]
    iron_everywhere = odeint(ode_iron_everywhere, iron0, time, args=(r_mr, r_rl, r_d, c_max, m))
    print(iron_everywhere)


    plt.plot(time, x_matrix, 'b', label='x_matrix(t)')
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.grid()
    plt.show()

    tablse_size = iron_everywhere.shape
    legendLabels = ['Fe in matrix','Fe in roots','Fe in leaves']
    for i in range(3):
        plt.plot(time, iron_everywhere[:,i],'o',label=legendLabels[i])
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.grid()
    plt.show()

