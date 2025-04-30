import os
import copy
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy import integrate
from scipy.special import hermite
from tqdm import tqdm
from mpmath import laguerre
from mpmath import hyperu
from mpmath import pcfd
import timeit

## linear quench analytic
## parameter setting
global J, hi, hf, k, quenchtime
hi = 10  # initial transverse field
hf = 2  # final transverse field
quenchtime = 0

J = 1;
# vmax = 2 * J * min(1,hf); # 2 * J * min(1,hf) / 2

dt = 1e-2;
tend = 10/(1); # match for vmax*t
step = int(np.ceil(tend/dt));
t_analytic = np.arange(0,step+1,1) * dt;

N = 1000
eps = 1e-5
n = np.arange(- N / 2, N/2-1+eps)
if N % 2 == 0: # even sector : Neveu-Schwarz(NS) sector
    k = 2*np.pi/N * (n+1/2)
    pstep = N;
    prange = 2 * np.pi;
    dk = prange / pstep;

else: # odd sector : Ramond(R) sector
    k = 2*np.pi/N * n
    pstep = N;
    prange = 2 * np.pi;
    dk = prange / pstep;

#k = np.linspace(-prange/2, prange/2, pstep+1);

# the diagonalization condition, which the initial condition is exp(i*theta_k) = (g - exp(i*k)) / sqrt(1+g^2-2*g*cosk)
theta_k = - 1j * np.log((hi - np.exp(1j*k)) / np.sqrt(1+hi**2 - 2*hi*np.cos(k)))
theta_minus_k = - 1j * np.log((hi - np.exp(1j*(-k))) / np.sqrt(1+hi**2 - 2*hi*np.cos(-k)))
theta_k = np.real(theta_k)
theta_minus_k = np.real(theta_minus_k)
uki = np.cos(theta_k/2)
v_minus_ki = np.sin(theta_minus_k/2)
v_conj_minus_ki = np.conj(v_minus_ki) # initial condition
uki_dif = - 1j * 2 * J * (hi - np.cos(k)) * uki - 1j * 2 * J * np.sin(k) * v_minus_ki
v_conj_minus_ki_dif = - 1j * 2 * J * np.sin(k) * uki + 1j * 2 * J * (hi - np.cos(k)) * v_minus_ki


## after the quench, the equations for the bogoliubov coefficients simplify to d^2/dt^2(y_k(t)) + epsilon_k(g_f)^2 * y_k(t) = 0
def TM_function_post_quench(t, uk0, v_conj_minus_k0):
    global J, hi, hf, k
    A_k = 2 * J * (hf - np.cos(k))
    B_k = 2 * J * np.sin(k)
    E_k = np.sqrt((A_k)**2 + (B_k)**2)
    # y_k = c3_y*exp(i*E_k*t) + c4_y*exp(-i*E_k*t)
    c3_u = np.exp(-1j*E_k*quenchtime) / (2*E_k) * ((E_k-A_k)*uk0 - B_k*(v_conj_minus_k0))
    c4_u = np.exp(1j*E_k*quenchtime) / (2*E_k) * ((E_k+A_k)*uk0 + B_k*(v_conj_minus_k0))
    c3_v = np.exp(-1j*E_k*quenchtime) / (2*E_k) * ((E_k+A_k)*(v_conj_minus_k0) - B_k*uk0)
    c4_v = np.exp(1j*E_k*quenchtime) / (2*E_k) * ((E_k-A_k)*(v_conj_minus_k0) + B_k*uk0)
    uk = c3_u * np.exp(1j*E_k*t) + c4_u * np.exp(-1j*E_k*t)
    v_conj_minus_k = c3_v * np.exp(1j*E_k*t) + c4_v * np.exp(-1j*E_k*t)
    return uk, v_conj_minus_k


## numerical integration
transverse_magnetization_t_analytic = np.zeros(len(t_analytic))
transverse_magnetization_instant_t = ( - np.real(1j  / np.pi * dk * ((uki[0]*(v_minus_ki[0])+uki[-1]*(v_minus_ki[-1])) + np.sum(uki[1:-1]*(v_minus_ki[1:-1])))) + 1 / np.pi * dk * ((abs((uki[0]))**2+abs((uki[-1]))**2) + np.sum(abs((uki[1:-1]))**2)) - 1 ) # without divide by 2 for first and last element of function (yet sum over all elements of function) # since it's integrate in a 2*pi circle, nothing overlap
transverse_magnetization_t_analytic[0] = transverse_magnetization_instant_t
print('Time = ', t_analytic[0])
print('transverse magnetization = ', transverse_magnetization_instant_t)

start = timeit.default_timer()
for i in range(1, step+1):
    ukn, v_conj_minus_kn = TM_function_post_quench(t_analytic[i], uki,  v_conj_minus_ki);
    v_minus_kn = np.conj(v_conj_minus_kn)
    transverse_magnetization_instant_t =  ( - np.real(1j  / np.pi * dk * ((ukn[0]*(v_minus_kn[0])+ukn[-1]*(v_minus_kn[-1])) + np.sum(ukn[1:-1]*(v_minus_kn[1:-1])))) + 1 / np.pi * dk * ((abs((ukn[0]))**2+abs((ukn[-1]))**2) + np.sum(abs((ukn[1:-1]))**2)) - 1 )
    transverse_magnetization_t_analytic[i] = transverse_magnetization_instant_t;


    if t_analytic[i] % 1 == 0:
        print('Time = ', t_analytic[i])
        print('transverse magnetization = ', transverse_magnetization_instant_t)

end = timeit.default_timer()
print('Total computing time = ', end-start)

#transverse_magnetization_t_rk4_linear_k100 = transverse_magnetization_t_rk4_linear
# fprintf('transverse magnetization = %d\n', transverse_magnetization_t)
plt.figure(dpi=100,figsize=(12,8))
plt.plot((1)*t_analytic,transverse_magnetization_t_analytic,'-');
plt.title(r'Transverse Magnetization, ' + r'$B_{x_{i}} = $' + str(hi) + '$, B_{x_{f}} = $' + str(hf));
plt.xlabel('t');
plt.ylabel(r'$M_x(t)$');
plt.show()
