```python
import time
import numpy as np
import matplotlib.pyplot as plt
from numba.experimental import jitclass
from numba import jit, njit, prange, float64, int32
from quantecon.optimize.scalar_maximization import brent_max
from interpolation import interp
```

```python
data_type = [
    ('bet', float64),
    ('sig', float64),
    ('k_lim', float64),
    ('Z', float64[:]),
    ('Pi', float64[:,:]),
    ('q', float64)
]

@jitclass(data_type)
class huggett93:
    def __init__(self, bet, sig, k_lim, Z, Pi, q):
        self.bet, self.sig, self.k_lim = bet, sig, k_lim
        self.Z, self.Pi, self.q = Z, Pi, q
```

```python
@njit
def obj_fnc(kp, ik, iz, k_vec, V_mat, H_class):
    # Get parameters values from class
    bet, sig = H_class.bet, H_class.sig
    z_vec, Pi, q = H_class.Z, H_class.Pi, H_class.q

    # Auxiliary variables
    Nz = len(z_vec)

    # Get k and z from indexes ik and iz
    k = k_vec[ik]
    z = z_vec[iz]

    # Compute Vp_mat = V(k'=kp,z') for all z'
    Vp_vec = np.zeros(Nz)
    for j in range(Nz):
        Vp_vec[j] = interp(k_vec, V_mat[:,j], kp)

    # Compute optimal leisure implied by intra temporal FOC
    # l = h - ((1-alph)*np.exp(z)*k**alph/zet)**(1/(alph+thet))

    # Ensure optmal leisure choice is feasible
    # if l<0:
    #    l=0
    # if l>h:
    #    l=h

    # Auxiliary variables: y, c, c_subs
    # y = np.exp(z)*k**alph*(h-l)**(1-alph)
    c = z + k - q*kp
    # c_subs = zet*(h-l)**(1+thet)/(1+thet)

    # Auxiliary variable E[V(k',z')|z]: conditional expectation given z
    EVp_vec =  Vp_vec @ Pi[iz,:].T

    # Compute objective function W ensuring kp is feasible
    if c > 0:
        u = (c**(1-sig))/(1-sig)
        W = u + bet*EVp_vec
    else:
        W = -1e100

    return W

@njit(parallel=True)
def Vnew_mat_fnc(k_vec, Vold_mat, H_class):
# Method that numerically solves the maximization problem
    # Get parameters values from class
    bet, sig = H_class.bet, H_class.sig
    z_vec, Pi, q = H_class.Z, H_class.Pi, H_class.q

    # Initialize vectors to store updated value and policy functions
    Nk = k_vec.shape[0]
    Nz = z_vec.shape[0]
    Vnew_mat = np.zeros((Nk,Nz))
    Gnew_kp_mat = np.zeros((Nk,Nz))

    # Find kp that maximizes objective function
    # For each z,
    for j in range(Nz):
        # For each k,
        for i in prange(Nk):
            # Axuliary variables
            k = k_vec[i]
            z = z_vec[j]

            # Compute optimal FESIBLE leisure choice
            # l = h - ((1-alph)*np.exp(z)*k**alph/zet)**(1/(alph+thet))
            # if l<0:
            #    l=0
            # if l>h:
            #    l=h

            # Auxiliary variables: y, c_subs
            # y = np.exp(z)*k**alph*(h-l)**(1-alph)
            # c_subs = zet*(h-l)**(1+thet)/(1+thet)

            # Bounds for kp
            kp_min = kp_min = H_class.k_lim
            kp_max = (z + k - 1e-8) / q


            # Maximize objective function
            Gnew_kp_mat[i,j], Vnew_mat[i,j], info = brent_max(obj_fnc, kp_min, kp_max, args=(i, j, k_vec, Vold_mat, H_class), xtol=1e-5, maxiter=500)

            # Store l
            # Gnew_l_mat[i,j] = l

    # Return value and policy functions
    return Vnew_mat, Gnew_kp_mat

@njit
def VFI_fnc(k_vec, V0_mat, H_class, eps_v, max_iter, display):
# Method that performs value function iteration to find value and policy functions
    # Step 0:
    # z_vec, k_vec and V0_vec are provided

    # Step 1:
    Vold_mat = V0_mat.copy()

    # Steps 2 & 3:
    iter_count = 0              # iteration counter
    stop_crit = eps_v + 1       # must ensure while loop starts, any number > eps_v will do
    while (stop_crit > eps_v) & (iter_count<max_iter):
        # Step 2: numerically solve problem given Vold_vec
        Vnew_mat, Gnew_kp_mat = Vnew_mat_fnc(k_vec=k_vec, Vold_mat=Vold_mat, H_class=H_class)

        # Step 3: update stop_crit, Vold_vec & iter_count
        stop_crit = np.max(np.abs(Vnew_mat-Vold_mat))
        Vold_mat = Vnew_mat.copy()
        iter_count+=1

        # Optional display of converence criteria
        if display:
            print(stop_crit)

    # Return value and policy functions if algorithm converged
    V_mat, G_kp_mat = Vnew_mat.copy(), Gnew_kp_mat.copy()
    if iter_count >= max_iter:
        print('VFI did not converge')

    # Return value and policy functions
    return V_mat, G_kp_mat
```

```python
# Markov chain for earnings
Z = np.array([1., .1])
Pi = np.array([[.925, 1-.925],[.5, .5]])

# Set remaining parameters
β = .99322
σ = 1.5
k_lim = -2.

# Asset price
q = 1.0124

# Create economy
e1 = huggett93(bet=β, sig=σ, k_lim=k_lim, Z=Z, Pi=Pi, q=q)

# Set up initial guess
Nz = len(Z)
Nk = 200
k_max = 2*abs(k_lim)
k_vec = np.linspace(k_lim,k_max,Nk)
V0_mat = np.zeros((Nk,Nz))

# Set up iteration parameters
eps_v = 1e-6
max_iter = 3000
```

```python
V_mat, G_kp_mat = VFI_fnc(k_vec=k_vec, V0_mat=V0_mat,
                          H_class=e1, eps_v=eps_v,
                          max_iter=max_iter, display=True)
```

```python
plt.plot(k_vec, G_kp_mat[:,0], label="Policy k' when z = z1")
plt.plot(k_vec, k_vec, linestyle='--', label='45-degree line')
plt.plot(k_vec, G_kp_mat[:,1], label="Policy k' when z = z2")
plt.legend(); plt.xlabel('k'); plt.ylabel("k'")
plt.show()
```

```python
def cdfnew_fnc(k_vec, kp_mat, cdfold_mat, huggett93_class):

    Nk = len(k_vec)
    Nz = len(huggett93_class.Z)
    Pi = huggett93_class.Pi

    N_mat = np.zeros((Nk, Nz))

    for iz in range(Nz):
        for ik_prime in range(Nk):
            kp_target = k_vec[ik_prime]

            if kp_target == k_vec[0]:
                k_i = k_vec[0]

                for ik in range(Nk):

                    if kp_mat[ik, iz] == k_vec[0]:
                        k_i = k_vec[ik]
                    else:
                        break
            else:
                k_i = interp(kp_mat[:, iz], k_vec, kp_target)
                k_i = min(max(k_i, k_vec[0]), k_vec[-1])

            N_mat[ik_prime, iz] = interp(k_vec, cdfold_mat[:, iz], k_i)

    cdf_new_mat = np.zeros((Nk, Nz))
    for iz_prime in range(Nz):
        for ik_prime in range(Nk):
            for iz in range(Nz):
                cdf_new_mat[ik_prime, iz_prime] += N_mat[ik_prime, iz] * Pi[iz, iz_prime]

    return cdf_new_mat


def cdf_iter_fnc(k_vec, kp_mat, cdf0_mat, huggett93_class, eps_cdf, max_iter, display):
    cdfold_mat = cdf0_mat.copy()
    iter_count = 0
    stop_crit = eps_cdf + 1

    while (stop_crit > eps_cdf) and (iter_count < max_iter):
        cdfnew_mat = cdfnew_fnc(k_vec, kp_mat, cdfold_mat, huggett93_class)
        stop_crit = np.max(np.abs(cdfnew_mat - cdfold_mat))
        cdfold_mat = cdfnew_mat.copy()
        iter_count += 1

        if display:
            print(stop_crit)

    if iter_count >= max_iter:
        print('CDF iteration did not converge')

    stationary_cdf = cdfnew_mat.copy()
    return stationary_cdf
```

```python
def a_agg(k_vec, kp_mat, cdf_mat, huggett93_class):
# Computes aggregate level of assets given CDF and asset policy
    # Get parameters values from class
    z_vec = huggett93_class.Z

    # Auxiliary variables
    Nz = len(z_vec)
    dk_vec =np.outer(k_vec[1:]-k_vec[:-1],np.ones(Nz))

    # Approximate pdf
    pdf_mat = np.zeros(cdf_mat.shape)
    pdf_mat[0,:] = cdf_mat[0,:]
    pdf_mat[1:,:] = (cdf_mat[1:,:]-cdf_mat[:-1,:])

    # Use trapezoidal rule to approximate integral
    fnc = kp_mat * pdf_mat
    agg_assets = np.sum(dk_vec*(fnc[1:,:]+fnc[:-1,:]))/2

    return agg_assets
```

```python
# Markov chain for earnings
Z = np.array([1., .1])
Pi = np.array([[.925, 1-.925],[.5, .5]])
Nz = len(Z)

# Set remaining parameters
β = .99322
σ = 1.5
k_lim = -2.

# Asset price
q = 1.0124

# Create economy
e1 = huggett93(bet=β, sig=σ, k_lim=k_lim, Z=Z, Pi=Pi, q=q)

# Set up capital grid
Nk = 200
k_max = 2*abs(k_lim)
k_vec = np.linspace(k_lim,k_max,Nk)

# VFI parameters
eps_v = 1e-4
max_iter = 3000
V0_mat = np.zeros((Nk,Nz))

# CDF iteration parameters
eps_cdf = 1e-6
max_iter_cdf = 500
cdf0_mat = np.cumsum(np.ones((Nk,Nz))/(2*Nk), axis=0)
```

```python
V_mat, G_kp_mat = VFI_fnc(k_vec=k_vec, V0_mat=V0_mat, H_class=e1, eps_v=eps_v, max_iter=max_iter, display=False)
plt.plot(k_vec, G_kp_mat[:,0], label="Policy k' when z = z1")
plt.plot(k_vec, k_vec, linestyle='--', label='45-degree line')
plt.plot(k_vec, G_kp_mat[:,1], label="Policy k' when z = z2")
plt.legend(); plt.xlabel('k'); plt.ylabel("k'")
```

```python
stationary_cdf = cdf_iter_fnc(k_vec, G_kp_mat, cdf0_mat, e1, eps_cdf, max_iter_cdf, display=False)
plt.plot(k_vec, stationary_cdf[:, 0], label='z = 1.0 (high)')
plt.plot(k_vec, stationary_cdf[:, 1], label='z = 0.1 (low)')
plt.xlim([-2, 1])
plt.ylim([0, 1])
plt.legend(); plt.xlabel('k'); plt.ylabel('CDF')
```

```python
agg_assets = a_agg(k_vec, G_kp_mat, stationary_cdf, e1)
print(f"Aggregate assets at q = {q}: {agg_assets:.6f}, which is ~0")
```
