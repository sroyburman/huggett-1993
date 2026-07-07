# ============================================================
# Imports
# ============================================================
#
# Load numerical libraries, visualization tools, JIT compilation,
# optimization routines, and interpolation utilities used
# throughout the project.

import time
import numpy as np
import matplotlib.pyplot as plt
from numba.experimental import jitclass
from numba import jit, njit, prange, float64, int32
from quantecon.optimize.scalar_maximization import brent_max
from interpolation import interp

# ============================================================
# Huggett Model Class
# ============================================================
#
# Stores the structural parameters defining the Huggett (1993)
# economy. Using a jitclass allows Numba to efficiently compile
# the model for high-performance numerical computation.

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
    """
    Container for the structural parameters of the Huggett (1993) economy.
 
    Bundles the household's preference parameters, the borrowing limit,
    the exogenous endowment (labor income) process, and the equilibrium
    asset price into a single Numba jitclass so that it can be passed
    efficiently into JIT-compiled functions (e.g. the Bellman objective
    and value function iteration).
 
    Parameters
    ----------
    bet : float
        Household's discount factor, beta (0 < bet < 1).
    sig : float
        Coefficient of relative risk aversion (CRRA utility parameter).
    k_lim : float
        Ad hoc (exogenous) borrowing limit on asset holdings.
    Z : np.ndarray of float64
        Grid of possible values for the idiosyncratic endowment, z.
    Pi : np.ndarray of float64, shape (len(Z), len(Z))
        Transition probability matrix for the Markov chain governing z,
        where Pi[i, j] = P(z' = Z[j] | z = Z[i]).
    q : float
        Price of the one-period discount bond (asset price).
    """
    def __init__(self, bet, sig, k_lim, Z, Pi, q):
        self.bet, self.sig, self.k_lim = bet, sig, k_lim
        self.Z, self.Pi, self.q = Z, Pi, q

# ============================================================
# Bellman Objective Function
# ============================================================
#
# Computes the value associated with a candidate choice of
# next-period assets. The objective consists of
# current-period utility + discounted expected continuation 
# value.
#
# This function is repeatedly maximized during value
# function iteration.

@njit
def bellman_objective(kp, ik, iz, k_vec, V_mat, H_class):
    """
    Evaluate the household's Bellman objective for a candidate choice of
    next-period asset holdings.
 
    Computes current-period CRRA utility from consumption plus the
    discounted expected continuation value, given the household is at
    asset grid point k_vec[ik] with endowment Z[iz] and chooses
    next-period assets kp.
 
    Parameters
    ----------
    kp : float
        Candidate choice of next-period asset holdings, k'.
    ik : int
        Index of the current asset grid point (k_vec[ik] = k).
    iz : int
        Index of the current endowment state (H_class.Z[iz] = z).
    k_vec : np.ndarray of float64
        Grid of asset holdings (state space for k).
    V_mat : np.ndarray of float64, shape (len(k_vec), len(Z))
        Current guess of the value function, V_mat[i, j] = V(k_vec[i], Z[j]).
    H_class : huggett93
        Instance holding the model's structural parameters.
 
    Returns
    -------
    float
        The value of choosing kp: current utility plus discounted
        expected continuation value. Returns -1e100 if the implied
        consumption is non-positive (infeasible choice).
    """
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

# ============================================================
# Value Function Iteration
# ============================================================
#
# Solves the household's dynamic programming problem by
# repeatedly applying the Bellman operator until convergence.
#
# Outputs:
#   - Value function
#   - Optimal savings policy function

@njit(parallel=True)
def update_value_function(k_vec, Vold_mat, H_class):
    """
    Perform one Bellman-operator update of the value function.
 
    For every point on the (assets, endowment) grid, maximizes the
    Bellman objective over feasible next-period asset choices using
    Brent's method (via quantecon's brent_max) bounded below by the
    borrowing limit and above by the level implied by non-negative
    consumption. The outer loop over endowment states is parallelized
    using Numba's prange.
 
    Parameters
    ----------
    k_vec : np.ndarray of float64
        Grid of asset holdings.
    Vold_mat : np.ndarray of float64, shape (len(k_vec), len(Z))
        Value function from the previous iteration.
    H_class : huggett93
        Instance holding the model's structural parameters.
 
    Returns
    -------
    Vnew_mat : np.ndarray of float64, shape (len(k_vec), len(Z))
        Updated value function after one Bellman iteration.
    Gnew_kp_mat : np.ndarray of float64, shape (len(k_vec), len(Z))
        Corresponding optimal next-period asset choice for each grid point.
    """
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
            Gnew_kp_mat[i,j], Vnew_mat[i,j], info = brent_max(bellman_objective, kp_min, kp_max, args=(i, j, k_vec, Vold_mat, H_class), xtol=1e-5, maxiter=500)

            # Store l
            # Gnew_l_mat[i,j] = l

    # Return value and policy functions
    return Vnew_mat, Gnew_kp_mat

@njit
def value_function_iteration(k_vec, V0_mat, H_class, eps_v, max_iter, display):
    """
    Solve the household's dynamic programming problem via value function
    iteration.
 
    Repeatedly applies the Bellman operator (update_value_function) until the
    sup-norm distance between successive value functions falls below a
    tolerance, or a maximum number of iterations is reached. Prints a
    warning if the maximum number of iterations is hit without
    convergence.
 
    Parameters
    ----------
    k_vec : np.ndarray of float64
        Grid of asset holdings.
    V0_mat : np.ndarray of float64, shape (len(k_vec), len(Z))
        Initial guess for the value function.
    H_class : huggett93
        Instance holding the model's structural parameters.
    eps_v : float
        Convergence tolerance on the sup-norm of successive value
        function iterates.
    max_iter : int
        Maximum number of iterations to perform.
    display : bool
        If True, prints the convergence criterion at each iteration.
 
    Returns
    -------
    V_mat : np.ndarray of float64, shape (len(k_vec), len(Z))
        Converged (or final) value function.
    G_kp_mat : np.ndarray of float64, shape (len(k_vec), len(Z))
        Optimal next-period asset policy function associated with V_mat.
    """
    # Step 0:
    # z_vec, k_vec and V0_vec are provided

    # Step 1:
    Vold_mat = V0_mat.copy()

    # Steps 2 & 3:
    iter_count = 0              # iteration counter
    stop_crit = eps_v + 1       # must ensure while loop starts, any number > eps_v will do
    while (stop_crit > eps_v) & (iter_count<max_iter):
        # Step 2: numerically solve problem given Vold_vec
        Vnew_mat, Gnew_kp_mat = update_value_function(k_vec=k_vec, Vold_mat=Vold_mat, H_class=H_class)

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

# ============================================================
# Model Calibration
# ============================================================
#
# Specify the income process, household preferences,
# borrowing constraint, asset price, and numerical grids
# used throughout the benchmark economy.

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

# ============================================================
# Solve Household Problem
# ============================================================
#
# Solve the household optimization problem using value
# function iteration and recover the optimal savings policy.

V_mat, G_kp_mat = value_function_iteration(k_vec=k_vec, V0_mat=V0_mat,
                          H_class=e1, eps_v=eps_v,
                          max_iter=max_iter, display=True)

# ============================================================
# Figure 1: Optimal Savings Policy Functions
# ============================================================
#
# Plot the optimal next-period asset holdings for households
# in the high- and low-endowment states.

plt.plot(k_vec, G_kp_mat[:,0], label="Policy k' when z = z1")
plt.plot(k_vec, k_vec, linestyle='--', label='45-degree line')
plt.plot(k_vec, G_kp_mat[:,1], label="Policy k' when z = z2")
plt.legend(); plt.xlabel('k'); plt.ylabel("k'")
plt.show()

# ============================================================
# Stationary Distribution
# ============================================================
#
# Compute the invariant cross-sectional distribution of
# households over assets and endowment states implied by
# the optimal policy function.

def update_cdf(k_vec, kp_mat, cdfold_mat, huggett93_class):
    """
    Perform one iteration of the CDF (distribution) operator.
 
    Updates the cross-sectional CDF of (assets, endowment) pairs one
    period forward, given the asset policy function kp_mat and the
    endowment transition matrix. For each next-period asset grid point,
    finds the current-period asset level(s) that map into it under the
    policy function, evaluates the current CDF there, and then
    aggregates across current endowment states using the Markov
    transition probabilities to obtain the next-period CDF.
 
    Parameters
    ----------
    k_vec : np.ndarray of float64
        Grid of asset holdings.
    kp_mat : np.ndarray of float64, shape (len(k_vec), len(Z))
        Asset policy function, kp_mat[i, j] = k'(k_vec[i], Z[j]).
    cdfold_mat : np.ndarray of float64, shape (len(k_vec), len(Z))
        Current guess of the joint CDF over (assets, endowment).
    huggett93_class : huggett93
        Instance holding the model's structural parameters.
 
    Returns
    -------
    np.ndarray of float64, shape (len(k_vec), len(Z))
        Updated joint CDF over (assets, endowment) after one period.
    """
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


def compute_stationary_distribution(k_vec, kp_mat, cdf0_mat, huggett93_class, eps_cdf, max_iter, display):
    """
    Iterate the CDF operator to find the stationary cross-sectional
    distribution of assets and endowments.
 
    Repeatedly applies update_cdf until the sup-norm distance between
    successive CDF iterates falls below a tolerance, or a maximum number
    of iterations is reached. Prints a warning if the maximum number of
    iterations is hit without convergence.
 
    Parameters
    ----------
    k_vec : np.ndarray of float64
        Grid of asset holdings.
    kp_mat : np.ndarray of float64, shape (len(k_vec), len(Z))
        Asset policy function.
    cdf0_mat : np.ndarray of float64, shape (len(k_vec), len(Z))
        Initial guess for the joint CDF over (assets, endowment).
    huggett93_class : huggett93
        Instance holding the model's structural parameters.
    eps_cdf : float
        Convergence tolerance on the sup-norm of successive CDF iterates.
    max_iter : int
        Maximum number of iterations to perform.
    display : bool
        If True, prints the convergence criterion at each iteration.
 
    Returns
    -------
    np.ndarray of float64, shape (len(k_vec), len(Z))
        The (approximately) stationary joint CDF over (assets, endowment).
    """
    cdfold_mat = cdf0_mat.copy()
    iter_count = 0
    stop_crit = eps_cdf + 1

    while (stop_crit > eps_cdf) and (iter_count < max_iter):
        cdfnew_mat = update_cdf(k_vec, kp_mat, cdfold_mat, huggett93_class)
        stop_crit = np.max(np.abs(cdfnew_mat - cdfold_mat))
        cdfold_mat = cdfnew_mat.copy()
        iter_count += 1

        if display:
            print(stop_crit)

    if iter_count >= max_iter:
        print('CDF iteration did not converge')

    stationary_cdf = cdfnew_mat.copy()
    return stationary_cdf

# ============================================================
# Aggregate Asset Holdings
# ============================================================
#
# Integrate the optimal policy function over the stationary
# distribution to compute aggregate asset holdings.
#
# In equilibrium, aggregate assets should be approximately
# zero for the benchmark bond price.

def compute_aggregate_assets(k_vec, kp_mat, cdf_mat, huggett93_class):
    """
    Compute the aggregate level of next-period asset holdings implied by
    the stationary distribution and the asset policy function.
 
    Approximates the cross-sectional density (pdf) by differencing the
    CDF, then integrates the policy function against this density using
    the trapezoidal rule to obtain aggregate assets. Used to check
    whether a candidate bond price q clears the asset market
    (aggregate assets ≈ 0).
 
    Parameters
    ----------
    k_vec : np.ndarray of float64
        Grid of asset holdings.
    kp_mat : np.ndarray of float64, shape (len(k_vec), len(Z))
        Asset policy function.
    cdf_mat : np.ndarray of float64, shape (len(k_vec), len(Z))
        Stationary joint CDF over (assets, endowment).
    huggett93_class : huggett93
        Instance holding the model's structural parameters.
 
    Returns
    -------
    float
        Aggregate (average) level of next-period asset holdings implied
        by the stationary distribution.
    """
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

# ============================================================
# Benchmark Calibration
# ============================================================
#
# Reconstruct the benchmark economy and initialize the
# stationary distribution iteration.

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

V_mat, G_kp_mat = value_function_iteration(k_vec=k_vec, V0_mat=V0_mat, H_class=e1, eps_v=eps_v, max_iter=max_iter, display=False)
plt.plot(k_vec, G_kp_mat[:,0], label="Policy k' when z = z1")
plt.plot(k_vec, k_vec, linestyle='--', label='45-degree line')
plt.plot(k_vec, G_kp_mat[:,1], label="Policy k' when z = z2")
plt.legend(); plt.xlabel('k'); plt.ylabel("k'")

# ============================================================
# Figure 2: Stationary Asset Distribution
# ============================================================
#
# Plot the stationary cumulative distribution of assets
# for households in each endowment state.

stationary_cdf = compute_stationary_distribution(k_vec, G_kp_mat, cdf0_mat, e1, eps_cdf, max_iter_cdf, display=False)
plt.plot(k_vec, stationary_cdf[:, 0], label='z = 1.0 (high)')
plt.plot(k_vec, stationary_cdf[:, 1], label='z = 0.1 (low)')
plt.xlim([-2, 1])
plt.ylim([0, 1])
plt.legend(); plt.xlabel('k'); plt.ylabel('CDF')

# ============================================================
# Aggregate Asset Market Check
# ============================================================
#
# Compute aggregate asset holdings implied by the stationary
# distribution to verify approximate market clearing.

agg_assets = compute_aggregate_assets(k_vec, G_kp_mat, stationary_cdf, e1)
print(f"Aggregate assets at q = {q}: {agg_assets:.6f}, which is ~0")
