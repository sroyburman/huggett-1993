# Huggett (1993) Replication

A Python implementation of the Huggett (1993) incomplete markets model using dynamic programming, value function iteration, and numerical methods to compute optimal savings policies, the stationary wealth distribution, and aggregate asset holdings.

--

## Overview

The Huggett (1993) model is a foundational heterogeneous-agent macroeconomic model in which infinitely-lived households face idiosyncratic income risk and make optimal savings decisions subject to a borrowing constraint. Unlike representative-agent models, households differ according to their current asset holdings and income state, generating a nontrivial equilibrium wealth distribution.

This project implements the computational solution of the Huggett model from scratch in Python. This implementation solves the household optimization problem using value function iteration, computes the invariant distribution of assets across households, and evaluates aggregate asset holdings implied by the equilibrium policy functions.

## Objectives

The main objectives of this project are to:

1. Solve the household dynamic programming problem
2. Compute optimal savings policy functions
3. Construct the stationary distribution of household assets
4. Compute aggregate asset holdings implied by the stationary distribution
5. Verify approximate asset market clearing for benchmark asset price

## Model

Households maximize expected discount lifetime utility

$$
V(k, z; q) = \max_{c, k'} \; \frac{c^{1-\sigma}}{1-\sigma} + \beta \sum_{z' \in \mathbf{Z}} \pi(z'|z) V(k', z'; q)
$$

subject to

$$
\begin{aligned}
c &\ge 0, \\
k' &\ge \underline{k}, \\
c &= z + k - qk'.
\end{aligned}
$$

where

- $k$ denotes current asset holdings,
- $k'$ denotes next-period asset holdings,
- $z$ is the current income state,
- $q$ is the asset price,
- $\beta$ is the discount factor,
- $\sigma$ is the coefficient of relative risk aversion.

Income follows a two-state Markov process

$$
Z=\{z_1,z_2\},
$$

with transition matrix

$$
\Pi=
\begin{bmatrix}
\pi_{11} & \pi_{12} \\
\pi_{21} & \pi_{22}
\end{bmatrix}.
$$

## Parameters

Benchmark parameters:

| Parameter | Value |
|-----------|------:|
| Discount factor ($\beta$) | 0.99322 |
| Risk aversion ($\sigma$) | 1.5 |
| Borrowing limit | -2.0 |
| Asset price ($q$) | 1.0124 |

Income process:

```python
Z = np.array([1.0, 0.1])

Pi = np.array([
    [0.925, 0.075],
    [0.500, 0.500]
])
```

## Computational Approach

The implementation consists of four primary components.

### Value Function Iteration

The Bellman equation is solved numerically using value function iteration.

The optimization problem is solved using Brent's method while future values are evaluated through linear interpolation.

To improve computational efficiency, the implementation uses

- Numba JIT compilation
- Parallel loops (`prange`)
- Interpolation
- Brent optimization

### Policy Function Computation

Once the value function converges, the model computes the optimal asset policy function

$$
k' = g(k,z)
$$

for every point on the asset grid and for each income state.

The resulting policy functions describe optimal household saving decisions.

### Stationary Distribution

Using the optimal policy functions, this project computes the invariant distribution of assets.

Beginning from an initial cumulative distribution function, the distribution is repeatedly updated using optimal policy rules and Markov transitional probabilities until convergence.

The stationary distribution characterizes the long-run distribution of household wealth across income states.

### Aggregate Asset Holdings

Lastly, aggregate asset holdings are computed by numerically integrating over the stationary distribution using the trapezoidal rule. 

This allows the model to evaluate whether a chosen asset price approximately clears the asset market.

## Results

The implementation produces several important outputs.

### Policy Functions

The optimal savings policy is computed for both high-income and low-income households.

The policy function illustrates how optimal savings vary with current asset holdings and income.

### Stationary Distribution

The invariant distribution describes the long-run distribution of assets across households.

The distribution differs from income states as a result of persistent earnings risk.

### Aggregate Assets

Aggregate assets are computed from the stationary distribution.

For the benchmark parameterization,

$$
q = 1.0124
$$

aggregate asset holdings are ~0, indicating approximate market clearing.

## Visualizations

The repository includes:

- Value function convergence
- Optimal savings policy functions
- Stationary cumulative distribution functions
- Aggregate asset calculation

## Skills Demonstrated

- Python
- Object-oriented programming
- Dynamic programming
- Value function iteration
- Markov chains
- Numerical optimization
- Numba acceleration
- Brent maximization
- Linear interpolation
- Numerical integration
- Computational macroeconomics

## Packages Used

`numpy`; `matplotlib`; `numba`; `quantecon`; `interpolation`

## Potential Extensions

Some future improvements include:

- Simulating individual household asset paths
- Comparing alternative borrowing constraints
- Adding wealth inequality statistics
- Including endogenous labor supply
- Computing the stationary PDF
