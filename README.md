# TFG_PINNs

This repository contains the code developed for Physics-Informed Neural Networks (PINNs) applied to solving differential equations and parameter estimation in physics.

It includes PINN implementations for the following systems:

Simple ordinary differential equation (ODE)

Time-independent Schrödinger equation (TISE)

Time-dependent Schrödinger equation (TDSE)

Hydrogen atom

For the simple ODE, each file includes the construction of the PINN class, its architecture, the training process, and evaluation.

For the TISE, TDSE, and 1D Hydrogen cases, the structure is as follows:

pinn.py: definition of the PINN class and architecture

adap.py: training with adaptive mesh

fija.py: training with fixed mesh

eval.py: evaluation of results

**Important note**:
For the TISE case, the states must be solved in ascending order (first n = 0, then n = 1, etc.) because orthogonality constraints require having the previous states available.
