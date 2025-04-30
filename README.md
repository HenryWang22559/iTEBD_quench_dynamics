# 1D Transverse Field Ising Model Simulation using iTEBD

This repository contains Python code for simulating the one-dimensional Transverse Field Ising Model (TFIM) using the infinite Time-Evolving Block Decimation (iTEBD) algorithm.

## Project Goal

The primary goal is to study the ground state properties and real-time dynamics (specifically, a quench of the transverse field) of the 1D TFIM with antiferromagnetic interaction (J=-1).

## Implementations

-   **`itebd_numpy_gs.py`**: A NumPy-based implementation of iTEBD to find the ground state of the TFIM.
-   **`itebd_torch_quench.py`**: A PyTorch-based implementation of iTEBD to simulate the real-time evolution following a quench of the transverse magnetic field.
-   **`itebd_torch_quench.ipynb`**: A Jupyter Notebook demonstrating the usage and results of the PyTorch-based quench simulation.
-   **`Exact_solution_TFIM_1D.py`**: Contains code related to the exact analytical solution of the 1D TFIM, likely used for benchmarking and comparison with the iTEBD results.

## Model Parameters

-   Interaction Strength (J): -1 (Antiferromagnetic)
-   Transverse Field (h): Varies (used for ground state search and quench protocols)

## Method

Infinite Time-Evolving Block Decimation (iTEBD) is employed to efficiently simulate the quantum dynamics of the infinite 1D chain.
