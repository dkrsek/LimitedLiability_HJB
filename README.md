# PDEs for First-Best and Second-Best Contracting Problems

This project is a work in progress that numerically solves a Hamilton–Jacobi–Bellman partial differential equation arising from ongoing joint work with Dylan Possamaï and Stéphane Villeneuve.

The current numerical scheme might not be stable for all parameter choices, but it performs well for the scenarios defined in the main file PDE.py.

## Requirements

- Python 3.9+
- numpy
- matplotlib


## Running

In the main script `PDE.py`, select a scenario index from the list of predefined parameter sets.

## File Overview

- `first_best.py`		Computes the first-best solution of the control problem using an analytically derived closed-form solution
- `second_best.py`	Solves the second-best control problem using a finite difference method (explicit scheme)
- `PDE.py`			Main script to run both methods and generate plots
