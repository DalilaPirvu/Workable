# Workable

### Bubble Tools:  algorithms for extracting observables in bubble nucleation

Author: Dalila M. Pîrvu

Date: 21 Aug 2024

Version: 1.0

### Overview:

This repository contains a collection of Python scripts, modules, and Jupyter notebooks designed to facilitate the analysis and visualization of bubble nucleation in real-time, field-theoretic simulations.

Utilize the tools in the bubble_codes/ folder to process your simulation data, calculate key physical quantities, and produce insightful visualizations.

### Repository Structure:

1. bubble_codes/

bubble_tools.py: The main script containing tools for processing, centering, and analyzing bubble data from simulations.

plotting.py: A module dedicated to plotting and formatting graphs, ensuring that visualizations are clear and consistent.

triage.py: A module for extracting simulation data and generating relevant data files for further analysis.

experiment.py: Contains simulation parameters, definitions of the physical model, and paths to necessary data files.

2. Jupyter Notebooks designed to demonstrate the usage of the functions provided in the bubble_tools.py script.

3. Simulation Codes

w/ and wevolvebubble/: These folders contain the codes necessary to run the simulations whose outputs are analyzed by the scripts in bubble_codes/. The output format of these simulations is assumed in the data extraction and analysis scripts. Additional details in triage.py and bubble_tools.py

### Run Simulations: 

Use the codes in the w/ and wevolvebubble/ folders to generate data compatible with the analysis scripts.

These represent modified versions of the original code "1d-Scalar" by J. Braden. For instructions on how to use it, as well as dependencies, see: https://github.com/jonathanbraden/1d-Scalar


### Additional dependencies:

Python 3.x

numpy, scipy: For numerical operations and scientific computing.

Jupyter: For running and interacting with the provided notebooks.
