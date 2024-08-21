"""

Bubble Tools Script
Author: Dalila M. Pîrvu
Date: 21 Aug 2024
Version: 1.0

Description: This Python script is designed to process and analyze data from simulations 
involving bubble nucleation, evolution, and decay in a relativistic field-theoretic context. 
The script provides tools to extract, manipulate, and analyze the simulation data, 
allowing users to extract real-time dynamical observables of bubble nucleation.

Key Features:

                             Data Extraction: Functions to load and reshape raw simulation data for further analysis.
      Bubble Identification and Manipulation: Tools to center bubbles within the simulation grid and to apply transformations such as boosting, which accounts for relativistic effects.
         Centre-of-mass Velocity Calculation: Methods to compute the center-of-mass (COM) velocity of bubbles.
Survival Probability and Decay Rate Analysis: Functions to calculate the survival probability of simulations in an ensemble over time and to fit decay rates.
                      Stacking and Averaging: Tools to stack bubbles across an ensemble.
                                    Plotting: Visualization options to inspect the results at various stages of the analysis.

Usage: This script is intended for researchers working on field-theoretic simulations of bubble nucleation and decay.
It provides a suite of tools to extract meaningful physical quantities from raw simulation data, 
analyze the behavior of bubbles under different conditions, and visualize the outcomes.

Dependencies:

Python 3.x
numpy: For numerical operations on arrays.
scipy: For scientific computing, including optimization, interpolation, and signal processing.
Custom modules: plotting and experiment, which should be present in the specified directory.

Documentation:

This script was documented with the assistance of ChatGPT-4 to ensure clarity 
and comprehensiveness, making it easier for others to understand and use.

"""


# Import necessary libraries
import os  # Imports the os module, providing functions for interacting with the operating system.
import sys  # Imports the sys module, providing access to system-specific parameters and functions.

# Adds a specific directory to the Python path, allowing the import of modules from that directory.
sys.path.append('/home/dpirvu/project/paper_prefactor/bubble_codes/')

import numpy as np  # Imports the numpy library, used for numerical operations on arrays and matrices.
import random  # Imports the random module, used for generating random numbers and performing random operations.

from functools import partial  # Imports partial from functools, which allows the creation of partial functions.
from itertools import cycle  # Imports cycle from itertools, which allows cycling through iterables indefinitely.

import scipy as scp  # Imports the scipy library under the alias scp, used for scientific and technical computing.
from scipy import optimize as sco, signal as scs, interpolate as si, ndimage
# Imports specific submodules from scipy:
# optimize as sco: for optimization algorithms
# signal as scs: for signal processing
# interpolate as si: for interpolation
# ndimage: for multidimensional image processing

from scipy.integrate import odeint  # Imports the odeint function from scipy.integrate, used for integrating ODEs.
from scipy.signal import find_peaks, peak_widths  # Imports functions for detecting and analyzing peaks in signals.
from scipy.interpolate import interp1d, interp2d  # Imports 1D and 2D interpolation functions from scipy.interpolate.
from scipy.ndimage import gaussian_filter, gaussian_filter1d  # Imports Gaussian filter functions for smoothing data.

from plotting import *  # Imports all functions and variables from the custom plotting module.
from experiment import *  # Imports all functions and variables from the custom experiment module.



##############################################################
####################### Data Extraction ######################
## Depends entirely on how the simulation data is formatted ## 
##############################################################
##############################################################




def extract_spec_data(nL, path_sim):
    # Open the simulation data file
    with open(path_sim) as file:
        # Read the first 8 lines of the file and store them in the params list
        params = [next(file) for x in range(8)]
        # Append the last line of the file to the params list
        params.append(file.readlines()[-1])

    # Load the data from the file, skipping the first 8 header lines
    data = np.genfromtxt(path_sim, skip_header=8)
    # Determine the shape of the data array (nNnT: total number of data points, nC: number of columns)
    nNnT, nC = np.shape(data)
    # Reshape each column of data into (nNnT // nL) rows and nL columns, then stack them into a new array
    reshaped_dat = np.asarray([np.reshape(data[:, cc], (nNnT // nL, nL)) for cc in range(nC)])
    # Return the parameters and the reshaped data
    return params, reshaped_dat


def save_txt_file(filename, data):
    # Open a file for writing
    with open(filename, 'w') as f:
        # Write each row of the data to the file, with values separated by tabs
        for row in data:
            f.write('\t'.join(map(str, row)) + '\n')


def extract_data(nL, path_sim):
    # Open the simulation data file
    with open(path_sim) as file:
        # Read the first 8 lines of the file and store them in the params list
        params = [next(file) for x in range(8)]
        # Append the last line of the file to the params list
        params.append(file.readlines()[-1])

    # Load the data from the file, skipping the first 8 header lines and the last line
    data = np.genfromtxt(path_sim, skip_header=8, skip_footer=1)
    # Determine the shape of the data array (nNnT: total number of data points, nC: number of columns)
    nNnT, nC = np.shape(data)
    # Reshape each column of data into (nNnT // nL) rows and nL columns, then stack them into a new array
    reshaped_dat = np.asarray([np.reshape(data[:, cc], (nNnT // nL, nL)) for cc in range(nC)])
    # Return the parameters and the reshaped data
    return params, reshaped_dat


def get_realisation(nL, nTimeMAX, path_sim):
    # Extract data and parameters from the simulation file
    params, data = extract_data(nL, path_sim)
    # Extract the tdecay value from the last parameter (assumed to be the last 13 characters of the last line)
    tdecay = int(params[-1][-13:])

    # Determine the outcome based on the tdecay value
    # If the decay time equals the total simulation time, the outcome is 2
    if tdecay >= nTimeMAX:
        outcome = 2
    # If the decay time is less than the simulation time, then a bubble was detected
    # Below, the check_decay function decides whether the bubble nucleated to the left or to the right of the metastable minimum
    # This is an old feature; not important at this stage
    else:
        # Check the decay condition on the last slice of data
        slice = data[0, -1, :]
        outcome = check_decay(slice)

    # Extract initial conditions, the full data array, and specific slices for prebubble and bubble
    # These are used to run statistics on the initial and late-time ensemble including power spectrum and energy conservation
    # Prebubble data is used to simulate the bubble nucleation without its history, and later to extract the observables
    initcond = data[:, 0, :]
    real = data[:, :, :]
    prebubble = data[:, -2, :]
    bubble = data[:, -1, :]
    # Return the tdecay value, outcome, initial conditions, full data, prebubble, and bubble
    return tdecay, outcome, initcond, real, prebubble, bubble


def extract_bubble_data(nL, path_sim):
    # Load the full data from the simulation file
    data = np.genfromtxt(path_sim)
    # Determine the shape of the data array (nNnT: total number of data points, nC: number of columns)
    nNnT, nC = np.shape(data)
    # Reshape each column of data into (nNnT // nL) rows and nL columns, then stack them into a new array
    reshaped_dat = np.asarray([np.reshape(data[:, cc], (nNnT // nL, nL)) for cc in range(nC)])
    # Return the reshaped data
    return reshaped_dat


def get_bubble_realisation(nL, path_sim):
    # Extract the reshaped bubble data from the simulation file
    data = extract_bubble_data(nL, path_sim)

    # Flatten the last 10 slices of the first column into a single array
    slice = data[0, -10:, :].flatten()
    # Determine the outcome based on the decay condition of the slice
    outcome = check_decay(slice)

    # If the outcome indicates a certain condition, invert the sign of the first two columns of data
    # specifically, if the bubble decays to the left/right of the metastable state, we reverse the field amplitude and momentum
    # This doubles the number of bubble observables that we can average over
    # This function CAN fail if the potential is changed; must do more testing
    if outcome == 1:
        data[0] = -data[0]
        data[1] = -data[1]
    # Return the modified data and the outcome
    return data, outcome


def check_decay(slice):
    '''' Warning: the values below are potential-specific. 
    Must change criterium and test if the potential is changed.'''

    # Count how many elements in the slice are greater than 10
    right_phi = np.sum(slice > 10.)
    # Count how many elements in the slice are less than -10
    left_phi = np.sum(slice < -10.)
    # Return 0 if right_phi is greater, otherwise return 1
    return 0 if right_phi >= left_phi else 1


def get_decay_time(real):
    # Extract the first component of the full realisation data
    # This should be the field
    fldreal = real[0, :, :]
    # Count the number of elements in each row that exceed an absolute value of 10
    ums = np.sum(np.abs(fldreal) > 10., axis=-1)
    # Find the index of the first row where any element exceeds the threshold
    return np.argwhere(ums > 0.)[0][0]






##############################################################
######################## Post-processing #####################
#### Functions below are helpful in extracting observables ###
##############################################################





########################
#### Random tools  #####
########################


# Define a lambda function to calculate rapidity given a velocity v
rapidity = lambda v: np.arctanh(v)

# Define a lambda function to calculate the Lorentz factor (gamma) for a given velocity v
gamma    = lambda v: (1. - v**2.)**(-0.5)

# Define a lambda function to return a fixed value based on the value of beta
# Essentially, this decides how many times a given simulation should be multiplied
# The faster the bubble, the more times it needs to be multiplied so that more of its history (i.e. the precursor) is recovered after the boost
fold     = lambda beta: 2 if 0.8 > np.abs(beta) > 0.7 else 3 if np.abs(beta) > 0.8 else 1

# Define a lambda function to add two velocities v1 and v2 using the relativistic velocity addition formula
addvels  = lambda v1,v2: (v1 + v2) / (1. + v1*v2)


# Define a function to calculate the time and position in a boosted frame of reference
def coord_pair(tt, xx, beta, ga, c):
    ''' WARNING: Factors of c not tested throughtout; please ensure speed of light in simulations output is 1.'''
    # Calculate the transformed time coordinate
    t0 = (tt + beta * xx / c) * ga
    # Calculate the transformed spatial coordinate
    x0 = (xx + beta * tt) * ga
    # Return the transformed time and spatial coordinates
    return t0, x0


# Define a function to calculate the total velocity from a list of velocities using relativistic addition
def get_totvel_from_list(vels):
    # Initialize the total velocity to zero
    totvel = 0.
    # Iterate over each velocity in the list
    for ii in vels:
        # Update the total velocity by adding the current velocity to the accumulated total
        totvel = addvels(ii, totvel)
    # Return the total velocity
    return totvel


# Define a function to center the bubble in the simulation by rolling the array
def centre_bubble(real, tdecay):
    # Get the shape of the real array (nC: data columns in simulation output = field and momentum, nT: time steps, nN: lattice points)
    nC, nT, nN = np.shape(real)
    # Calculate a temporal offset to trim the array if necessary
    # Warning: only truncate history if bubbles are expected to be slow; for fast bubbles, adjust/remove this condition
    tamp = max(0, nT - 2 * nN)
    # Trim the data array by removing the initial time steps if needed
    real = real[:, tamp:]
    
    # Apply the rolling operation (twice to be sure) to center the bubble
    for _ in range(2):  
        # Find the critical slice at the decay time
        critslice = np.abs(real[0, tdecay, :])
        # Find the location of the nucleation event
        x_centre = int(round(np.mean(np.argwhere(critslice > 1.))))
        # Roll the array to center the bubble at the midpoint of the spatial axis
        real = np.roll(real, nN // 2 - x_centre, axis=-1)
    # Return the centered bubble array
    return real


# Define a function to multiply the bubble data array with the goal to unfold its causal tail from periodic boundary conditions
# Causal tail = precursor OR onset of hyperbolic expansion
def multiply_bubble(simulation, phi_init, normal, nL, light_cone=1, vCOM=0.9):
    '''
    NOTE:
    After multiplying, one should apply another bubble centering operation.
    Also, only multiply after centering!
    '''
    # If the center-of-mass velocity is negative, flip the bubble along the spatial axis
    if vCOM < 0:
        simulation = simulation[:, :, ::-1]
    # Get the shape of the data (C: data columns, T: time steps, N: spatial points)
    C, T, N = np.shape(simulation)
    
    # Tile the bubble array according to the fold factor determined by the center-of-mass velocity
    simulation = np.asarray([np.tile(simulation[col], fold(vCOM)) for col in range(C)])
    
    # Get the new shape of the tiled bubble array (TT: total time steps, NN: total spatial points)
    TT, NN = np.shape(simulation[0])
    
    # Calculate the mean value of the first component of the bubble data array
    mn = np.mean(simulation[0])
   
    # So far: we have multiplied the simulation data along the x axis. Now we have #fold bubbles side by side
    # We only want the precursor to be unfolded, so we must cover up the actual expanding bubble
    # Having more than one expanding bubble will ruin the rest of the deboosting procedure
    # We simply cover the duplicate bubbles with a constant field value:

    # Iterate over each time step in the tiled array
    # light_cone should always be 1; this is the speed of light on the lattice.
    for t in range(TT):
        # Calculate spatial boundaries of the bubble we want to preserve
        a, b = int((TT - t) / light_cone) + N, int((TT - t) / light_cone / 3.) - N // 4
        # Generate arrays of spatial points beyond which we want to mask the data
        x1, x2 = np.arange(a, NN), np.arange(b)
        x1, x2 = x1 - a, x2 - (b - NN)
        # Apply mask on the field at the calculated spatial points
        for x in np.append(x1, x2):
            if 0 <= x < NN:
                simulation[0, t, x] = mn
    # If the center-of-mass velocity is negative, flip the bubble back along the spatial axis to restore original
    if vCOM < 0:
        simulation = simulation[:, :, ::-1]
    # Return the modified bubble array
    return simulation



# Define a function to count the number of points in a simulation exceeding a threshold at each time step
def bubble_counts_at_fixed_t(bubble, thresh):
    # Return the count of points greater than or equal to the threshold
    return np.count_nonzero(bubble >= thresh, axis=1)

# Define a function to count the number of points in a simulation exceeding a threshold at each spatial location
def bubble_counts_at_fixed_x(bubble, thresh):
    # Return the count of points greater than or equal to the threshold
    return np.count_nonzero(bubble >= thresh, axis=0)

# Define a function to reflect the simulation against the metastable value at initialization
def reflect_against_equil(bubble, phi_init):
    return np.abs(bubble - phi_init) + phi_init


# Define a function to find the nucleation center of a bubble given certain criteria
def find_nucleation_center(bubble, phi_init, crit_thresh, crit_rad):
    ''' 
    # WARNING: the best way to find a nucleation center WILL depend on the bubble properties i.e. potential shape!
    # Must be tested out if potential is changed. The above methods simply worked for me.
    # Test results with an alternative method to ensure consistency and absence of huge systematics
    # Verify by eye!
    '''
    # Get the shape of the bubble array (T: time steps, N: spatial points)
    T, N = np.shape(bubble)
    # Calculate the number of bubble points exceeding the critical threshold at each time step
    bubble_counts = bubble_counts_at_fixed_t(bubble, crit_thresh)
    # Find the time step where the bubble count is closest to the critical radius
    t0 = np.argmin(np.abs(bubble_counts - crit_rad))

    # Calculate the number of bubble points exceeding the critical threshold across the spatial axis (over a limited time interval)
    bubble_counts = bubble_counts_at_fixed_x(bubble[:int(min(t0 + 5 * crit_rad, T))], crit_thresh)
    # Find the spatial location where the bubble count is maximum
    x0 = np.argmax(bubble_counts)
    # Return the time and spatial coordinates of the nucleation center
    return min(T - 1, t0), min(N - 1, x0)


# Define an alternative function to find the nucleation center with a different approach
def find_nucleation_center2(bubble, phi_init, crit_thresh, crit_rad):
    # Get the shape of the bubble array (T: time steps, N: spatial points)
    T, N = np.shape(bubble)
    # Calculate the number of bubble points exceeding the critical threshold at each time step
    bubble_counts = bubble_counts_at_fixed_t(bubble, crit_thresh)
    # Find the time step where the bubble count is closest to the critical radius
    t0 = np.argmin(np.abs(bubble_counts - crit_rad))

    # Extract the bubble slice at time t0
    bub = bubble[t0]
    # Find the spatial locations where the bubble exceeds the critical threshold
    cds = np.argwhere(bub > crit_thresh).flatten()
    # If the number of such locations is less than the critical radius, move forward in time until it is
    while len(cds) < crit_rad // 5:
        t0 += 1
        bub = bubble[t0]
        cds = np.argwhere(bub >= crit_thresh).flatten()
    # Calculate the center position as the mean of the critical positions
    x0 = (cds[0] + cds[-1]) // 2
    # Return the time and spatial coordinates of the nucleation center
    return min(T - 1, t0), x0



# Define a function to find the time at which the bubble width reaches a maximum
# light_cone = c = speed of light; ensure this is 1!
def find_t_max_width(bubble, light_cone, phi_init, tv_thresh, crit_rad):
    nT, nN = np.shape(bubble)
    refl_bubble = reflect_against_equil(bubble, phi_init)

    # Calculate the number of bubble points exceeding the threshold at each time step
    bubble_counts = bubble_counts_at_fixed_t(refl_bubble, tv_thresh)
    # Calculate the differences in bubble counts between consecutive time steps
    bubble_diffs = bubble_counts[1:] - bubble_counts[:-1]

    # Reverse the differences and find the time steps where the difference is significant
    # Bubble expands symmetrically at speed 1 against its center; so every time step it should gain light_cone * 2 in width
    tmax = np.argwhere(bubble_diffs[::-1] >= light_cone * 2).flatten()
    # Find the first instance where two consecutive time steps meet this condition
    out = next((ii for ii, jj in zip(tmax[:-1], tmax[1:]) if jj - ii == 1), 1)
    # Return the time step corresponding to the maximum width
    return nT - out


# Functions below no longer in use

# We want to find the point where the bubble has reached its maximum spatial extent in expanding on the lattice
# Define a function to find the index where the order of elements in an array changes
def find_order_changes(arr):
    changes = []  # Initialize a list to store indices where changes occur
    decreasing = False  # Initialize a flag to track if the array is decreasing

    for i in range(len(arr) - 1):
        # If the current element is less than the next, and the array was previously decreasing
        if arr[i] < arr[i + 1]:
            if decreasing:
                # Add the index to the list of changes and break the loop
                changes.append(i)
                break
        # If the current element is greater than the next, set the decreasing flag
        elif arr[i] > arr[i + 1]:
            if not decreasing:
                decreasing = True
            # If a change was recorded and the previous change is at the current index, remove it
            if changes and changes[-1] == i:
                changes.pop()
        # If any changes were recorded, break the loop
        if len(changes) != 0:
            break
        # Skip if the elements are equal (not considered a change in trend)
    # If no changes were found, set the change to index 0
    if changes == []:
        changes = [0]
    # Return the index of the first change
    return changes[0]

# Define a function to find the index of the last zero in an array followed by two non-positive values
def find_final_zero_index(arr):
    for i in range(len(arr) - 2):
        # If the current element is zero and the next two elements are non-positive
        if arr[i] == 0:
            if arr[i + 1] <= 0 and arr[i + 2] <= 0:
                # Return the index of the second non-positive element
                return i + 2
    # Return 1 if no such sequence is found
    return 1




########################################
#### Tools for decay rate fitting  #####
########################################

# Define a function for linear fitting of decay times within a specified range
def lin_fit_times(times, num, tmin, tmax):
    """
    Given a collection of decay times, do a linear fit to
    the logarithmic survival probability between given times.

    Input:
      times : array of decay times
      num   : original number of samples
      tmin  : minimum time to fit inside
      tmax  : maximum time to fit inside

    Output:
      Coefficients (slope and intercept) of the linear fit.
    """
    # Sort the decay times in ascending order
    t = np.sort(times)
    # Compute the survival probability for the given times and number of samples
    p = survive_prob(times, num)
    # Find the indices of times that fall within the specified range [tmin, tmax]
    ii = np.where((t > tmin) & (t < tmax))
    # Perform a linear fit (degree 1 polynomial) to the logarithm of the survival probabilities
    return np.polyfit(t[ii], np.log(p[ii]), deg=1)


# To do: Debug more to ensure all offsets are correct.
# I've done a first go through and I think they're ok

# Define a function to calculate the survival probability as a function of time
def survive_prob(t_decay, num_samp):
    """
    Return the survival probability as a function of time.

    Input:
      t_decay  : Decay times of trajectories
      num_samp : Total number of samples in Monte Carlo

    Note: Since some trajectories may not decay, t_decay.size isn't always num_samp.

    Output:
      prob     : Survival probability array.

    These can be plotted as plt.plot(t_sort, prob) to get a survival probability graph.
    """
    # Calculate the fraction of remaining (non-decayed) samples
    frac_remain = float(num_samp - t_decay.size) / float(num_samp)
    # Calculate the survival probability at each time point
    prob = 1. - np.linspace(1. / num_samp, 1. - frac_remain, t_decay.size, endpoint=True)
    # Return the array of survival probabilities
    return prob


# Define a function to generate a line with a given slope and offset
def get_line(dataset, slope, offset):
    # Return the dataset scaled by the slope and shifted by the offset
    return dataset * slope + offset

# Define a function to compute the survival function (fraction remaining) over time
def f_surv(times, ntot):
    # Return an array representing the fraction of samples surviving up to each time point
    return np.array([1. - len(times[times <= ts]) / ntot for ts in times])




########################################
#### Tools for de-boosting bubbles  ####
########################################


# Define a function to generate a hyperbolic tangent profile with asymmetry
def retired_tanh_profile(x, r0L, r0R, vL, vR, dr, a):
    # Calculate the widths of the left and right sides based on the velocities and dr
    wL, wR = dr/gamma(vL), dr/gamma(vR)
    # Return the combined tanh profile scaled by the amplitude 'a'
    return ( np.tanh( (x - r0L)/wL ) + np.tanh( (r0R - x)/wR ) ) * a


# Define a function to generate a hyperbolic tangent profile without additional parameters
def tanh_profile(x, r0L, r0R, vL, vR):
    return ( np.tanh( (x - r0L)/vL ) + np.tanh( (r0R - x)/vR ) )


# Define a function to fit a hyperbolic tangent profile to given data
def get_profile_bf(xlist, phibubble, prior):
    # Define the bounds for the fitting parameters
    # Physically motivated bounds
    bounds = ((xlist[0], 0., 0., 0., xlist[0], -1.), (0., xlist[-1], 1., 1., xlist[-1], 1.))
    # Perform curve fitting using the retired_tanh_profile function
    tanfit, _ = sco.curve_fit(retired_tanh_profile, xlist, phibubble, p0=prior, bounds=bounds)
    # Return the fitted parameters
    return tanfit


# Define a function to fit a right-moving hyperbola to data
def hypfit_right_mover(tt, rr):
    # Define a lambda function for a right-moving hyperbola
    hyperbola = lambda t, a, b, c: np.sqrt(c + b * t + t**2.) + a
    try:
        # Set the initial guess for the fitting parameters
        prior = (float(min(rr)), float(tt[np.argmin(rr)]), 1e3)
        # Perform curve fitting using the hyperbola function
        fit, _ = sco.curve_fit(hyperbola, tt, rr, p0=prior)
        # Generate the fitted trajectory using the fitted parameters
        traj = hyperbola(tt, *fit)
        # Return the fitted trajectory
        return traj
    except:
        # Return an empty list if fitting fails
        return []


# Define a function to fit a left-moving hyperbola to data
def hypfit_left_mover(tt, ll):
    hyperbola = lambda t, d, e, f: -np.sqrt(f + e * t + t**2.) + d
    try:
        prior = (float(max(ll)), float(tt[np.argmax(ll)]), 1e3)
        fit, _ = sco.curve_fit(hyperbola, tt, ll, p0=prior)
        traj = hyperbola(tt, *fit)
        return traj
    except:
        return []



# Define a function to calculate velocities based on fitted trajectories
def get_velocities(rrwallfit, llwallfit):
    # Calculate the gradient (velocity) of the right- and left-moving walls
    uu = np.gradient(rrwallfit)  # Wall traveling with the center of mass (COM)
    vv = np.gradient(llwallfit)  # Wall traveling against the COM

    # Clip the velocities to prevent them from exceeding the speed of light (1) or becoming NaN
    uu[np.abs(uu) >= 1.] = np.sign(uu[np.abs(uu) >= 1.]) * (1. - 1e-15)
    vv[np.abs(vv) >= 1.] = np.sign(vv[np.abs(vv) >= 1.]) * (1. - 1e-15)
    # Replace NaN values in uu with a value close to 1, determined by the sign of the last value in vv
    uu[np.isnan(uu)] = np.sign(vv[-1]) * (1. - 1e-15)
    # Replace NaN values in vv with a value close to 1, determined by the sign of the last value in uu
    vv[np.isnan(vv)] = np.sign(uu[-1]) * (1. - 1e-15)

    # Starting from asymmetric left- and right-wall velocities, below we find the symmetric wall velocity + COM velocity
   
    # Calculate the center of mass velocity
    aa = (1. + uu * vv - np.sqrt((-1. + uu**2.) * (-1. + vv**2.))) / (uu + vv)
    # Calculate the instantaneous wall velocity
    bb = (-1. + uu * vv + np.sqrt((-1. + uu**2.) * (-1. + vv**2.))) / (-uu + vv)
    # Return the calculated velocities
    return uu, vv, aa, bb



# Define a function to find the center of mass (COM) velocity in a simulation
def find_COM_vel(real, fldamp, winsize, nL, light_cone, phi_init, tv_thresh, crit_thresh, crit_rad, plots=False):
    '''
    The function calculates the center of mass (COM) velocity of a bubble within a simulation. 
    It does so by analyzing the field data, locating the nucleation center, and then computing the velocities for different field amplitudes.

    Parameters:

        real: The simulation data array containing field+momentum values over time and space.
      fldamp: A list of field amplitudes to consider for velocity calculation. Delimitates the walls and determines the bubble extent.
     winsize: The size of the window to use around the nucleation center for analysis.
          nL: A parameter influencing the range of data considered around the time of maximum width.
  light_cone: Speed of light. Must be 1. Not tested for any other value.
    phi_init: The initial average field value in the false vacuum
       plots: A boolean flag that enables plotting for visualizing the process.
     tv_thresh, crit_thresh, crit_rad: Thresholds and critical radius used in finding nucleation centers and calculating velocities.

    Workflow:

        Step 1: Extract and truncate the field data to focus on the region only up to maximum width.
                (recall that boosting distorts the late time bubble expansion and artifacts must be removed; bubble must always grow bigger with time)
        Step 2: Identify the nucleation center within this region.
        Step 3: Narrow down the region of interest based on a window centered around the nucleation center.
        Step 4: Calculate the COM velocity for each field amplitude.
        Step 5: Optionally, plot the results for visual inspection.
        Step 6: Return the mean and variance of the calculated COM velocities.
    '''

    # Extract the dimensions (nC: components, nT: time steps, nN: spatial points)
    nC, nT, nN = np.shape(real)
    # Use only the first component (assumed to be the field of interest)
    real = real[0]

    # Find the time corresponding to the maximum width of the bubble
    t_maxwid = find_t_max_width(real, light_cone, phi_init, tv_thresh, crit_rad)

    # Truncate the simulation data to only include up to the time of maximum width
    real = real[:t_maxwid]
    # Calculate the spatial edge margin after truncation
    edge = np.abs(nT - t_maxwid)
    # Calculate the minimum time index to start considering data for vCOM extraction, before nucleation
    # The bubble should not be wrapping around the lattice boundaries. Maximum bubble expansion time = nLat//2
    mint = int(np.max((0, t_maxwid - nL // 2)))

    # Extract the region of interest in the truncated data
    wth = real[mint:t_maxwid, edge:(nN - edge)]
    # Update the dimensions of the region of interest
    nT, nN = np.shape(wth)
    # Find the initial nucleation center in the region of interest
    t_centre0, x_centre = find_nucleation_center(wth, phi_init, crit_thresh, crit_rad)

    # If plotting is enabled, display the region of interest
    if plots:
        t, x = np.linspace(-t_centre0, nT-1-t_centre0, nT), np.linspace(-x_centre, nN-1-x_centre, nN)
        simple_imshow([wth], x, t, title=r'wth', contour=False, ret=False)

    # Recenter coordinate grid around the nucleation location (x,t)
    # Adjust the x_centre by adding the edge margin
    x_centre += edge
    # Adjust the time center by adding the starting offset
    t_centre = t_centre0 + max(t_maxwid - nL // 2, 0)
    # Define the time and space limits for the window of interest around the nucleation center
    tl_stop, tr_stop = int(max(0, t_centre - winsize)), int(min(nT, t_centre + winsize // 2))
    xl_stop, xr_stop = int(max(0, x_centre - winsize)), int(min(nN, x_centre + winsize))

    # Extract the simulation data within the defined window of interest
    simulation = real[tl_stop:tr_stop, xl_stop:xr_stop]
    # Update the dimensions of the windowed simulation data
    nT, nN = np.shape(simulation)
    # Find the nucleation center within the windowed data
    tcen, xcen = find_nucleation_center(simulation, phi_init, crit_thresh, crit_rad)

    # If plotting is enabled, display the windowed region of interest
    if plots:
        t, x = np.linspace(-tcen, nT-1-tcen, nT), np.linspace(-xcen, nN-1-xcen, nN)
        simple_imshow([simulation], x, t, title=r'wth 2 process', contour=False, ret=False)

    # Goal: up to now, the code should have extracted a square window around the bubble nucleation site at (0,0)

    # Initialize an array to store the calculated velocities for each field amplitude
    betas = np.zeros((len(fldamp)))
    # Iterate over each field amplitude size to calculate the COM velocity
    for vv, v_size in enumerate(fldamp):
        # Enable velocity plotting for every second field amplitude if plotting is enabled
        vel_plots = (True if (vv % 2 == 0 and plots) else False)
        # Calculate the COM velocity and store it in the betas array
        betas[vv] = get_COM_velocity(simulation, phi_init, crit_thresh, crit_rad, v_size, tcen, xcen, vel_plots)

    # If plotting is enabled, plot the calculated velocities against the field amplitudes
    if plots:
        fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
        plt.plot(fldamp, betas, marker='o', ms=3)
        beautify(ax, times=-70, ttl=r'${{ \rm Mean \ }} v={:.2f}$'.format(np.nanmean(betas))); plt.show()

    # Return the mean and variance of the calculated velocities
    return np.nanmean(betas), np.nanvar(betas)
 
    

def get_COM_velocity(simulation, phi_init, crit_thresh, crit_rad, amp, tcen, xcen, plots=False):
    """
    Calculate the center-of-mass (COM) velocity of a bubble in the simulation by analyzing the movement of the 
    bubble's walls over time.

    Parameters:
    simulation: 2D numpy array representing the field values over time (nT) and space (nN).
    phi_init: The initial average field value in the false vacuum.
    crit_thresh: The threshold used to identify the bubble walls.
    crit_rad: The critical radius used in the analysis of the bubble.
    amp: The amplitude threshold used to track the movement of the bubble walls.
    tcen: The time center around which the analysis is performed.
    xcen: The spatial center around which the analysis is performed.
    plots: A boolean flag that enables plotting for visualizing the process.

    Returns:
    vCOM: The calculated center-of-mass velocity of the bubble.
    """

    # Get the dimensions of the simulation data (nT: time steps, nN: spatial points)
    nT, nN = np.shape(simulation)
    # Initialize some variables for later use
    data_list, prior, target = [], None, nN / 2.

    # Iterate through time steps in reverse order
    for tt in reversed(range(nT)):
        # Select field slice
        slice = simulation[tt]

        try:
            # Determine the target position of the bubble wall based on the amplitude threshold
            target = int(np.round(np.nanmean(np.argwhere(slice > amp))))
        except:
            # If no field value above threshold is found, stop process
            break

        coord_list = np.arange(nN) - target

        try:
            # Fit a profile to the bubble slice and store the fitting parameters
            prior = get_profile_bf(coord_list, slice, prior)
            # Quantities below will be used as prior to find the wall location at the next time step
            # Without a prior, the wall trajectory data is much noisier and the fits tend to be worse
            r0L, r0R, vL, vR, _, _ = prior

            curve = retired_tanh_profile(coord_list, *prior)
            # save bubble wall trajectories, as identified in the fits
            data_list.append([tt, r0L + target, r0R + target])

            if plots and False:
                if tt % 50 != 10:
                    continue
                print(prior)
                fig, ax = plt.subplots(1, 1, figsize=(3, 3))
                plt.plot(coord_list, retired_tanh_profile(coord_list, *prior), 'r')
                plt.plot(coord_list, slice, 'bo', ms=1)
                plt.plot(coord_list, curve, 'go', ms=1)
                plt.axhline(amp, ls=':', color='k')
                plt.title(r'$t={:.1f}$'.format(tt))
                beautify(ax, times=-70)
                plt.show()

        except:
            continue

    # Get velocities from derivatives of trajectories
    try:
        # Reverse the order of data_list for proper time progression
        data_list = np.array(data_list)[::-1]
        ttwallfit, ll, rr = data_list[:, 0], data_list[:, 1] - xcen, data_list[:, 2] - xcen

        # Fit walls to a hyperbolic trajectory
        llwallfit = hypfit_left_mover(ttwallfit, ll)
        rrwallfit = hypfit_right_mover(ttwallfit, rr)

        # Calculate velocities
        uu, vv, aa, bb = get_velocities(rrwallfit, llwallfit)
        # select COM velocity from arrays
        indix = np.nanargmin(np.abs(uu - vv))
        vCOM = aa[indix]

        if plots:
            fig, ax = plt.subplots(1, 2, figsize=(7, 4))
            ext = np.array([-xcen, nN - xcen, -tcen, nT - tcen])
            im0 = ax[0].imshow(simulation, interpolation='antialiased', extent=ext, origin='lower', cmap='tab20c', aspect='auto')

            ax[0].plot(rr, (ttwallfit - tcen), color='b', ls='-', lw=1, label=r'$\rm rr$')
            ax[0].plot(ll, (ttwallfit - tcen), color='g', ls='-', lw=1, label=r'$\rm ll$')

            try:
                ax[0].plot(rrwallfit, (ttwallfit - tcen), color='b', ls=':', lw=1, label=r'$\rm rr \ hyp \ fit$')
                ax[0].plot(llwallfit, (ttwallfit - tcen), color='g', ls=':', lw=1, label=r'$\rm ll \ hyp \ fit$')

                ax[1].plot((ttwallfit - tcen), uu, color='b', ls='-', lw=1, label=r'$\rm wall \, travelling \, with \, COM$')
                ax[1].plot((ttwallfit - tcen), vv, color='g', ls='-', lw=1, label=r'$\rm wall \, travelling \, against \, COM$')
                ax[1].plot((ttwallfit - tcen), aa, color='r', ls='--', lw=1, label=r'$v_{\rm COM}(t)$')
                ax[1].plot((ttwallfit - tcen), bb, color='orange', ls=':', label=r'$v_{\rm walls}(t)$')
                ax[1].plot((ttwallfit - tcen), -bb, color='orange', ls=':', label=r'$v_{\rm walls}(t)$')
                ax[1].plot((ttwallfit - tcen)[indix], vCOM, 'ko', ms=3)
            except:
                ax[0].plot(0., 0., color='yellow', marker='o', ms=3, label=r'$\rm failed \ fit$')

            cbar = plt.colorbar(im0, ax=ax[0])
            cbar.ax.setTitle(r'$\bar{\phi}$')
            ax[0].set(ylabel=r'$t$')
            ax[0].set(xlabel=r'$r$')
            ax[1].set(ylabel=r'$t$')
            ax[1].set(xlabel=r'$v(t)$')
            beautify(ax, times=-70, ttl=r'$\phi={:.2f}$'.format(amp))
            plt.tight_layout()
            plt.show()

    except:
        return 'nan'

    return vCOM



def boost_bubble(simulation, t0, x0, vCOM, c=1):
    '''
    The function boost_bubble applies a Lorentz boost to a simulated bubble, transforming its coordinates 
    according to the given center-of-mass velocity (vCOM). This is useful in relativistic simulations 
    where bubbles expand and move at relativistic speeds.

    Parameters:
    simulation: 3D numpy array representing the field values over time and space.
                The shape of the array is (C, T, N), where:
                - C is the number of components (e.g., field components),
                - T is the number of time steps,
                - N is the number of spatial points.
    t0: The time offset used to center the time axis.
    x0: The spatial offset used to center the spatial axis.
    vCOM: The center-of-mass velocity at which the bubble is boosted.
    c: The speed of light in the simulation, set to 1.

    Returns:
    t: 1D numpy array representing the transformed time coordinates after boosting.
    x: 1D numpy array representing the transformed spatial coordinates after boosting.
    rest_bubble: 3D numpy array of the same shape as the input simulation, containing the boosted bubble field values.
    '''

    # Get the dimensions of the simulation data
    C, T, N = np.shape(simulation)

    # Calculate the boost factor (beta) and the corresponding Lorentz factor (gamma)
    beta = vCOM / c
    ga = gamma(beta)

    # Create the old grid centered around the time (t0) and spatial (x0) offsets
    t, x = np.linspace(-t0, T-1-t0, T), np.linspace(-x0, N-1-x0, N)

    # Initialize an array to store the boosted bubble field values
    rest_bubble = np.zeros(np.shape(simulation))

    # Loop over each component of the simulation (e.g., different field components)
    for col, element in enumerate(simulation):
        # Interpolate the 2D field component (element) over the original time-space grid
        g = interp2d(x, t, element, kind='cubic', bounds_error=True, fill_value=0.)

        # Loop over each time value in the time grid
        for tind, tval in enumerate(t):
            # Calculate the transformed (boosted) time and spatial coordinates
            tlensed, xlensed = coord_pair(tval, x, beta, ga, c)
            # Interpolate the field values onto the new (boosted) coordinates
            interpolated = si.dfitpack.bispeu(g.tck[0], g.tck[1], g.tck[2], g.tck[3], g.tck[4], xlensed, tlensed)[0]
            # Store the interpolated (boosted) field values in the rest_bubble array
            rest_bubble[col, tind, :] = interpolated

    # Return the transformed time, spatial grids, and the boosted bubble field values
    return t, x, rest_bubble





#####################################
#### Tools for stacking bubbles  ####
#####################################


def quadrant_coords(real, phi_init, crit_thresh, crit_rad, maxwin, plots=False):
    """
    Identifies the four quadrants (upper-right, upper-left, lower-right, lower-left) around the nucleation
    center of a bubble in the simulation data and returns these quadrants as separate arrays.

    Parameters:
    real: 3D numpy array representing the simulation data with dimensions (components, time, space).
    phi_init: The initial average field value in the false vacuum.
    crit_thresh: The threshold used to identify the bubble walls.
    crit_rad: The critical radius used in the analysis of the bubble.
    maxwin: The maximum window size to consider around the nucleation center.
    plots: A boolean flag that enables plotting for visualizing the process.

    Returns:
    upright_quad: 3D numpy array representing the upper-right quadrant of the bubble.
    upleft_quad: 3D numpy array representing the upper-left quadrant of the bubble.
    lowright_quad: 3D numpy array representing the lower-right quadrant of the bubble.
    lowleft_quad: 3D numpy array representing the lower-left quadrant of the bubble.
    """

    # Copy and smooth the first component of the real data
    bub = np.copy(np.abs(real[0]))
    bub = gaussian_filter(bub, 0.5, mode='nearest')
    bub[bub > crit_thresh] = crit_thresh

    # Find the nucleation center in the smoothed bubble data
    tcen, xcen = find_nucleation_center(bub, phi_init, crit_thresh, crit_rad)
    nT, nN = np.shape(bub)

    # Define the boundaries for the quadrants based on the nucleation center and maxwin
    aa, bb = max(0, xcen-maxwin), min(nN, xcen+maxwin+1)
    cc, dd = max(0, tcen-maxwin), min(nT, tcen+maxwin+1)

    # Define the coordinate ranges for the four quadrants
    aaL, bbL = np.arange(aa, xcen), np.arange(xcen, bb)
    ccL, ddL = np.arange(cc, tcen), np.arange(tcen, dd)

    # Extract the four quadrants from the real data
    ddd, bbb = np.meshgrid(ddL, bbL, sparse='True')
    upright_quad = real[:, ddd, bbb]
    ddd, aaa = np.meshgrid(ddL, aaL, sparse='True')
    upleft_quad = real[:, ddd, aaa]
    ccc, bbb = np.meshgrid(ccL, bbL, sparse='True')
    lowright_quad = real[:, ccc, bbb]
    ccc, aaa = np.meshgrid(ccL, aaL, sparse='True')
    lowleft_quad = real[:, ccc, aaa]

    # Plot the quadrants if plotting is enabled
    if plots:
        if len(bbL) > 0 and len(aaL) > 0 and len(ccL) > 0 and len(ddL) > 0:
            a1, a2, a3, a4 = np.copy(upright_quad[0]), np.copy(upleft_quad[0]), np.copy(lowright_quad[0]), np.copy(lowleft_quad[0])
            avec = [a1, a2, a3, a4]
            for ai, amat in enumerate(avec):
                ati = np.abs(amat)
                ati = gaussian_filter(ati, 0.5, mode='nearest')
                ati[ati > crit_thresh] = crit_thresh
                avec[ai] = ati

            fig, ax = plt.subplots(2, 2, figsize=(5,5))
            ext00, ext01 = [ddL[0],ddL[-1],bbL[0],bbL[-1]], [ddL[0],ddL[-1],aaL[0],aaL[-1]]
            ext10, ext11 = [ccL[0],ccL[-1],bbL[0],bbL[-1]], [ccL[0],ccL[-1],aaL[0],aaL[-1]]
            ax[0,0].imshow(avec[3], interpolation='none', extent=ext11, aspect='equal', cmap='tab20c')
            ax[0,1].imshow(avec[1], interpolation='none', extent=ext01, aspect='equal', cmap='tab20c')
            ax[1,0].imshow(avec[2], interpolation='none', extent=ext10, aspect='equal', cmap='tab20c')
            ax[1,1].imshow(avec[0], interpolation='none', extent=ext00, aspect='equal', cmap='tab20c')
            for aa in ax.flatten():
                aa.set_xticklabels([]); aa.set_yticklabels([])
            beautify(ax, times=-70); plt.tight_layout(); plt.show()
        else:
            print('Failed')

    return upright_quad, upleft_quad, lowright_quad, lowleft_quad


def stack_bubbles(data, maxwin, phi_init, crit_thresh, crit_rad, plots=False):
    """
    Stacks the quadrants of bubbles from multiple simulations into four separate lists (one for each quadrant).

    Parameters:
    data: List of tuples, where each tuple contains a simulation identifier and the corresponding real data array.
    maxwin: The maximum window size to consider around the nucleation center.
    phi_init: The initial average field value in the false vacuum.
    crit_thresh: The threshold used to identify the bubble walls.
    crit_rad: The critical radius used in the analysis of the bubble.
    plots: A boolean flag that enables plotting for visualizing the process.

    Returns:
    upright_stack: List of 3D numpy arrays, each representing the upper-right quadrant from different simulations.
    upleft_stack: List of 3D numpy arrays, each representing the upper-left quadrant from different simulations.
    lowright_stack: List of 3D numpy arrays, each representing the lower-right quadrant from different simulations.
    lowleft_stack: List of 3D numpy arrays, each representing the lower-left quadrant from different simulations.
    """

    # Initialize empty lists to store the stacked quadrants
    upright_stack, upleft_stack, lowright_stack, lowleft_stack = ([] for ii in range(4))

    # Loop through each simulation in the data
    for sim, real in data:

        # Plot the bubble if plotting is enabled
        if plots:
            bub = np.copy(np.abs(real[0]))
            bub = gaussian_filter(bub, 0.5, mode='nearest')
            bub[bub > crit_thresh] = crit_thresh
            nT, nN = np.shape(bub)

            tcen, xcen = find_nucleation_center(bub, phi_init, crit_thresh, crit_rad)
            tl, tr = max(0, tcen-maxwin), min(nT, tcen+maxwin+1)
            xl, xr = max(0, xcen-maxwin), min(nN, xcen+maxwin+1)

            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            ext = [xl, xr, tl, tr]
            plt.imshow(bub[tl:tr,xl:xr], interpolation='none', extent=ext, aspect='equal', origin='lower', cmap='tab20c')
            plt.plot(xcen, tcen, 'bo')
            plt.xlabel('x'); plt.ylabel('t')
            beautify(ax, times=-70, ttl=r'${{\rm Sim}} = {:.0f}$'.format(sim)); plt.show()

        # Get the quadrants for the current simulation
        ur, ul, lr, ll = quadrant_coords(real, phi_init, crit_thresh, crit_rad, maxwin, plots)

        # Check if the quadrants are large enough; if not, skip the simulation
        bool = True
        for ii in [ur, ul, lr, ll]:
            if np.shape(ii)[1] <= maxwin // 5 or np.shape(ii)[2] <= maxwin // 5:
                bool = False
                break

        # If the quadrants are large enough, add them to the respective stacks
        if bool:
            upright_stack.append(ur)
            upleft_stack.append(ul)
            lowright_stack.append(lr)
            lowleft_stack.append(ll)

    return upright_stack, upleft_stack, lowright_stack, lowleft_stack



def average_stacks(data, winsize, normal, plots=False):
    """
    Averages the stacked quadrants from multiple simulations to create an averaged bubble and its error matrix.

    Parameters:
    data: List of lists, where each list contains the stacked quadrants from different simulations.
    winsize: The size of the window used to extract the quadrants.
    normal: A normalization parameter (not explicitly used in the function).
    plots: A boolean flag that enables plotting for visualizing the process.

    Returns:
    av_mat: 3D numpy array representing the averaged bubble for each component.
    av_err_mat: 3D numpy array representing the standard deviation of the averaged bubble for each component.
    """

    nS = len(data[0])
    print(nS, 'simulations for this combination.')
    nC = len(data[0][0])

    # Initialize matrices to store the averaged results and errors
    av_mat, av_err_mat = np.zeros((2, nC, 2*winsize+2, 2*winsize+2))
    MATRIX = np.zeros((4, nS, winsize+1, winsize+1))
    MATRIX[:] = np.nan

    # Process each component of the simulation data
    for col in range(nC):
        # Iterate over each quadrant in the data
        for ijk, corner in enumerate(data):
            for ss, simulation in enumerate(corner):
                real = simulation[col]
                nT, nN = np.shape(real)
                if ijk % 2 != 0:
                    real = real[::-1]
                if ijk in [2, 3]:
                    MATRIX[ijk, ss, :nT, -nN:] = real
                else:
                    MATRIX[ijk, ss, :nT, :nN] = real

        # Combine the quadrants into a full bubble
        whole_bubble = np.zeros((nS, 2*winsize+2, 2*winsize+2))
        for ss in range(nS):
            top = np.concatenate((MATRIX[1, ss][::-1], MATRIX[0, ss]), axis=0)
            bottom = np.concatenate((MATRIX[3, ss][::-1], MATRIX[2, ss]), axis=0)
            whole_bubble[ss] = np.concatenate((bottom, top), axis=1).transpose()

        # Calculate the mean and standard deviation of the full bubble
        mean = np.nanmean(whole_bubble, axis=0)
        mvar = np.nanstd(np.abs(whole_bubble), axis=0)
        dims = np.count_nonzero(~np.isnan(whole_bubble), axis=0)
        mvar /= dims

        av_mat[col] = mean
        av_err_mat[col] = mvar

    # Plot the averaged results if plotting is enabled
    if plots:
        fig, ax = plt.subplots(2, 2, figsize=(5, 5))
        for col in range(nC):
            im = ax[col, 0].imshow(av_mat[col], origin='lower', interpolation='none', aspect='equal', cmap='tab20c')
            clb = plt.colorbar(im, ax=ax[col, 0], shrink=0.6)
            clb.ax.set_title([r'$\left\langle\varphi \right\rangle$', r'$\left\langle \Pi \right\rangle$'][col], \
                             size=11, horizontalalignment='center', verticalalignment='bottom')

            im = ax[col, 1].imshow(av_err_mat[col], origin='lower', interpolation='none', aspect='equal', cmap='tab20c')
            clb = plt.colorbar(im, ax=ax[col, 1], shrink=0.6)
            clb.ax.set_title([r'$\left\langle \delta\varphi \right\rangle$', r'$\left\langle \delta\Pi \right\rangle$'][col], \
                             size=11, horizontalalignment='center', verticalalignment='bottom')
        beautify(ax, times=-70)
        plt.tight_layout()
        plt.show()

    return av_mat, av_err_mat

