import os
import sys

sys.path.append('/home/dpirvu/project/paper_prefactor/bubble_codes/')

import numpy as np
import random

from functools import partial
from itertools import cycle

import scipy as scp
from scipy import optimize as sco, signal as scs, interpolate as si, ndimage
from scipy.integrate import odeint
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from plotting import *
from experiment import *


def extract_spec_data(nL, path_sim):
    with open(path_sim) as file:
        params = [next(file) for x in range(8)]
        params.append(file.readlines()[-1])

    data = np.genfromtxt(path_sim, skip_header=8)
    nNnT, nC = np.shape(data)
    reshaped_dat = np.asarray([np.reshape(data[:,cc], (nNnT//nL, nL)) for cc in range(nC)])
    return params, reshaped_dat


def save_txt_file(filename, data):
    with open(filename, 'w') as f:
        for row in data:
            f.write('\t'.join(map(str, row)) + '\n')
    return


def extract_data(nL, path_sim):
    with open(path_sim) as file:
        params = [next(file) for x in range(8)]
        params.append(file.readlines()[-1])

    data = np.genfromtxt(path_sim, skip_header=8, skip_footer=1)
    nNnT, nC = np.shape(data)
    reshaped_dat = np.asarray([np.reshape(data[:,cc], (nNnT//nL, nL)) for cc in range(nC)])
    return params, reshaped_dat


def get_realisation(nL, nTimeMAX, path_sim):
    params, data = extract_data(nL, path_sim)
    tdecay = int(params[-1][-13:])

    if tdecay >= nTimeMAX:
        outcome = 2
    else:
        slice   = data[0,-1,:]
        outcome = check_decay(slice)

    initcond = data[:,0,:]
    real     = data[:,:,:]
    prebubble= data[:,-2,:]
    bubble   = data[:,-1,:]
    return tdecay, outcome, initcond, real, prebubble, bubble



def extract_bubble_data(nL, path_sim):
    data = np.genfromtxt(path_sim)
    nNnT, nC = np.shape(data)
    reshaped_dat = np.asarray([np.reshape(data[:,cc], (nNnT//nL, nL)) for cc in range(nC)])
    return reshaped_dat

def get_bubble_realisation(nL, path_sim):
    data = extract_bubble_data(nL, path_sim)

    slice   = data[0,-10:,:].flatten()
    outcome = check_decay(slice)

    if outcome == 1:
        data[0] = - data[0]
        data[1] = - data[1]
    return data, outcome

def check_decay(slice):
    right_phi = np.sum(slice > 10.)
    left_phi = np.sum(slice < -10.)
    if right_phi == 0 and left_phi == 0:
        return 2
    return 0 if right_phi > left_phi else 1


def get_decay_time(real):
    fldreal = real[0,:,:]
    ums = np.sum(np.abs(fldreal) > 10., axis=-1)
    return np.argwhere(ums > 0.)[0][0]


def centre_bubble(real, tdecay):
    nC, nT, nN = np.shape(real)
    tamp = max(0, nT - 2 * nN)
    real = real[:, tamp:]
    for _ in range(2):  # Apply the rolling twice in one loop
        critslice = np.abs(real[0, tdecay, :])
        x_centre = int(round(np.mean(np.argwhere(critslice > 1.))))
        real = np.roll(real, nN // 2 - x_centre, axis=-1)
    return real

def bubble_counts_at_fixed_t(bubble, thresh):
    return np.count_nonzero(bubble >= thresh, axis=1)

def bubble_counts_at_fixed_x(bubble, thresh):
    return np.count_nonzero(bubble >= thresh, axis=0)

def reflect_against_equil(bubble, phi_init):
    return np.abs(bubble-phi_init) + phi_init

def find_nucleation_center(bubble, phi_init, crit_thresh, crit_rad):
    T, N = np.shape(bubble)
    bubble_counts = bubble_counts_at_fixed_t(bubble, crit_thresh)
    t0 = np.argmin(np.abs(bubble_counts - crit_rad))

    bubble_counts = bubble_counts_at_fixed_x(bubble[:int(min(t0+5*crit_rad,T))], crit_thresh)
    x0 = np.argmax(bubble_counts)
    return min(T-1,t0), min(N-1,x0)

def find_nucleation_center2(bubble, phi_init, crit_thresh, crit_rad):
    T, N = np.shape(bubble)
    bubble_counts = bubble_counts_at_fixed_t(bubble, crit_thresh)
    t0 = np.argmin(np.abs(bubble_counts - crit_rad))

    bub = bubble[t0]
    cds = np.argwhere(bub>crit_thresh).flatten()
   # while len(cds) < crit_rad:
    while len(cds) < crit_rad//5:
        t0 += 1
        bub = bubble[t0]
        cds = np.argwhere(bub>=crit_thresh).flatten()
    x0 = (cds[0] + cds[-1]) // 2
    return min(T-1,t0), x0

#    slice = bubble[t0, int(N/2-crit_rad*2):int(N/2+crit_rad*2)]
#    bubble_counts = np.argwhere(slice>crit_thresh)
#    x0 = int(np.mean(bubble_counts))
#    return min(T-1,t0), min(N-1,x0+int(N/2-crit_rad*2))

def find_final_zero_index(arr):
    for i in range(len(arr) - 2):
        if arr[i] == 0:
            if arr[i+1] <= 0 and arr[i+2] <= 0:
                return i + 2
    return 1  # Return 1 if no such sequence found

def find_t_max_width(bubble, light_cone, phi_init, tv_thresh, crit_rad):
    nT, nN = np.shape(bubble)
    refl_bubble = reflect_against_equil(bubble, phi_init)

    bubble_counts = bubble_counts_at_fixed_t(refl_bubble, tv_thresh)
    bubble_diffs = bubble_counts[1:] - bubble_counts[:-1]

 #   out = find_final_zero_index(bubble_diffs[::-1])
    tmax = np.argwhere(bubble_diffs[::-1] >= light_cone*2).flatten()
    out = next((ii for ii, jj in zip(tmax[:-1], tmax[1:]) if jj-ii == 1), 1)
    return nT-out


def multiply_bubble(bubble, light_cone, phi_init, vCOM, normal, nL):
    # multiplies bubbles so causal tail is unfolded from PBC
    if vCOM<0:
        bubble = bubble[:,:,::-1]
    C, T, N = np.shape(bubble)
    bubble = np.asarray([np.tile(bubble[col], fold(vCOM)) for col in range(C)])
    TT, NN = np.shape(bubble[0])
    mn = np.mean(bubble[0])
    for t in range(TT):
        a, b = int((TT-t)/light_cone) + N, int((TT-t)/light_cone/3.) - N//4
        x1, x2 = np.arange(a, NN), np.arange(b)
        x1, x2 = x1 - a, x2 - (b-NN)
        for x in np.append(x1, x2):
            if 0 <= x < NN:
                bubble[0,t,x] = mn
    if vCOM<0:
        bubble = bubble[:,:,::-1]
    return bubble

def retired_tanh_profile(x, r0L, r0R, vL, vR, dr, a):
    wL, wR = dr/gamma(vL), dr/gamma(vR)
    return ( np.tanh( (x - r0L)/wL ) + np.tanh( (r0R - x)/wR ) ) * a

def tanh_profile(x, r0L, r0R, vL, vR):
    return ( np.tanh( (x - r0L)/vL ) + np.tanh( (r0R - x)/vR ) )

def get_profile_bf(xlist, phibubble, prior):
    bounds = ((xlist[0], 0., 0., 0., xlist[0], -1.), (0., xlist[-1], 1., 1., xlist[-1], 1.))
    tanfit, _ = sco.curve_fit(retired_tanh_profile, xlist, phibubble, p0=prior, bounds=bounds)
    return tanfit

def hypfit_right_mover(tt, rr):
#    hyperbola  = lambda t, a, b, c: np.sqrt(c + (t - b)**2.) + a
    hyperbola  = lambda t, a, b, c: np.sqrt(c + b*t + t**2.) + a
    try:
        prior   = (float(min(rr)), float(tt[np.argmin(rr)]), 1e3)
        fit, _  = sco.curve_fit(hyperbola, tt, rr, p0 = prior)
        traj    = hyperbola(tt, *fit)
        return traj
    except:
        return []

def hypfit_left_mover(tt, ll):
#    hyperbola  = lambda t, d, e, f: - np.sqrt(f + (t - e)**2.) + d
    hyperbola  = lambda t, d, e, f: - np.sqrt(f + e*t + t**2.) + d
    try:
        prior   = (float(max(ll)), float(tt[np.argmax(ll)]), 1e3)
        fit, _  = sco.curve_fit(hyperbola, tt, ll, p0 = prior)
        traj    = hyperbola(tt, *fit)
        return traj
    except:
        return []

def get_velocities(rrwallfit, llwallfit):
    uu = np.gradient(rrwallfit) #wall travelling with the COM
    vv = np.gradient(llwallfit) #wall travelling against
    uu[np.abs(uu)>=1.] = np.sign(uu[np.abs(uu)>=1.])*(1.-1e-15)
    vv[np.abs(vv)>=1.] = np.sign(vv[np.abs(vv)>=1.])*(1.-1e-15)
    uu[np.isnan(uu)] = np.sign(vv[-1])*(1.-1e-15)
    vv[np.isnan(vv)] = np.sign(uu[-1])*(1.-1e-15)

    # centre of mass velocity
    aa = ( 1.+uu*vv-np.sqrt((-1.+uu**2.)*(-1.+vv**2.)))/( uu+vv)
    # instantaneous velocity of wall
    bb = (-1.+uu*vv+np.sqrt((-1.+uu**2.)*(-1.+vv**2.)))/(-uu+vv)
    return uu, vv, aa, bb


def find_order_changes(arr):
    changes = []
    decreasing = False
    
    for i in range(len(arr) - 1):
        if arr[i] < arr[i + 1]:
            if decreasing:
                changes.append(i)
                break
        elif arr[i] > arr[i + 1]:
            if not decreasing:
                decreasing = True
            if changes and changes[-1] == i:
                changes.pop()
        if len(changes) != 0:
            break
        # If the elements are equal, we skip because it's not considered a change in trend.
    if changes == []:
        changes = [0]
    return changes[0]

def find_COM_vel(real, fldamp, winsize, nL, light_cone, phi_init, tv_thresh, crit_thresh, crit_rad, plots=False):
    nC, nT, nN = np.shape(real)
    real = real[0]
    t_maxwid = find_t_max_width(real, light_cone, phi_init, tv_thresh, crit_rad)

    real = real[:t_maxwid]
    edge = np.abs(nT-t_maxwid)
    mint = int(np.max((0, t_maxwid-nL//2)))
 #   print('t_maxwid, nT', t_maxwid, nT)

    wth = real[mint:t_maxwid, edge:(nN-edge)]
    nT, nN = np.shape(wth)
    t_centre0, x_centre = find_nucleation_center(wth, phi_init, crit_thresh, crit_rad)

    if plots:
        t, x = np.linspace(-t_centre0, nT-1-t_centre0, nT), np.linspace(-x_centre, nN-1-x_centre, nN)
        simple_imshow([wth], x, t, title=r'wth', contour=False, ret=False)

    x_centre+= edge
    t_centre = t_centre0 + max(t_maxwid - nL//2, 0)
    tl_stop, tr_stop = int(max(0, t_centre - winsize)), int(min(nT, t_centre + winsize//2))
    xl_stop, xr_stop = int(max(0, x_centre - winsize)), int(min(nN, x_centre + winsize))
#    print('t_centre0, t_centre, tl_stop, tr_stop', t_centre0, t_centre, tl_stop, tr_stop)

    simulation = real[tl_stop:tr_stop, xl_stop:xr_stop]
    nT, nN = np.shape(simulation)
    tcen, xcen = find_nucleation_center(simulation, phi_init, crit_thresh, crit_rad)

    if plots:
        t, x = np.linspace(-tcen, nT-1-tcen, nT), np.linspace(-xcen, nN-1-xcen, nN)
        simple_imshow([simulation], x, t, title=r'wth 2 process', contour=False, ret=False)

    betas = np.zeros((len(fldamp)))
    for vv, v_size in enumerate(fldamp):
        vel_plots = (True if (vv%2==0 and plots) else False)
        betas[vv] = get_COM_velocity(simulation, phi_init, crit_thresh, crit_rad, v_size, tcen, xcen, vel_plots)

    if plots:
        fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
        plt.plot(fldamp, betas, marker='o', ms=3)
        beautify(ax, times=-70, ttl=r'${{ \rm Mean \ }} v={:.2f}$'.format(np.nanmean(betas))); plt.show()
    return np.nanmean(betas), np.nanvar(betas)



def get_COM_velocity(simulation, phi_init, crit_thresh, crit_rad, vvv, tcen, xcen, plots=False):
    nT, nN = np.shape(simulation)
    data_list, prior, target = [], None, nN/2.
    for tt in reversed(range(nT)):
        slice = simulation[tt]

        try: target = int(np.round(np.nanmean(np.argwhere(slice > vvv))))
        except: break

        coord_list = np.arange(nN) - target
        try:
            prior = get_profile_bf(coord_list, slice, prior)
            r0L,r0R,vL,vR,_,_ = prior

            curve = retired_tanh_profile(coord_list, *prior)
            data_list.append([tt, r0L+target, r0R+target])

            if plots and False:
                if tt%50!=10: continue
                print(prior)
                fig, ax = plt.subplots(1, 1, figsize=(3, 3))
                plt.plot(coord_list, retired_tanh_profile(coord_list, *prior), 'r')
                plt.plot(coord_list, slice, 'bo', ms=1)
                plt.plot(coord_list, curve, 'go', ms=1)
                plt.axhline(vvv, ls=':', color='k')
                plt.title(r'$t={:.1f}$'.format(tt)); 
                beautify(ax, times=-70)
                plt.show()

        except:
            continue

    # get velocities from derivatives of trajectories
    try:
        # right wall travels along with COM and left wall against
        data_list = np.array(data_list)[::-1]
        ttwallfit, ll, rr = data_list[:,0], data_list[:,1] - xcen, data_list[:,2] - xcen

        # fit walls to hyperbola
        llwallfit = hypfit_left_mover(ttwallfit, ll)
        rrwallfit = hypfit_right_mover(ttwallfit, rr)

        uu, vv, aa, bb = get_velocities(rrwallfit, llwallfit)
        indix = np.nanargmin(np.abs(uu - vv))
        vCOM = aa[indix]

        if plots:
            fig, ax = plt.subplots(1, 2, figsize = (7, 4))
            ext = np.array([-xcen, nN-xcen, -tcen, nT-tcen])
            im0 = ax[0].imshow(simulation, interpolation='antialiased', extent=ext, origin='lower', cmap='tab20c', aspect='auto')
           # ax.contour(np.arange(-xcen,nN-xcen), np.arange(-tcen, nT-tcen), simulation, levels=3, linewidths=0.5, colors='k')

            ax[0].plot(rr, (ttwallfit-tcen), color='b', ls='-', lw=1, label=r'$\rm rr$')
            ax[0].plot(ll, (ttwallfit-tcen), color='g', ls='-', lw=1, label=r'$\rm ll$')

            try:
                ax[0].plot(rrwallfit, (ttwallfit-tcen), color='b', ls=':', lw=1, label=r'$\rm rr \ hyp \ fit$')
                ax[0].plot(llwallfit, (ttwallfit-tcen), color='g', ls=':', lw=1, label=r'$\rm ll \ hyp \ fit$')

                ax[1].plot((ttwallfit-tcen), uu, color='b', ls='-',  lw=1, label=r'$\rm wall \, travelling \, with \, COM$')
                ax[1].plot((ttwallfit-tcen), vv, color='g', ls='-',  lw=1, label=r'$\rm wall \, travelling \, against \, COM$')
                ax[1].plot((ttwallfit-tcen), aa, color='r', ls='--', lw=1, label=r'$v_{\rm COM}(t)$')
                ax[1].plot((ttwallfit-tcen), bb, color='orange', ls=':', label=r'$v_{\rm walls}(t)$')
                ax[1].plot((ttwallfit-tcen),-bb, color='orange', ls=':', label=r'$v_{\rm walls}(t)$')
                ax[1].plot((ttwallfit-tcen)[indix], vCOM, 'ko', ms=3)
            except:
                ax[0].plot(0., 0., color='yellow', marker='o', ms=3, label=r'$\rm failed \ fit$')

            cbar = plt.colorbar(im0, ax=ax[0]); cbar.ax.set_title(r'$\bar{\phi}$')
            ax[0].set(ylabel=r'$t$'); ax[0].set(xlabel=r'$r$'); ax[1].set(ylabel=r'$t$'); ax[1].set(xlabel=r'$v(t)$')
            beautify(ax, times=-70, ttl=r'$\phi={:.2f}$'.format(vvv)); plt.tight_layout(); plt.show()

    except:
        return 'nan'
    return vCOM

rapidity = lambda v: np.arctanh(v)
gamma    = lambda v: (1. - v**2.)**(-0.5)
fold     = lambda beta: 3
#fold     = lambda beta: 2 if 0.8 > np.abs(beta) > 0.7 else 3 if np.abs(beta) > 0.8 else 1
addvels  = lambda v1,v2: (v1 + v2) / (1. + v1*v2)

def get_totvel_from_list(vels):
    totvel = 0.
    for ii in vels:
        totvel = addvels(ii, totvel)
    return totvel

def coord_pair(tt, xx, beta, ga, c):
    t0 = (tt + beta*xx/c) * ga
    x0 = (xx + beta*tt  ) * ga
    return t0, x0

def boost_bubble(simulation, t0, x0, vCOM, c=1):
    # create template for new bubble
    C, T, N = np.shape(simulation)

    # boost factor
    beta = vCOM / c
    ga = gamma(beta)

    # old grid
    t, x = np.linspace(-t0, T-1-t0, T), np.linspace(-x0, N-1-x0, N)

    # array to fill
    rest_bubble = np.zeros(np.shape(simulation))
    for col, element in enumerate(simulation):
        # interpolate image onto 2d rectangular grid
        g = interp2d(x, t, element, kind = 'cubic', bounds_error=True, fill_value=0.)

        for tind, tval in enumerate(t):
            tlensed, xlensed = coord_pair(tval, x, beta, ga, c)
            interpolated = si.dfitpack.bispeu(g.tck[0], g.tck[1], g.tck[2], g.tck[3], g.tck[4], xlensed, tlensed)[0]
            rest_bubble[col, tind, :] = interpolated
    return t, x, rest_bubble

#### Tools for averaging bubbles

def quadrant_coords(real, phi_init, crit_thresh, crit_rad, maxwin, plots=False):
    bub = np.copy(np.abs(real[0]))
    bub = gaussian_filter(bub, 0.5, mode='nearest')
    bub[bub > crit_thresh] = crit_thresh
    tcen, xcen = find_nucleation_center(bub, phi_init, crit_thresh, crit_rad)
    nT, nN = np.shape(bub)

    aa,bb = max(0, xcen-maxwin), min(nN, xcen+maxwin+1)
    cc,dd = max(0, tcen-maxwin), min(nT, tcen+maxwin+1)

    aaL, bbL = np.arange(aa, xcen), np.arange(xcen, bb)
    ccL, ddL = np.arange(cc, tcen), np.arange(tcen, dd)

    ddd, bbb = np.meshgrid(ddL, bbL, sparse='True')
    upright_quad = real[:, ddd, bbb]
    ddd, aaa = np.meshgrid(ddL, aaL, sparse='True')
    upleft_quad  = real[:, ddd,aaa]
    ccc, bbb = np.meshgrid(ccL, bbL, sparse='True')
    lowright_quad= real[:, ccc, bbb]
    ccc, aaa = np.meshgrid(ccL, aaL, sparse='True')
    lowleft_quad = real[:, ccc, aaa]

    if plots:
        if len(bbL) > 0 and len(aaL) > 0 and len(ccL) > 0 and len(ddL) > 0:
            a1, a2 = np.copy(upright_quad[0]), np.copy(upleft_quad[0])
            a3, a4 = np.copy(lowright_quad[0]), np.copy(lowleft_quad[0])
            avec = [a1,a2,a3,a4]
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
                #aa.set_xlabel(r'$x$'); aa.set_ylabel(r'$t$')
                aa.set_xticklabels([]); aa.set_yticklabels([])
            beautify(ax, times=-70); plt.tight_layout(); plt.show()
        else:
            print('Failed')

    return upright_quad, upleft_quad, lowright_quad, lowleft_quad

def stack_bubbles(data, maxwin, phi_init, crit_thresh, crit_rad, plots=False):
    upright_stack, upleft_stack, lowright_stack, lowleft_stack = ([] for ii in range(4))
    for sim, real in data:

        if plots:
            bub = np.copy(np.abs(real[0]))
            bub = gaussian_filter(bub, 0.5, mode='nearest')
            bub[bub > crit_thresh] = crit_thresh
            nT, nN = np.shape(bub)

            tcen, xcen = find_nucleation_center(bub, phi_init, crit_thresh, crit_rad)
            tl,tr = max(0, tcen-maxwin), min(nT, tcen+maxwin+1)
            xl,xr = max(0, xcen-maxwin), min(nN, xcen+maxwin+1)

            fig, ax = plt.subplots(1, 1, figsize = (3, 3))
            ext = [xl,xr,tl,tr]
            plt.imshow(bub[tl:tr,xl:xr], interpolation='none', extent=ext, aspect='equal', origin='lower', cmap='tab20c')
            plt.plot(xcen,tcen,'bo')
            plt.xlabel('x'); plt.ylabel('t')
            beautify(ax, times=-70, ttl=r'${{\rm Sim}} = {:.0f}$'.format(sim)); plt.show()

        ur, ul, lr, ll = quadrant_coords(real, phi_init, crit_thresh, crit_rad, maxwin, plots)

        bool = True
        for ii in [ur, ul, lr, ll]:
            if np.shape(ii)[1] <= maxwin//5 or np.shape(ii)[2] <= maxwin//5:
                bool = False
                break

        if bool:
            upright_stack.append(ur)
            upleft_stack.append(ul)
            lowright_stack.append(lr)
            lowleft_stack.append(ll)
     #   else:
     #       print('skipped sim', sim)
    return upright_stack, upleft_stack, lowright_stack, lowleft_stack

def average_stacks(data, winsize, normal, plots=False):
    nS = len(data[0])
    print(nS, 'simulations for this combination.')
    nC = len(data[0][0])

    av_mat, av_err_mat = np.zeros((2, nC, 2*winsize+2, 2*winsize+2))
    MATRIX = np.zeros((4, nS, winsize+1, winsize+1))
    MATRIX[:] = np.nan

    for col in range(nC):
        for ijk, corner in enumerate(data): #for each quadrant
            for ss, simulation in enumerate(corner):
                real = simulation[col]
                nT, nN = np.shape(real)
                if ijk%2!=0:
                    real = real[::-1]
                if ijk in [2,3]:
                    MATRIX[ijk, ss, :nT, -nN:] = real
                else:
                    MATRIX[ijk, ss, :nT, :nN] = real

#         if plots:
#             for ss in range(nS):
#                 if ss%50!=0: continue
#                 fig, ax = plt.subplots(2, 2, figsize=(5,5))
#                 ax[1,0].imshow(MATRIX[3][ss], interpolation='none', aspect='equal', cmap='tab20c')
#                 ax[1,1].imshow(MATRIX[1][ss], interpolation='none', aspect='equal', cmap='tab20c')
#                 ax[0,0].imshow(MATRIX[2][ss][::-1], interpolation='none', aspect='equal', cmap='tab20c')
#                 ax[0,1].imshow(MATRIX[0][ss][::-1], interpolation='none', aspect='equal', cmap='tab20c')
#                 for aa in ax.flatten():
#                     aa.set_xticklabels([]); aa.set_yticklabels([])
#                 plt.suptitle([r'$\left\langle\varphi \right\rangle$',r'$\left\langle \Pi \right\rangle$'][col])
#                 beautify(ax, times=-70); plt.tight_layout(); plt.show()

        whole_bubble = np.zeros((nS, 2*winsize+2, 2*winsize+2))
        for ss in range(nS):
            top = np.concatenate((MATRIX[1,ss][::-1], MATRIX[0,ss]), axis=0)
            bottom = np.concatenate((MATRIX[3,ss][::-1], MATRIX[2,ss]), axis=0)
            whole_bubble[ss] = np.concatenate((bottom, top), axis=1).transpose()

        mean = np.nanmean(whole_bubble, axis=0)
        mvar = np.nanstd(np.abs(whole_bubble), axis=0)
        dims = np.count_nonzero(~np.isnan(whole_bubble), axis=0)
        mvar/= dims

        av_mat[col] = mean
        av_err_mat[col] = mvar

    if plots:
        fig, ax = plt.subplots(2, 2, figsize=(5,5))
        for col in range(nC):
            im  = ax[col,0].imshow(av_mat[col], origin='lower', interpolation='none', aspect='equal', cmap='tab20c')
            clb = plt.colorbar(im, ax = ax[col,0], shrink=0.6)
            clb.ax.set_title([r'$\left\langle\varphi \right\rangle$',r'$\left\langle \Pi \right\rangle$'][col], \
                                    size=11, horizontalalignment='center', verticalalignment='bottom')

            im  = ax[col,1].imshow(av_err_mat[col], origin='lower', interpolation='none', aspect='equal', cmap='tab20c')
            clb = plt.colorbar(im, ax = ax[col,1], shrink=0.6)
            clb.ax.set_title([r'$\left\langle \delta\varphi \right\rangle$',r'$\left\langle \delta\Pi \right\rangle$'][col], \
                                    size=11, horizontalalignment='center', verticalalignment='bottom')
        beautify(ax, times=-70); plt.tight_layout(); plt.show()
    return av_mat, av_err_mat

def lin_fit_times(times,num,tmin,tmax):
    """
    Given a collection of decay times, do a linear fit to
    the logarithmic survival probability between given times

    Input
      times : array of decay times
      num   : original number of samples
      tmin  : minimum time to fit inside
      tmax  : maximum time to fit inside
    """
    t = np.sort(times)
    p = survive_prob(times, num)
    ii = np.where( (t>tmin) & (t<tmax) )
    return np.polyfit(t[ii], np.log(p[ii]), deg=1)


# To do: Debug more to ensure all offsets are correct.
# I've done a first go through and I think they're ok
def survive_prob(t_decay, num_samp):
    """
    Return the survival probability as a function of time.

    Input:
      t_decay  : Decay times of trajectories
      num_samp : Total number of samples in Monte Carlo

    Note: Since some trajectories may not decay, t_decay.size isn't always num_sampe

    Output:
      prob     : Survival Probability

    These can be plotted as plt.plot(t_sort,prob) to get a survival probability graph.
    """
    frac_remain = float(num_samp-t_decay.size)/float(num_samp)
    prob = 1. - np.linspace(1./num_samp, 1.-frac_remain, t_decay.size, endpoint=True)
    return prob

def get_line(dataset, slope, offset):
    return dataset * slope + offset

def f_surv(times, ntot):
    return np.array([1.-len(times[times<=ts])/ntot for ts in times])


def FWHM(X,Y):
    half_max = np.amax(Y)/2.
    #find when function crosses line half_max (when sign of diff flips)
    #take the 'derivative' of signum(half_max - Y[])
    d = (half_max - Y[:-1]) - (half_max - Y[1:])
    plt.plot(X[0:len(d)], d) #if you are interested
    #find the left and right most indexes
    left_idx = np.argwhere(d > 0)[0]; plt.axvline(left_idx, color='r')
    right_idx = np.argwhere(d < 0)[-1]; plt.axvline(right_idx, color='r'); plt.show()
    return X[right_idx] - X[left_idx] #return the difference (full width)

######################################
# Tools for oscillons

def find_nucleation_center(bubble, phi_init, crit_thresh, crit_rad):
    T, N = np.shape(bubble)

    bubble_counts = np.count_nonzero(bubble >= crit_thresh, axis=1)
    t0 = np.argmin(np.abs(bubble_counts - crit_rad))

    bubble_counts = np.count_nonzero(bubble >= crit_thresh, axis=0)
    x0 = np.argmax(bubble_counts)

    return min(T-1,t0), min(N-1,x0)

def get_bubble(exp_params, sim, crit_thresh, crit_rad, minduration):
    path2CLEANsim = clean_sim_location(*exp_params, sim)
    bubble = np.load(path2CLEANsim)

    real = np.abs(bubble[0])
    
    # smoothing recommended
    real = gaussian_filter(real, 1.5, mode='nearest')
    
    tcen, xcen = find_nucleation_center(real, phieq, crit_thresh, crit_rad)

    if tcen - minduration < 0:
        return None
    ind = max(0, tcen - minduration)
    return bubble[0, ind:tcen, :]

def get_HT(array):  
    array = array - np.mean(array)
    w  = np.fft.fftfreq(len(array), d=1) * len(array)
    FD = np.fft.fft(array, axis=0)
    FD[w<=0.,:] = 0.
    FD[w>0.,:] *= 2.
    HD = np.fft.ifft(FD, axis=0)
    return np.abs(HD)

def get_osc_trajectory(array, row, col, extent):
    T, N = np.shape(array)
    maxLine = []
    CCol = col
    for rr in range(row)[::-1]:
        if rr == row-1:
            col = CCol
        colmin = col - extent
        colmax = col + extent+1
        val = 0
        for cc in range(colmin, colmax):
            cc = cc%N
            if array[rr][cc] > val:
                val = array[rr][cc]
                col = cc
        maxLine.append(col)
    maxLine = maxLine[::-1]
    for rr in range(row, T):
        if rr == row:
            col = CCol
        colmin = col - extent
        colmax = col + extent+1
        val = 0
        for cc in range(colmin, colmax):
            cc = cc%N
            if array[rr][cc] > val:
                val = array[rr][cc]
                col = cc
        maxLine.append(col)
    return np.array(maxLine)

# average oscillon trajectories
def tolerant_mean(arrs):
    lens = np.array([len(i) for i in arrs])
    arr  = np.zeros((len(lens), np.max(lens)))
    arr[:] = np.nan
    for ri, osc in enumerate(arrs):
        arr[ri, :len(osc)] = osc
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

def flatten_comprehension(matrix):
    return np.array([item for row in matrix for item in row])

def flatten_comprehension2(matrix):
    return np.array([np.abs(np.round(item,10)) for row in matrix for item in row])
