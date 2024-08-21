import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid

# Define color palette
allcolors = ['#377eb8', '#ff7f00', 'forestgreen', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

# Update Matplotlib configuration for consistent styling
plt.rcParams.update({
    'backend': 'Qt5Agg',
    'text.usetex': True,
    'font.size': 11.0,
    'axes.titlesize': 14.0,
    'axes.titlepad': 10.0,
    'axes.labelsize': 14.0,
    'axes.labelpad': 10.0,
    'xtick.labelsize': 10.0,
    'ytick.labelsize': 10.0,
    'axes.spines.left': True,
    'axes.spines.right': True,
    'axes.spines.top': True,
    'axes.spines.bottom': True,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'pdf.compression': 6
})


# Formatter for scientific notation
f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
fmt = mticker.FuncFormatter(lambda x, pos: f"${f._formatSciNotation('%1.1e' % x)}$")


# Define function for string formatting of scientific notation
def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if exponent is None:
        exponent = int(np.floor(np.log10(abs(num))))
    coeff = round(num / float(10.**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits
    return r"${0:.{2}f}\times 10^{{{1:d}}}$".format(coeff, exponent, precision)


def sci_notation1(num, decimal_digits=1, precision=None, exponent=None):
    if exponent is None:
        exponent = int(np.floor(np.log10(abs(num))))
    coeff = round(num / float(10.**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits
    return r"$10^{{{1:d}}}$".format(coeff, exponent, precision)


# Clear the last column of a subplot grid
def clear_last_coln(ax, title=None):
    last_ax = ax[-1]
    last_ax.legend(loc='center', ncol=1, frameon=False, title=title)
    last_ax.set_xlim((-1, 0))
    last_ax.set_ylim((-1, 0))
    for spine in last_ax.spines.values():
        spine.set_visible(False)
    last_ax.tick_params(left=False, right=False, top=False, bottom=False)
    last_ax.set_xticklabels([])
    last_ax.set_yticklabels([])
    last_ax.grid(False)
    return ax

# Generate a colormap with distinct colors
def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


# Simple function to display an image with optional contours
def simple_imshow(bubble, xList, tList, contour=False, title=None, ret=False, cmap='tab20c'):
    fig, ax = plt.subplots(figsize=(5, 4))
    ext = [xList[0], xList[-1], tList[0], tList[-1]]
    im = ax.imshow(bubble[0], interpolation='none', extent=ext, aspect='equal', origin='lower', cmap=cmap)
    if contour:
        ax.contour(xList, tList, bubble[0], levels=5, linewidths=0.5, colors='k')
    plt.colorbar(im, ax=ax, shrink=0.5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$t$')
    ax.set_title(title)
    if ret:
        return ax
    else:
        beautify(ax, times=-70)
        plt.tight_layout()
        plt.show()
        

def beautify(ax, loc='best', times=1, ncol=1, ttl=None, bb=None):
    try:
        len(ax)
    except:
        ax = np.array([ax])
    legs = []
    for ai, aa in enumerate(ax.flatten()):
    #    aa.grid(which='both', ls=':', color='lightgray', alpha=0.7)
        aa.grid(which='major', ls=':', color='darkgray', alpha=0.7)
        aa.tick_params(direction='in', which='both', top=True, right=True)
        #aa.ticklabel_format(axis='both', style='scientific', scilimits=[0.,0.])
        aa.xaxis.set_label_coords(0.5, times*0.0015)
        aa.yaxis.set_label_coords(times*0.0015, 0.5)
        aa.xaxis.label.set_color('k')
        aa.yaxis.label.set_color('k')
        aa.tick_params(axis='x', colors='k')
        aa.tick_params(axis='y', colors='k')
        aa.tick_params(direction='in', which='major')#, bottom=None, left=None, top=None, right=None)
        aa.tick_params(direction='in', which='minor', bottom=None, left=None, top=None, right=None)
        aa.spines['left'].set_color('k')
        aa.spines['right'].set_color('k')
        aa.spines['top'].set_color('k')
        aa.spines['bottom'].set_color('k')
        leg = aa.legend(title=ttl, ncol=ncol, loc=loc, bbox_to_anchor=bb, frameon=False, handlelength=1.5, labelspacing=0.3, columnspacing=0.6)
        legs.append(leg)
    return np.array(legs), ax


def beautify_nolegs(ax, loc='best', times=1, ncol=1, ttl=None, bb=None):
    try:
        len(ax)
    except:
        ax = np.array([ax])
    for ai, aa in enumerate(ax.flatten()):
    #    aa.grid(which='both', ls=':', color='lightgray', alpha=0.7)
        aa.grid(which='major', ls=':', color='lightgray', alpha=0.7)
        aa.tick_params(direction='in', which='both', top=True, right=True)
        #aa.ticklabel_format(axis='both', style='scientific', scilimits=[0.,0.])
        aa.xaxis.set_label_coords(0.5, times*0.0015)
        aa.yaxis.set_label_coords(times*0.0015, 0.5)
        aa.xaxis.label.set_color('k')
        aa.yaxis.label.set_color('k')
        aa.tick_params(axis='x', colors='k')
        aa.tick_params(axis='y', colors='k')
        aa.tick_params(direction='in', which='major')#, bottom=None, left=None, top=None, right=None)
        aa.tick_params(direction='in', which='minor', bottom=None, left=None, top=None, right=None)
        aa.spines['left'].set_color('k')
        aa.spines['right'].set_color('k')
        aa.spines['top'].set_color('k')
        aa.spines['bottom'].set_color('k')
    return ax

