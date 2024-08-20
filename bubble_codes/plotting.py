import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator

import scipy.ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d

from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from mpl_toolkits.axes_grid1 import ImageGrid

allcolors = ['#377eb8', '#ff7f00', 'forestgreen', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

plt.rcParams.update({'backend' : 'Qt5Agg'})
plt.rcParams.update({'text.usetex' : True})

plt.rcParams.update({'font.size' : 11.0})
plt.rcParams.update({'axes.titlesize': 14.0})  # Font size of title
plt.rcParams.update({'axes.titlepad' : 10.0})
plt.rcParams.update({'axes.labelsize': 14.0})  # Axes label sizes
plt.rcParams.update({'axes.labelpad' : 10.0})
plt.rcParams.update({'xtick.labelsize': 14.0})
plt.rcParams.update({'ytick.labelsize': 14.0})
plt.rcParams.update({'xtick.labelsize': 10.0})
plt.rcParams.update({'ytick.labelsize': 10.0})

plt.rcParams.update({'axes.spines.left'   : True})
plt.rcParams.update({'axes.spines.right'  : True})
plt.rcParams.update({'axes.spines.top'    : True})
plt.rcParams.update({'axes.spines.bottom' : True})
plt.rcParams.update({'savefig.format'     : 'pdf'})
plt.rcParams.update({'savefig.bbox'       : 'tight'})
plt.rcParams.update({'savefig.pad_inches' : 0.1})
plt.rcParams.update({'pdf.compression' : 6})

f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.1e'%x))
fmt = mticker.FuncFormatter(g)

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
 #   return r"$10^{{{1:d}}}$".format(coeff, exponent, precision)

def sci_notation1(num, decimal_digits=1, precision=None, exponent=None):
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

    return r"$10^{{{1:d}}}$".format(coeff, exponent, precision)

def clear_last_coln(ax, title=None):
    ax[len(ax)-1].legend(loc='center', ncol=1, frameon=False, title=title)
    ax[len(ax)-1].set_ylim((-1,0))
    ax[len(ax)-1].set_xlim((-1,0))
    ax[len(ax)-1].spines['right'].set_visible(False)
    ax[len(ax)-1].spines['left'].set_visible(False)
    ax[len(ax)-1].spines['top'].set_visible(False)
    ax[len(ax)-1].spines['bottom'].set_visible(False)
    ax[len(ax)-1].axes.yaxis.set_ticklabels([])
    ax[len(ax)-1].axes.xaxis.set_ticklabels([])
    ax[len(ax)-1].grid(False)
    ax[len(ax)-1].tick_params(left = False,top = False,right = False,bottom = False)
    return ax


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def simple_imshow(bubble, xList, tList, contour=False, title=None, ret=False, cmap='tab20c'):
    fig, ax = plt.subplots(1, 1, figsize = (5, 4))
    ext = [xList[0],xList[-1],tList[0],tList[-1]]
    im = plt.imshow(bubble[0], interpolation='none', extent=ext, aspect='equal', origin='lower', cmap=cmap)
    if contour:
        ax.contour(xList, tList, bubble[0], levels=5, linewidths=0.5, colors='k')
    clb = plt.colorbar(im, ax = ax, shrink=0.5)
   # plt.plot(0, 0, 'bo', ms=3)
    plt.xlabel(r'$x$'); plt.ylabel(r'$t$')
    plt.title(title)
    if ret:
        return ax
    else:
        beautify(ax, times=-70); plt.tight_layout(); plt.show()
        return

def simple_imshow_continue(ax, bubble, xList, tList, vmin, vmax, contour=False, title=None, cmap='tab20c'):
    ext = [xList[0],xList[-1],tList[0],tList[-1]]
    im = ax.imshow(bubble[0], interpolation='none', extent=ext, vmin=vmin, vmax=vmax, aspect='equal', origin='lower', cmap=cmap)
    if contour:
        ax.contour(xList, tList, bubble[0], levels=5, linewidths=0.5, colors='k')
    clb = plt.colorbar(im, ax = ax, shrink=0.25)
    plt.plot(0, 0, 'bo', ms=3)
    plt.xlabel(r'$x$'); plt.ylabel(r'$t$')
    plt.title(title)
    return ax


#Label line with line2D label data
def labelLine(line,x,label=None,align=True,**kwargs):
    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = degrees(atan2(dy,dx))

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x,y,label,rotation=trans_angle,**kwargs)

def labelLines(lines,align=True,xvals=None,**kwargs):
    ax = lines[0].axes
    labLines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin,xmax = ax.get_xlim()
        xvals = np.linspace(xmin,xmax,len(labLines)+2)[1:-1]

    for line,x,label in zip(labLines,xvals,labels):
        labelLine(line,x,label,align,**kwargs)

def annot_max(xmax, ymax, lab, col, xcap, mind, ax):
    text = lab
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec=col, lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60",color=col)
    kw = dict(xycoords='data',textcoords="data",
              arrowprops=arrowprops, bbox=bbox_props, ha="left", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(xmax-0.01, ymax-5e6), **kw)

def gcd(a, b):
    while b:
        a, b = b, a%b
    return a

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))
