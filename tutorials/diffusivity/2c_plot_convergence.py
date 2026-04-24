import string, re, struct, sys, math, os
import time
import types
from sys import argv
from shutil import move
from os import remove, close
from subprocess import PIPE, Popen
import numpy as np
import pandas as pd
import matplotlib

# Use the non-interactive 'Agg' backend so figures are saved to files.
# This is needed on headless servers (no display) and for reproducible output.
matplotlib.use('Agg')

import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import rc
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.optimize import curve_fit
from matplotlib import rc
from matplotlib import rcParams
from ase.io import read, write
from typing import Tuple
import scipy


# ---------------------------------------------------------------------------
# Font and LaTeX settings
# ---------------------------------------------------------------------------

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']   # Tahoma font matches the paper style

matplotlib.rcParams.update({'font.size': 12})

# LaTeX preamble: load siunitx for SI units and amsmath for equations
matplotlib.rcParams['text.latex.preamble'] = (
    r"\usepackage{siunitx} \sisetup{detect-all} \usepackage{amsmath, amssymb}"
)



# ----------------------------------------
#              COLORS
# ---------------------------------------

c_orange=np.array([172./255., 90./255., 22./255.,1.0])
c_red_nature=np.array([206./255., 30./255., 65./255.,1.0])
c_green_nature=np.array([96./255., 172./255., 63./255.,1.0])
c_blue_nature =np.array([54./255., 79./255., 156./255.,1.0])
c_purple_nature=np.array([245./255., 128./255., 32./255.,1.0])#np.array([192./255., 98./255., 166./255.,1.0])
c_black1=np.array([50./255., 50./255., 50./255.,1.0])
c_black2=np.array([100./255., 100./255.,100./255.,1.0])
c_black3=np.array([150./255., 150./255., 150./255.,1.0])
c_black4=np.array([200./255., 200./255., 200./255.,1.0])
c_cyna='#00aeff' #42cef4'
c_dark_green='#004d00'
c_pink='#ff1aff'
c_purple='#990099'
c_dark_orange='#db5f00'
c_orange='orange'


c_black=np.array([40./255., 41./255., 35./255.,1.0])
#c_black=np.array([116./255., 112./255., 93./255.,1.0])

c_red=np.array([249./255., 36./255.,114./255.,1.0])
c_purple=np.array([172./255., 128./255., 255./255.,1.0])
c_orange=np.array([253./255., 150./255., 33./255.,1.0])


c_red=np.array([206./255., 30./255., 65./255.,1.0])

#c_green=np.array([62./255., 208./255., 102./255.,1.0])
c_green=np.array([96./255., 172./255., 63./255.,1.0])

c_blue=np.array([26./255., 97./255., 191./255.,1.0])
# c_pink=np.array([144./255., 110./255., 209./255.,1.0])#'#AE5BB3'  #F379FB'
c_orange=np.array([245./255., 128./255., 32./255.,1.0])
c_cyan=np.array([30./255., 165./255., 180./255.,1.0])
c_black=np.array([0./255., 0./255., 0./255.,1.0])



# Unknown functions
def tc_composite(k_m, v_fra_air):
  k_eff=k_m*(2*k_m-2.*v_fra_air*k_m)/(2*k_m+v_fra_air*k_m)
  return k_eff

def smoot_points(x,y,dense_x):
  spl=splrep(x,y,s=0.003)
  y2 = splev(dense_T, spl)
  return y2

c_green=np.array([96./255., 172./255., 63./255.,1.0])
c_blue=np.array([26./255., 97./255., 191./255.,1.0])

class _Colors(object):
    """Helper class with different colors for plotting"""
    red = '#F15854'
    blue = '#5DA5DA'
    orange = '#FAA43A'
    green = '#60BD68'
    pink = '#F17CB0'
    brown = '#B2912F'
    purple = '#B276B2'
    yellow = '#DECF3F'
    gray = '#4D4D4D'
    cyan = '#00FFFF'
    rebecca_purple = '#663399'
    chartreuse = '#7FFF00'
    dark_red = '#8B0000'

    def __getitem__(self, i):
        color_list = [
            self.red,
            self.orange,
            self.green,
            self.blue,
            self.pink,
            self.brown,
            self.purple,
            self.yellow,
            self.gray,
            self.cyan,
            self.rebecca_purple,
            self.chartreuse,
            self.dark_red
        ]
        return color_list[i % len(color_list)]


Colors = _Colors()

fig, (ax1, ax2) = pl.subplots(2, 1, sharex=True)
fig.subplots_adjust(wspace=0, hspace=0)

# read data
directory = "."
source = f"{directory}"
kappas_G, kappas_L, kappas_G_gamma, kappas_L_gamma = \
np.load(f"{source}/results/convergence_test.npy", allow_pickle=True)
list_smear = [0.025, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 30, 40, 60, 80, 100]
temp_list = [100, 300, 1500]
n_temp = len(temp_list)


convergence_pos_y = [0.6, 3.45, 6.55]
convergence_pos_x = [0.1, 0.1, 0.1]
ylim = 8.0
title_upper = [0.1, ylim/1.7, 13, 1.5]
title_lower = [0.1, ylim/1.7, 1, 1.5]


ax1.plot(list_smear, kappas_G[:, 0], label="Gauss", color=c_black, zorder=1)
ax1.plot(list_smear, kappas_G[:, 1], color=c_black, zorder=1)
ax1.plot(list_smear, kappas_G[:, 2], color=c_black, zorder=1)

ax1.plot(list_smear, kappas_L[:, 0], label="Lorentz", color=c_red, linestyle='dashed', zorder=1)
ax1.plot(list_smear, kappas_L[:, 1], color=c_red, linestyle='dashed', zorder=1)
ax1.plot(list_smear, kappas_L[:, 2], color=c_red, linestyle='dashed', zorder=1)

ax1.hlines(convergence_pos_y[2], list_smear[0], list_smear[-1], color=c_purple, zorder=0,alpha=0.6,lw=5)
ax1.text(convergence_pos_x[2], convergence_pos_y[2]+0.05, f"{temp_list[2]} K", color=c_purple, fontsize=14)
ax1.hlines(convergence_pos_y[1], list_smear[0], list_smear[-1], color=c_orange, zorder=0,alpha=0.6,lw=5)
ax1.text(convergence_pos_x[1], convergence_pos_y[1]+0.05, f"{temp_list[1]} K", color=c_orange, fontsize=14)
ax1.hlines(convergence_pos_y[0], list_smear[0], list_smear[-1], color=c_cyan, zorder=0,lw=5,alpha=0.6)
ax1.text(convergence_pos_x[0], convergence_pos_y[0]+0.05, f"{temp_list[0]} K", color=c_cyan, fontsize=14)

# ax1.vlines(3 * np.max(IC_datasets[0][3]) / (3 * 120), 0, 2.5, linestyle='dotted', color='grey')
# ax1.text(10, 0.1, r'$3 \Delta \omega_{avg}$' ,fontsize=14)

ax1.text(title_upper[0], title_upper[1], r'$\bf{q}$ mesh 5x5x5', fontsize=14)
# ax1.text(title_upper[2], title_upper[3], structure_labels[structure_idx], fontsize=14)


# set limits on x and y axes, ticks on axes, set scale to logs, set labels
ax1.set_ylim([0., ylim])
# ax1.
# ax1.set_xticks(np.arange(0,40,2.5),minor=True)
# ax1.set_xticks(np.arange(0,36,5),minor=False)
ax1.set_yticks(np.arange(0, ylim+0.1, 0.5), minor=True)
ax1.set_yticks(np.arange(0, ylim+0.1, 1.0), minor=False)

# ax1.set_yscale('log')
ax1.set_xscale('log')

# ax1.set_xlabel(r'$\hbar \eta \;\left(\rm{cm}^{-1}\right)$',fontsize=14)
ax1.set_ylabel(r'$\kappa \;\left(\rm{W} \cdot \rm{m}^{-1} \cdot \rm{K}^{-1}\right)$',fontsize=14, labelpad=8)


handles, labels = ax1.get_legend_handles_labels()
handles_mod=handles.copy()
labels_mod=labels.copy()


ax1.legend(loc='center right', fancybox=True, shadow=True, ncol=1, fontsize=14,
          columnspacing=1.,scatterpoints=1,handletextpad=0.2,handlelength=0.8,frameon=False)

ax1.tick_params(labelsize=14, direction='in',which='minor',bottom=True, top=True, left=True, right=True,length=2,pad=7)
ax1.tick_params(labelsize=14, direction='in',which='major',bottom=True, top=True, left=True, right=True,length=4,pad=7)

# --------------------------------- #

ax2.plot(list_smear, kappas_G_gamma[:, 0], label="Gauss", color=c_black, zorder=1)
ax2.plot(list_smear, kappas_G_gamma[:, 1], color=c_black, zorder=1)
ax2.plot(list_smear, kappas_G_gamma[:, 2], color=c_black, zorder=1)


ax2.plot(list_smear, kappas_L_gamma[:, 0], label="Lorentz", color=c_red, linestyle='dashed', zorder=1)
ax2.plot(list_smear, kappas_L_gamma[:, 1], color=c_red, linestyle='dashed', zorder=1)
ax2.plot(list_smear, kappas_L_gamma[:, 2], color=c_red, linestyle='dashed', zorder=1)


ax2.hlines(convergence_pos_y[2], list_smear[0], list_smear[-1], color=c_purple, zorder=0,alpha=0.6,lw=5)
ax2.text(convergence_pos_x[2], convergence_pos_y[2]+0.05, f"{temp_list[2]} K", color=c_purple, fontsize=14)
ax2.hlines(convergence_pos_y[1], list_smear[0], list_smear[-1], color=c_orange, zorder=0,alpha=0.6,lw=5)
ax2.text(convergence_pos_x[1], convergence_pos_y[1]+0.05, f"{temp_list[1]} K", color=c_orange, fontsize=14)
ax2.hlines(convergence_pos_y[0], list_smear[0], list_smear[-1], color=c_cyan, zorder=0,lw=5,alpha=0.6)
ax2.text(convergence_pos_x[0], convergence_pos_y[0]+0.05, f"{temp_list[0]} K", color=c_cyan, fontsize=14)

# ax2.vlines(3 * np.max(IC_datasets[0][3]) / (3 * 120), 0, 2.5, linestyle='dotted', color='grey')
# ax2.text(10, 0.1, r'$3 \Delta \omega_{avg}$' ,fontsize=14)

ax2.text(title_lower[0], title_lower[1], r'$\bf{q}$ gamma', fontsize=14)
# ax2.text(title_lower[2], title_lower[3], structure_labels[structure_idx], fontsize=14)

# set limits on x and y axes, ticks on axes, set scale to logs, set labels
ax2.set_ylim([0., ylim])
ax2.set_xlim([0.025, 60])
# ax2.set_xticks(np.arange(0,18,2.5),minor=True)
# ax2.set_xticks(np.arange(0,16,5),minor=False)
ax2.set_yticks(np.arange(0, ylim+0.1, 0.5), minor=True)
ax2.set_yticks(np.arange(0, ylim+0.1, 1.0), minor=False)

# ax2.set_yscale('log')
# ax2.set_xscale('log')

ax2.set_xlabel(r'$\hbar \eta \;\left(\rm{cm}^{-1}\right)$',fontsize=14)
ax2.set_ylabel(r'$\kappa \;\left(\rm{W} \cdot \rm{m}^{-1} \cdot \rm{K}^{-1}\right)$',fontsize=14, labelpad=8)


handles, labels = ax2.get_legend_handles_labels()
handles_mod=handles.copy()
labels_mod=labels.copy()


ax2.legend(loc='center right', fancybox=True, shadow=True, ncol=1, fontsize=14,
          columnspacing=1.,scatterpoints=1,handletextpad=0.2,handlelength=0.8,frameon=False)

ax2.tick_params(labelsize=14, direction='in',which='minor',bottom=True, top=True, left=True, right=True,length=2,pad=7)
ax2.tick_params(labelsize=14, direction='in',which='major',bottom=True, top=True, left=True, right=True,length=4,pad=7)


save_directory = "."
name_file_save=f't9_216_convergence_test.pdf'
scale = 1.
fig.set_size_inches(6.5*scale, 4.5*scale)
fig.savefig(save_directory + "/" + name_file_save, dpi=300, bbox_inches="tight",transparent=True)