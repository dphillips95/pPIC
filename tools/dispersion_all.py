import sys
import os
import itertools as itools
import functools as ftools
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xarray as xr
import numpy as np
import math
import scipy as sp
import scipy.constants as const
import argparse
import warnings
import multiprocessing as mp
from scipy.fft import fftfreq, fft2, fftshift, rfft2, rfftfreq
matplotlib.use('agg')

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--Ncores", help = "Number of CPU threads",
                    type = int, default = 1)
parser.add_argument("-w", "--Whistler", type = int,
                    help = "Plot whistler and ion cyclotron mode exact solutions; 1 = hybrid solution, 2 = full solution")
parser.add_argument("-g", "--gyro", action = 'store_true', help = "Scale frequency to gyrofrequency and wavenumber to ion inertial length")
parser.add_argument("-x", "--xZoom", type = int, default = 1,
                    help = "Zoom in x-axis by given factor towards origin")
parser.add_argument("-y", "--yZoom", type = int, default = 1,
                    help = "Zoom in y-axis by given factor towards origin")
parser.add_argument("-m", "--Mirror", type = int, help = "Include other regions of dispersion; 1 = -Frequency, 2 = -Wavenumber")
parser.add_argument("-F", "--Frequency", action = 'store_true', help = "Plot using actual frequency, not angular frequency")
parser.add_argument("-i", "--initial", help = "Initial time step",
                    type = int, default = 0)
parser.add_argument("-f", "--final", help = "Final time step",
                    type = int)
parser.add_argument("--cmap", default = "inferno",
                    help = "cmap to use for plotting (see matplotlib docs)")
parser.add_argument("-u", "--Unfinished", action = 'store_true',
                    help = "Skip last file in directory if still being written to")
parser.add_argument("-P", "--Pretty", action = 'store_true',
                    help = "Prettify the plots for posters, e.g. larger text etc.")
parser.add_argument("-d", "--Dimension", type = int, default = 0,
                    help = "Dimension direction to perform analysis in; 0 = x, 1 = y, 2 = z")
parser.add_argument("-L", "--Logarithmic", action = 'store_true',
                    help = "Plot frequency and wavenumber logarithmically")
parser.add_argument("vars", nargs = '*',
                    help = "Variables for which dispersion is to be plotted; include full name. Non-scalar quantities will plot all components and appropriate circularly polarised forms. Leave empty to plot all available variables.")
args = parser.parse_args()

def makeVec(dataset):
   # Recombine split data into vectors
   vec_list = set(x for x in [y[:-1] for y in var_name] if x + "x" in var_name and x + "y" in var_name and x + "z" in var_name)
   sca_list = set(x for x in var_name if not (x[:-1] + "x" in var_name and x[:-1] + "y" in var_name and x[:-1] + "z" in var_name))

   new_dataset = dataset[sca_list]

   for vec in vec_list:
      x = dataset[vec + "x"]
      y = dataset[vec + "y"]
      z = dataset[vec + "z"]

      vec_data = xr.concat((x,y,z), dim = 'comp')

      new_dataset[vec] = vec_data

   if len(vec_list) > 0:
      new_dataset = new_dataset.transpose('t','z','y','x','comp')

   return new_dataset
      
def doPlot():
   # plotters

   global v0
   global vA
   global CFL
   
   k = fftfreq(nx, dx)
   omega = fftfreq(nt, time_step)

   k = fft_NyquistShift(fftshift(k))
   omega = fft_NyquistShift(fftshift(omega))

   if len(k)%2 == 0:
      k[-1] = -k[-1]

   if len(omega)%2 == 0:
      omega[-1] = -omega[-1]
      
   k = k[d1:d2]
   omega = omega[c1:c2]

   dk = 2*math.pi/(dx*nx)
   domega = 2*math.pi/(time_step*nt)

   kmin = 2*math.pi*min(k)-dk/2
   kmax = 2*math.pi*max(k)+dk/2
   omegamin = 2*math.pi*min(omega)-domega/2
   omegamax = 2*math.pi*max(omega)+domega/2
   
   if args.Logarithmic:
      kmin_abs = 2*math.pi*min(abs(x) for x in k if x != 0)
      omegamin_abs = 2*math.pi*min(abs(x) for x in omega if x != 0)
      k = np.logspace(math.log(kmin_abs,10), math.log(kmax,10), 1000)
      omega = np.logspace(math.log(omegamin_abs,10), math.log(omegamax,10), 1000)
      k = np.concatenate((np.append(np.flip(-k), 0), k))
      omega = np.concatenate((np.append(np.flip(-omega), 0), omega))
   else:
      k = np.linspace(-kmax, kmax, 1000)
      omega = np.linspace(-omegamax, omegamax, 1000)

   if args.Whistler is not None:
      if min(omega) <= ion_gyro_freq and max(omega) >= ion_gyro_freq:
         omega = np.append(omega, ion_gyro_freq)
      if min(omega) <= -ion_gyro_freq and max(omega) >= -ion_gyro_freq:
         omega = np.append(omega, -ion_gyro_freq)
      if args.Whistler == 2:
         if min(omega) <= electron_gyro_freq and max(omega) >= electron_gyro_freq:
            omega = np.append(omega, electron_gyro_freq)
         if min(omega) <= -electron_gyro_freq and max(omega) >= -electron_gyro_freq:
            omega = np.append(omega, -electron_gyro_freq)
   
   
   if args.Whistler == 1:
      whistler_right = ftools.partial(whistler, right = True, hybrid = True, kmax = kmax)
      whistler_left = ftools.partial(whistler, right = False, hybrid = True, kmax = kmax)
   else:
      whistler_right = ftools.partial(whistler, right = True, hybrid = False, kmax = kmax)
      whistler_left = ftools.partial(whistler, right = False, hybrid = False, kmax = kmax)

   if args.Ncores == 1:
      tmp_r = map(whistler_right, omega)
      tmp_l = map(whistler_left, omega)
   elif __name__ == '__main__':
      with mp.Pool(args.Ncores) as pool:
         tmp_r = pool.map(whistler_right, omega)
         tmp_l = pool.map(whistler_left, omega)
   
   if args.Whistler is not None:
      whistler_p_r = np.array(list(tmp_r))
      whistler_p_l = np.array(list(tmp_l))
      
   # if args.gyro:
   #    k *= ion_gyro_radius/(2*math.pi)
   #    omega /= ion_gyro_freq
   #    v0 /= ion_gyro_freq*ion_gyro_radius
   #    vA /= ion_gyro_freq*ion_gyro_radius
   #    CFL /= ion_gyro_freq*ion_gyro_radius
   #    kmin *= ion_gyro_radius
   #    kmax *= ion_gyro_radius
   #    omegamin /= ion_gyro_freq
   #    omegamax /= ion_gyro_freq

   # if args.Whistler is not None:
   #    if args.gyro:
   #       whistler_p_r *= ion_gyro_radius
   #       whistler_p_l *= ion_gyro_radius
   #    # whistler_p_r = v0*k + whistler_p_r
   #    # whistler_p_l = v0*k + whistler_p_l

   kFactor = 1
   omegaFactor = 1

   if args.gyro:
      kFactor *= inertial_ion
      omegaFactor /= ion_gyro_freq

   if args.Frequency:
      kFactor *= 1e3/(2*math.pi)
      omegaFactor *= 1/(2*math.pi)
         
   k *= kFactor
   dk *= kFactor
   omega *= omegaFactor
   domega *= omegaFactor
   v0 *= omegaFactor/kFactor
   vA *= omegaFactor/kFactor
   CFL *= omegaFactor/kFactor
   kmin *= kFactor
   kmax *= kFactor
   omegamin *= omegaFactor
   omegamax *= omegaFactor
      
   if args.Whistler is not None:
      whistler_p_r *= kFactor
      whistler_p_l *= kFactor
   
      whistler_p_r = [x for _,x in sorted(zip(omega, whistler_p_r), key=lambda pair: pair[0])]
      whistler_p_l = [x for _,x in sorted(zip(omega, whistler_p_l), key=lambda pair: pair[0])]
   omega = sorted(omega)

   if args.Ncores == 1:
      pow_data = map(ff_pow, var_data_ff)
   elif __name__ == '__main__':
      with mp.Pool(args.Ncores) as pool:
         pow_data = pool.map(ff_pow, var_data_ff)
   
   kLims = [kmin, kmax]
   omegaLims = [omegamin, omegamax]
   if args.Whistler is not None:
      whistlers = np.array([whistler_p_r, whistler_p_l])

   filenames = [x + "_ff" for x in var_name_expand]

   titles = []

   for x in var_name_expand:
      if x == "faceB_x":
         titles.append('$B_x$')
      elif x == "faceB_y":
         titles.append('$B_y$')
      elif x == "faceB_z":
         titles.append('$B_z$')
      elif x == "nodeE_x":
         titles.append('$E_x$')
      elif x == "nodeE_y":
         titles.append('$E_y$')
      elif x == "nodeE_z":
         titles.append('$E_z$')
      elif x == "cellUe_x":
         titles.append('$U^e_x$')
      elif x == "cellUe_y":
         titles.append('$U^e_y$')
      elif x == "cellUe_z":
         titles.append('$U^e_z$')
      elif x == "faceB_yzp":
         titles.append('$B_y+iB_z$')
      elif x == "faceB_yzn":
         titles.append('$B_y-iB_z$')
      elif x == "nodeE_yzp":
         titles.append('$E_y+iE_z$')
      elif x == "nodeE_yzn":
         titles.append('$E_y-iE_z$')
      elif x == "cellUe_yzp":
         titles.append('$U^e_y+iU^e_z$')
      elif x == "cellUe_yzn":
         titles.append('$U^e_y-iU^e_z$')
      elif x.startswith("v_") and x.endswith("_yzp"):
         tmp_title = x[:-5]
         titles.append(tmp_title + '_y+i' + tmp_title + '_z')
      elif x.startswith("v_") and x.endswith("_yzn"):
         tmp_title = x[:-5]
         titles.append(tmp_title + '_y-i' + tmp_title + '_z')
      elif x == "n_e":
         titles.append('$n_e$')
      else:
         titles.append(x)

   if args.Whistler is None:
      gen_plot = ftools.partial(plotter, kLims = kLims,
                                omegaLims = omegaLims, k = k, omega = omega,
                                dk = dk, domega = domega)
   else:
      gen_plot = ftools.partial(plotter, kLims = kLims,
                                omegaLims = omegaLims, k = k, omega = omega,
                                dk = dk, domega = domega, whistlers = whistlers)

   # if args.Ncores == 1:
   #    itools.starmap(gen_plot, zip(pow_data, filenames, titles))
   # elif __name__ == '__main__':
   #    with mp.Pool(args.Ncores) as pool:
   #       pool.starmap(gen_plot, zip(pow_data, filenames, titles))

   for data,filename,title in zip(pow_data, filenames, titles):
      gen_plot(data, filename, title)

   Byzp_loc = var_name_expand.index("faceB_yzp")
   Byzn_loc = var_name_expand.index("faceB_yzn")

   if args.Whistler is None:
      plotter2([pow_data[Byzp_loc],pow_data[Byzn_loc]], filename = "faceB_yzpn_ff", titles = ["$B_y+iB_z$", "$B_y-iB_z$"], kLims = kLims, omegaLims = omegaLims, k = k, omega = omega, dk = dk, domega = domega)
   else:
      plotter2([pow_data[Byzp_loc],pow_data[Byzn_loc]], filename = "faceB_yzpn_ff", titles = ["$B_y+iB_z$", "$B_y-iB_z$"], kLims = kLims, omegaLims = omegaLims, k = k, omega = omega, dk = dk, domega = domega, whistlers = whistlers)

def annotate_line(ax,line,label,end='last',position=None):
   if position is not None:
      x = position[0]
      y = position[1]
   else:
      if end == 'last':
         data_point = -1
      elif end == 'first':
         data_point = 0
      else:
         data_point = end
      
      x = line.get_xdata()[data_point]
      y = line.get_ydata()[data_point]

   xbounds = ax.get_xbound()
   ybounds = ax.get_ybound()

   xbounds_mid = np.average(xbounds)/20
   ybounds_mid = np.average(ybounds)/20
   xbounds = [xbounds[0] + xbounds_mid, xbounds[1] - xbounds_mid]
   ybounds = [ybounds[0] + ybounds_mid, ybounds[1] - ybounds_mid]
   
   if x >= max(xbounds):
      # Right edge
      ha = 'left'
      va = 'center_baseline'
      xy = (1,y)
      xytext = (6,0)
      xycoords = ax.get_yaxis_transform()
   elif y >= max(ybounds):
      # Top edge
      ha = 'center'
      va = 'baseline'
      xy = (x,1)
      xytext = (30,6)
      color = line.get_color()
      xycoords = ax.get_xaxis_transform()
      textcoords = "offset points"
   elif x <= min(xbounds):
      # Left edge
      ha = 'right'
      va = 'center_baseline'
      xy = (0,y)
      xytext = (-6,0)
      xycoords = ax.get_yaxis_transform()
   elif y <= min(ybounds):
      # Bottom edge
      ha = 'center'
      va = 'top'
      xy = (x,0)
      xytext = (0,-6)
      xycoords = ax.get_xaxis_transform()
   else:
      # Middle
      ha = 'center'
      va = 'center_baseline'
      xy = (x,y)
      xytext = (6,6)
      xycoords = (ax.get_xaxis_transform(),ax.get_yaxis_transform())

   ax.annotate(label, xy = xy, xytext = xytext, color = line.get_color(),
               xycoords = xycoords, textcoords = "offset points", va = va,
               ha = ha)
      
def plotter2(data, filename, titles, kLims, omegaLims, k, omega, dk, domega, whistlers = None):
   Nsub = 1
   fig, ax = plt.subplots(ncols=2,figsize=figureSize,frameon=True,layout='constrained')
   
   noLog = False
   if max([*data[0].flat] + [*data[1].flat]) == 0:
      noLog = True
   else:
      minLog = np.percentile([x for x in [*data[0].flat] + [*data[1].flat] if x>0], 5)
   
   for i in range(2):
      if noLog:
         col = ax[i].imshow(data[i], interpolation = 'none', origin = 'lower',
                            cmap = args.cmap, extent = kLims + omegaLims,
                            aspect = 'auto')
      else:
         col = ax[i].imshow(data[i], interpolation = 'none', cmap = args.cmap,
                            norm = colors.SymLogNorm(minLog), origin = 'lower',
                            extent = kLims + omegaLims, aspect = 'auto')
      ax[i].set_title(titles[i], pad = 10)
      
      #if args.Logarithmic:
      #   ax[i].axline([kLims[0],kLims[0]*CFL], slope = CFL, color='r', linestyle='dashed', linewidth=2)
      #   ax[i].axline([kLims[0],kLims[0]*(v0+vA)], slope = v0+vA, color='k', linestyle='dashed', linewidth=1)
      #   ax[i].axline([kLims[0],kLims[0]*v0], slope = v0, color='k', linestyle='dotted', linewidth=1)
      #   ax[i].axline([kLims[0],kLims[0]*(v0-vA)], slope = v0-vA, color='k', linestyle='dashed', linewidth=1)
      #   ax[i].axline([kLims[0],-kLims[0]*CFL], slope = -CFL, color='r', linestyle='dashed', linewidth=2)
      #else:
      # ax[i].axline([0,0], slope = CFL, color='r', linestyle='dashed', linewidth=2)
      # ax[i].axline([0,0], slope = v0+vA, color='k', linestyle='dashed', linewidth=1)
      # ax[i].axline([0,0], slope = v0, color='k', linestyle='dotted', linewidth=1)
      # ax[i].axline([0,0], slope = v0-vA, color='k', linestyle='dashed', linewidth=1)
      # ax[i].axline([0,0], slope = -CFL, color='r', linestyle='dashed', linewidth=2)
      # ax[i].plot(k, k*CFL, color='r', linestyle='dashed', linewidth=2)
      alfvenLine = ax[i].plot(k, k*(v0+vA), color='k', linestyle='dashed', linewidth=2)
      ax[i].plot(k, k*v0, color='k', linestyle='dotted', linewidth=2)
      ax[i].plot(k, k*(v0-vA), color='k', linestyle='dashed', linewidth=2)
      # ax[i].plot(k, -k*CFL, color='r', linestyle='dashed', linewidth=2)
      
      if not args.gyro:
         if args.Frequency:
            xtickInertial_ion = 1e3/inertial_ion
            ytickIon_gyro_freq = ion_gyro_freq/(2*math.pi)
         else:
            xtickInertial_ion = 2*math.pi*1e3/inertial_ion
            ytickIon_gyro_freq = ion_gyro_freq

         inertialLine = ax[i].axvline(xtickInertial_ion, color='b', linestyle='dashed', linewidth=2)
         gyroLine = ax[i].axhline(ytickIon_gyro_freq, color='b', linestyle='dashed', linewidth=2)
      
      if args.gyro:
         ax[i].set_xlabel('$kd_i$')
         if i == 0:
            ax[i].set_ylabel('$\omega/\omega_i$')
      elif args.Frequency:
         ax[i].set_xlabel('Wavenumber [km$^{-1}$]')
         if i == 0:
            ax[i].set_ylabel('Frequency [Hz]')
      else:
         ax[i].set_xlabel('Angular Wavenumber [rad km$^{-1}$]')
         if i == 0:
            ax[i].set_ylabel('Angular Frequency [rad Hz]')
            
      plt.gcf().gca().tick_params(which='both',direction=tickDir,length=tickLength,width=tickWidth)
      
      if whistlers is not None:
         whistlerLine = ax[i].plot(whistlers[0,:], omega + v0*whistlers[0,:], color = 'k', linestyle = 'dashed', linewidth = 2)
         cycloLine = ax[i].plot(whistlers[1,:], omega + v0*whistlers[1,:], color = 'k', linestyle = 'dashed', linewidth = 2)
         ax[i].plot(-whistlers[0,:], omega - v0*whistlers[0,:], color = 'k', linestyle = 'dotted', linewidth = 2)
         ax[i].plot(-whistlers[1,:], omega - v0*whistlers[1,:], color = 'k', linestyle = 'dotted', linewidth = 2)
         
      if args.Logarithmic:
         if args.Mirror is None:
            ax[i].set_xscale('log')
            ax[i].set_yscale('log')
         elif args.Mirror == 1:
            ax[i].set_xscale('log')
            ax[i].set_yscale('symlog', linthresh = domega, linscale = 0.1)
         else:
            ax[i].set_xscale('symlog', linthresh = dk, linscale = 0.1)
            ax[i].set_yscale('symlog', linthresh = domega, linscale = 0.1)

      ax[i].set_xbound(kLims)
      ax[i].set_ybound(omegaLims)
      if args.gyro or not args.Logarithmic:
         ax[i].set_box_aspect(1.0)
      else:
         ax[i].set(aspect = 'equal')

      if not args.gyro:
         if kLims[0] <= xtickInertial_ion and kLims[1] >= xtickInertial_ion:
            ax[i].annotate("$d_i$", xy=(xtickInertial_ion,0), xytext=(-6,-15), color=inertialLine.get_color(),
                           xycoords = ax[i].get_xaxis_transform(), textcoords="offset points", size = 20, va="center")
            annotate_line(ax[i], line = inertialLine, label = "$d_i$",
                          end = 'last')
         if omegaLims[0] <= ytickIon_gyro_freq and omegaLims[1] >= ytickIon_gyro_freq:
            annotate_line(ax[i], line = gyroLine, label = "$\omega_i$",
                          end = 'first')

      annotate_line(ax[i], line = alfvenLine[0], label = "Alfvén",
                    end = 'last')
      if whistlers is not None:
         annotate_line(ax[i], line = whistlerLine[0], label = "R-mode",
                       end = 'last')
         position = next((x,y) for x,y in zip(cycloLine[0].get_xdata(),cycloLine[0].get_ydata()) if not (x <= max(ax[i].get_xbound()) and y <= max(ax[i].get_ybound())) and not (x == float("inf") or x == float("-inf") or y == float("inf") or y == float("-inf")))
         annotate_line(ax[i], line = cycloLine[0], label = "L-mode",
                       position = position)
      
   cbar = fig.colorbar(col, ax=ax, aspect=40, shrink=0.5)
   cbar.set_label('Magnitude [nT]',)

   attempts = 100
   delay = 1
   saveFigure(fig, "./" + filename + '.png', dpi = figDpi, attempts = attempts, delay = delay)
   # fig.savefig("./" + filename + '.png',bbox_inches='tight')
   plt.close(fig)
   
def plotter(data, filename, title, kLims, omegaLims, k, omega, dk, domega, whistlers = None):
   Nsub = 1
   fig, ax = plt.subplots(figsize=figureSize,frameon=True,layout='constrained')
   ax.set_xlim(kLims)
   ax.set_ylim(omegaLims)
   if data.max() == 0:
      col = ax.imshow(data, interpolation = 'none', origin = 'lower',
                      cmap = args.cmap, extent = kLims + omegaLims)
   else:
      minLog = np.percentile([x for x in list(data.flat) if x>0], 5)
      col = ax.imshow(data, interpolation = 'none', origin = 'lower',
                      cmap = args.cmap, norm = colors.SymLogNorm(minLog),
                      extent = kLims + omegaLims)
   ax.title.set_text(title)
   ax.set(aspect = 'auto')
   if args.gyro:
      # ax.set_xlabel('$kr_i$')
      ax.set_xlabel('$kd_i$')
      ax.set_ylabel('$\omega/\omega_i$')
   else:
      ax.set_xlabel('$k$ $[rad/m]$')
      ax.set_ylabel('$\omega$ $[rad/s]$')
   #if args.Logarithmic:
   #   ax.axline([kLims[0],kLims[0]*CFL], slope = CFL, color='r', linestyle='dashed', linewidth=2)
   #   ax.axline([kLims[0],kLims[0]*(v0+vA)], slope = v0+vA, color='k', linestyle='dashed', linewidth=1)
   #   ax.axline([kLims[0],kLims[0]*v0], slope = v0, color='k', linestyle='dotted', linewidth=1)
   #   ax.axline([kLims[0],kLims[0]*(v0-vA)], slope = v0-vA, color='k', linestyle='dashed', linewidth=1)
   #   ax.axline([kLims[0],-kLims[0]*CFL], slope = -CFL, color='r', linestyle='dashed', linewidth=2)
   #else:
   # ax.axline([0,0], slope = CFL, color='r', linestyle='dashed', linewidth=2)
   # ax.axline([0,0], slope = v0+vA, color='k', linestyle='dashed', linewidth=1)
   # ax.axline([0,0], slope = v0, color='k', linestyle='dotted', linewidth=1)
   # ax.axline([0,0], slope = v0-vA, color='k', linestyle='dashed', linewidth=1)
   # ax.axline([0,0], slope = -CFL, color='r', linestyle='dashed', linewidth=2)
   # ax.plot(k, k*CFL, color='r', linestyle='dashed', linewidth=2)
   ax.plot(k, k*(v0+vA), color='k', linestyle='dashed', linewidth=1)
   ax.plot(k, k*v0, color='k', linestyle='dotted', linewidth=1)
   ax.plot(k, k*(v0-vA), color='k', linestyle='dashed', linewidth=1)
   # ax.plot(k, -k*CFL, color='r', linestyle='dashed', linewidth=2)

   plt.gcf().gca().tick_params(which='both',direction=tickDir,length=tickLength,width=tickWidth)
   
   if whistlers is not None:
      ax.plot(whistlers[0,:], omega + v0*whistlers[0,:], color = 'k', linestyle = 'dotted', linewidth = 2)
      ax.plot(whistlers[1,:], omega + v0*whistlers[1,:], color = 'k', linestyle = 'dotted', linewidth = 2)
      ax.plot(-whistlers[0,:], omega - v0*whistlers[0,:], color = 'k', linestyle = 'dotted', linewidth = 2)
      ax.plot(-whistlers[1,:], omega - v0*whistlers[1,:], color = 'k', linestyle = 'dotted', linewidth = 2)
   if args.Logarithmic:
      if args.Mirror is None:
         plt.xscale('log')
         plt.yscale('log')
      elif args.Mirror == 1:
         plt.xscale('log')
         plt.yscale('symlog', linthresh = domega, linscale = 0.1)
      else:
         plt.xscale('symlog', linthresh = dk, linscale = 0.1)
         plt.yscale('symlog', linthresh = domega, linscale = 0.1)
   cbar = fig.colorbar(col, ax=ax, aspect=40)
   cbar.set_label('')

   attempts = 100
   delay = 1
   saveFigure(fig, "./" + filename + '.png', dpi = figDpi, attempts = attempts, delay = delay)
   # fig.savefig("./" + filename + '.png',bbox_inches='tight')
   plt.close(fig)

def saveFigure(fig, filePath, dpi, attempts, delay):
   for i in range(attempts):
      try:
         fig.savefig(filePath,dpi=dpi,transparent=False)
      except (SyntaxError, RuntimeError) as s:
         if i < attempts - 1:
            time.sleep(delay)
            continue
         else:
            print("Failed to plot \"" + os.path.basename(filePath) + "\"")
      if i > 0:
         print("Plot \"" + os.path.basename(filePath) + "\" generated on attempt " + str(i+1))
      break
   sys.stdout.flush()
   
def whistler(omega, right, hybrid, kmax):
   if omega == None:
      out = 0
   else:
      if hybrid:
         if right:
            if omega == -ion_gyro_freq:
               out = kmax
            else:
               # pre = omega**2 - omega*oscillation_ion**2/(omega + ion_gyro_freq) + omega*lim_electron
               pre = omega**2*(1 + oscillation_ion**2/(ion_gyro_freq*(ion_gyro_freq + omega)))
               if pre < 0:
                  out = float('Inf')
               else:
                  out = np.sqrt(pre)/const.c
         else:
            if omega == ion_gyro_freq:
               out = kmax
            else:
               # pre = omega**2 - omega*oscillation_ion**2/(omega - ion_gyro_freq) - omega*lim_electron
               pre = omega**2*(1 + oscillation_ion**2/(ion_gyro_freq*(ion_gyro_freq - omega)))
               if pre < 0:
                  out = float('Inf')
               else:
                  out = np.sqrt(pre)/const.c
      else:
         if right:
            if omega == -ion_gyro_freq or omega == electron_gyro_freq:
               out = kmax
            else:
               # pre = omega**2 - omega*oscillation_ion**2/(omega + ion_gyro_freq) - omega*oscillation_electron**2/(omega - electron_gyro_freq)
               pre = omega**2*(1 - (oscillation_ion**2 + oscillation_electron**2)/((omega + ion_gyro_freq)*(omega - electron_gyro_freq)))
               if pre < 0:
                  out = float('Inf')
               else:
                  out = np.sqrt(pre)/const.c
         else:
            if omega == ion_gyro_freq or omega == -electron_gyro_freq:
               out = kmax
            else:
               # pre = omega**2 - omega*oscillation_ion**2/(omega - ion_gyro_freq) - omega*oscillation_electron**2/(omega + electron_gyro_freq)
               pre = omega**2*(1 - (oscillation_ion**2 + oscillation_electron**2)/((omega - ion_gyro_freq)*(omega + electron_gyro_freq)))
               if pre < 0:
                  out = float('Inf')
               else:
                  out = np.sqrt(pre)/const.c

      if omega<0:
         out *= -1
   
   return out

def getvA(B_dat, ne_dat, t_pt):
   
   # B_dat2 = np.sum(B_dat**2, axis = -1)[t_pt,:]
   B_dat2 = B_dat[t_pt,:,0]**2
   ne_dat = ne_dat[t_pt,:]

   tmp_shape = np.shape(B_dat2)

   for ii,shape in enumerate(tmp_shape):
      if shape > 2:
         B_dat2 = np.apply_along_axis(shrinkArray_1D, ii, B_dat2)
         ne_dat = np.apply_along_axis(shrinkArray_1D, ii, ne_dat)
   
   vA = np.sqrt(B_dat2/(ne_dat*const.mu_0*const.m_p))
   
   return np.average(vA)

def shrinkArray_1D(array):

   return array[1:-1]

def getParam(B_dat, ne_dat, n_dat, T_dat, Ue_dat = None):
   tmp_shape = np.shape(ne_dat)[1:]

   for ii,shape in enumerate(tmp_shape):
      if shape > 2:
         B_dat = np.apply_along_axis(shrinkArray_1D, ii+1, B_dat[1:,:,:])
         ne_dat = np.apply_along_axis(shrinkArray_1D, ii+1, ne_dat[1:,:])
         n_dat = np.apply_along_axis(shrinkArray_1D, ii+2, n_dat[:,1:,:])
         T_dat = np.apply_along_axis(shrinkArray_1D, ii+2, T_dat[:,1:,:])
         if Ue_dat is not None:
            Ue_dat = np.apply_along_axis(shrinkArray_1D, ii+1, Ue_dat[1:,:,:])
   
   ne_av = np.average(ne_dat)

   av_tmp = np.average(n_dat*T_dat, axis = (0))

   av_tmp = [x/y for x,y in zip(list(av_tmp.flat), list(ne_dat.flat)) if y > 0]
   
   T_av = np.average(av_tmp)
   
   B_pt = np.average(np.sqrt(np.sum(B_dat[-1,:,:]**2, axis = (1))))
   ne_pt = np.average(ne_dat[-1,:])
   if Ue_dat is not None:
      Ue_pt = np.average(np.sqrt(np.sum(Ue_dat[-1,:,:]**2, axis = (1))))
   
   ion_gyro_freq = B_pt*const.e/const.m_p
   ion_velocity = np.sqrt(3*const.k*T_av/const.m_p)
   ion_gyro_radius = const.m_p*ion_velocity/(B_pt*const.e)

   electron_gyro_freq = B_pt*const.e/const.m_e
   if Ue_dat is not None:
      electron_gyro_radius = const.m_e*Ue_pt/(B_pt*const.e)
   else:
      electron_gyro_radius = const.m_e*ion_velocity/(B_pt*const.e)

   oscillation_ion = np.sqrt(ne_av*const.e**2/(const.epsilon_0*const.m_p))
   oscillation_electron = np.sqrt(ne_av*const.e**2/(const.epsilon_0*const.m_e))

   inertial_ion = np.sqrt(const.m_p/(ne_av*const.e**2*const.mu_0))
   inertial_electron = np.sqrt(const.m_e/(ne_av*const.e**2*const.mu_0))
   
   lim_electron = ne_av*const.e/(B_pt*const.epsilon_0)

   print("Ion Gyro Frequency: " + str(ion_gyro_freq) + " rad/s = " + str(2*math.pi/(ion_gyro_freq*dt)) + " dt")
   print("Ion Gyro Radius: " + str(ion_gyro_radius) + " m = " + str(ion_gyro_radius/dx) + " dx")
   print("Electron Gyro Frequency: " + str(electron_gyro_freq) + " rad/s = " + str(2*math.pi/(electron_gyro_freq*dt)) + " dt")
   print("Electron Gyro Radius: " + str(electron_gyro_radius) + " m = " + str(electron_gyro_radius/dx) + " dx")
   print("Ion Plasma Frequency: " + str(oscillation_ion) + " rad/s = " + str(2*math.pi/(oscillation_ion*dt)) + " dt")
   print("Electron Plasma Frequency: " + str(oscillation_electron) + " rad/s = " + str(2*math.pi/(oscillation_electron*dt)) + " dt")
   print("Ion Inertial Length: " + str(inertial_ion) + " m = " + str(inertial_ion/dx) + " dx")
   print("Electron Inertial Length: " + str(inertial_electron) + " m = " + str(inertial_electron/dx) + " dx")
   
   return [ion_gyro_freq, ion_gyro_radius, electron_gyro_freq,
           electron_gyro_radius, oscillation_ion, oscillation_electron,
           inertial_ion, inertial_electron, lim_electron]

def getv0(n_dat, v_dat, t_pt):
   
   n_dat = np.array(n_dat)
   v_dat = np.array(v_dat)
   
   n_dat_tot = np.sum(n_dat,axis = 0)
   v_dat_shape = list(np.shape(v_dat))

   v_dat_shape.pop(1)
   
   m_tmp = np.empty(v_dat_shape)
   v0_tmp = np.empty(v_dat_shape[1:])
   
   for i in range(3):
      m_tmp[:,:,i] = n_dat[:,t_pt,:]*v_dat[:,t_pt,:,i]
      for j in range(v_dat_shape[1]):
         if n_dat_tot[t_pt,j] == 0:
            v0_tmp[j,i] = float("NaN")
         else:
            v0_tmp[j,i] = np.sum(m_tmp[:,j,i],axis = 0)/n_dat_tot[t_pt,j]
      
   return np.nanmean(-v0_tmp, axis = (0))

def rfft2_to_fft2(rff_dat):
   im_shape = list(rff_dat.shape)
   im_shape[1] = 2*(im_shape[1] - 1)

   ff = np.zeros(im_shape, dtype = rff_dat.dtype)
   ff[:,:(im_shape[1]+2)//2] = rff_dat
   ff[0,(im_shape[1]+4)//2:] = np.conjugate(np.flip(rff_dat[0,1:(im_shape[1]-2)//2]))
   ff[1:,(im_shape[1]+4)//2:] = np.conjugate(np.flip(rff_dat[1:,1:(im_shape[1]-2)//2]))
   
   return ff

def fft_NyquistShift(fft_dat):
   result = fft_dat

   ax = []
   for ii,dim in enumerate(fft_dat.shape):
      if dim%2 == 0:
         ax.append(ii)
   
   shift = np.apply_over_axes(shunt_nD, fft_dat, axes = ax)

   return shift
   
def shunt_nD(array, axis):
   
   return np.apply_along_axis(shunt_1D, axis, array)
                              
def shunt_1D(array):
   n = array.shape[0]

   result = np.zeros(shape = [n], dtype = array.dtype)

   result[:n-1] = array[1:]
   result[-1] = array[0]

   return result

def fft_total(dat, real):

   if real:
      return fft_complete(dat, True)
   else:
      return fft_complete(dat, False)

def fft_complete(dat, real):

   if real:
      ff = rfft2_to_fft2(rfft2(dat))
   else:
      ff = fft2(dat)

   ff = fft_NyquistShift(fftshift(ff))
   ff = ff[c1:c2,d1:d2]
   # ff = np.flip(ff, axis = 0)
   return ff

def ff_pow(ff):

   # return 2*(np.abs(ff)*dx*time_step)**2/(xInterval*tInterval)
   # return np.abs(ff)**2
   return np.abs(ff)#/(dx*time_step)

def repeat_stack(arr, count, axis):
    return np.stack([arr for _ in range(count)], axis = axis)

figDpi = 100
figResolutionX = 1920
figResolutionY = 960
figureSize = (figResolutionX/figDpi,figResolutionY/figDpi)
if args.Pretty:
   matplotlib.rcParams.update({'font.size': 20})
else:
   matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['lines.linewidth'] = 2
tickDir = "out"
tickLength = 2
tickWidth = 1

xr_data = xr.open_dataset("fields.nc")

var_name = set(xr_data.variables) - set(('t','x','y','z','timestep'))
var_name = list(var_name)
var_name.sort()

xmin = xr_data.x.min().item()
xmax = xr_data.x.max().item()
ymin = xr_data.y.min().item()
ymax = xr_data.y.max().item()
zmin = xr_data.z.min().item()
zmax = xr_data.z.max().item()

nx = xr_data.x.shape[0]
ny = xr_data.y.shape[0]
nz = xr_data.z.shape[0]

xInterval = xmax - xmin 
dx = xInterval/nx # should be dx = dy = dz in rhybrid
ncells = nx*ny*nz # total number of cells

n_list = [i for i in var_name if i.endswith("_nodeN")]
species_count = len(n_list)

xr_data = makeVec(xr_data)

xr_data = xr_data.squeeze()
if args.Dimension == 0:
   xr_data.squeeze().drop_vars(('y','z'))
elif args.Dimension == 1:
   xr_data.squeeze().drop_vars(('x','z'))
elif args.Dimension == 2:
   xr_data.squeeze().drop_vars(('x','y'))

try:
   args.final += 1
except TypeError:
   pass

xr_data = xr_data.isel(t = slice(args.initial,args.final))

nt = xr_data.t.shape[0]

tmp1 = xr_data.t.min().item()
tmp2 = xr_data.t.max().item()

tInterval = (tmp2 - tmp1)
time_step = tInterval/(nt - 1)

dt = time_step

CFL = dx/dt

if args.Dimension == 0:
   av_axes = [0,1]
elif args.Dimension == 1:
   av_axes = [0,2]
elif args.Dimension == 2:
   av_axes = [1,2]

var_name = list(set(xr_data.variables) - set(('t','x','y','z')))
var_data = [xr_data[name] for name in var_name]

var_real = [True]*len(var_name)

n_list = [i for i in var_name if i.endswith("_nodeN")]

n_dataset = []
v_dataset = []
T_dataset = []

for x,y in zip(var_name,var_data):
   if x.endswith("_nodeN"):
      n_dataset.append(y)
   elif x.endswith("_nodeU"):
      v_dataset.append(y)
   elif x.endswith("_nodeT"):
      T_dataset.append(y)

n_dataset = np.array(n_dataset)
v_dataset = np.array(v_dataset)
T_dataset = np.array(T_dataset)
      
n_tmp = []

for x in n_list:
   if x.startswith("e-"):
      n_tmp.append(var_data[var_name.index(x)])

var_name.append("n_e")
var_data.append(np.sum(n_tmp, axis = 0))
var_real.append(True)

for x,y in zip(var_name,var_data):
   if len(np.shape(y)) > 2:
      if args.Dimension == 0:
         var_name.append(x + "_yzp")
         var_data.append(y[:,:,1] + 1j*y[:,:,2])
         var_real.append(False)
         var_name.append(x + "_yzn")
         var_data.append(y[:,:,1] - 1j*y[:,:,2])
         var_real.append(False)
      elif args.Dimension == 1:
         var_name.append(x + "_xzp")
         var_data.append(y[:,:,0] + 1j*y[:,:,2])
         var_real.append(False)
         var_name.append(x + "_xzn")
         var_data.append(y[:,:,0] - 1j*y[:,:,2])
         var_real.append(False)
      else:
         var_name.append(x + "_xyp")
         var_data.append(y[:,:,0] + 1j*y[:,:,1])
         var_real.append(False)
         var_name.append(x + "_xyn")
         var_data.append(y[:,:,0] - 1j*y[:,:,1])
         var_real.append(False)

B_loc = var_name.index("faceB")
ne_loc = var_name.index("n_e")
Ue_present = False
if "cellUe" in var_name:
   Ue_present = True
   Ue_loc = var_name.index("cellUe")

vA = getvA(var_data[B_loc], var_data[ne_loc], nt-1)

v0 = getv0(n_dataset, v_dataset, nt-1)
v0 = v0[0]

if Ue_present:
   [ion_gyro_freq, ion_gyro_radius, electron_gyro_freq,
    electron_gyro_radius, oscillation_ion, oscillation_electron,
    inertial_ion, inertial_electron, lim_electron] = getParam(
       var_data[B_loc], var_data[ne_loc], n_dataset, T_dataset,
       var_data[Ue_loc])
else:
   [ion_gyro_freq, ion_gyro_radius, electron_gyro_freq,
    electron_gyro_radius, oscillation_ion, oscillation_electron,
    inertial_ion, inertial_electron, lim_electron] = getParam(
       var_data[B_loc], var_data[ne_loc], n_dataset, T_dataset)
#ion_gyro_freq = tmp[0]
#ion_gyro_radius = tmp[1]
#electron_gyro_freq = tmp[2]
#electron_gyro_radius = tmp[3]
#oscillation_ion = tmp[4]
#oscillation_electron = tmp[5]
#inertial_ion = tmp[6]
#inertial_electron = tmp[7]
#lim_electron = tmp[8]

#for i in range(nt):
# By[i] = By[i] - sum(By[i])/nx

#By = By - sum(sum(By))/(nx*nt)

var_data[B_loc] = var_data[B_loc]*1e9

var_name_expand = []
var_data_expand = []
var_real_expand = []

for tmp_name, tmp_data, tmp_real in zip(var_name, var_data, var_real):
   if np.shape(tmp_data)[-1] == 3:
      var_name_expand.extend([tmp_name + "_x", tmp_name + "_y", tmp_name + "_z"])
      var_real_expand.extend([tmp_real]*3)
      for x in np.split(tmp_data, 3, axis = 2):
         var_data_expand.append(np.squeeze(x))
   else:
      var_name_expand.append(tmp_name)
      var_data_expand.append(tmp_data)
      var_real_expand.append(tmp_real)
      
window = repeat_stack(np.hamming(nt), nx, 1)

var_data_window = []

for x in var_data_expand:
   var_data_window.append(x*window)

if args.Mirror is None:
   c1 = (nt-1)//2
   c2 = nt
   if args.yZoom>1:
      c2 = c1 + int((nt+2)/(2*args.yZoom))
   if args.Logarithmic:
      c1 += 1
else:
   if args.yZoom>1:
      c2 = nt - int(nt/2 - (nt/2)/args.yZoom)
   else:
      c2 = nt
   c1 = nt - c2

if args.Mirror == 2:
   if args.xZoom>1:
      d2 = nx - int(nx/2 - (nx/2)/args.xZoom)
   else:
      d2 = nx
   d1 = nx - d2
else:
   d1 = (nx-1)//2
   d2 = nx
   if args.xZoom>1:
      d2 = d1 + int((nx+2)/(2*args.xZoom))
   if args.Logarithmic:
      d1 += 1
   # if zoom>1:
   #    mid = (d1 + d2)/2
   #    diff = d2 - mid
   #    d1 = int(mid - diff/zoom)
   #    d2 = int(mid + diff/zoom)

for ii,x in enumerate(var_data_expand):
   try:
      var_data_expand[ii] = x.to_numpy()
   except AttributeError:
      pass
      
if args.Ncores == 1:
   var_data_ff = itools.starmap(fft_total, zip(var_data_expand, var_real_expand))
elif __name__ == '__main__':
   with mp.Pool(args.Ncores) as pool:
      var_data_ff = pool.starmap(fft_total, zip(var_data_expand, var_real_expand))

doPlot()
