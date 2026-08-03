import os
import sys
import math
import numpy as np
import scipy.constants as const
import itertools as itools
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--vars", type = str, nargs = '+',
                    help = "List of variables to plot, if unset plots all")
parser.add_argument("-l", "--log", action = 'store_true',
                    help = "Plot appropriate plots (e.g. energy) as log plots")
parser.add_argument("-p", "--png", action = 'store_true',
                    help = "Plot as individual png files, not combined pdf")
parser.add_argument("-d", "--data", type = str,
                    help = "File containing data to be plotted")
args = parser.parse_args()

def saveFigure(fig, filePath, dpi, attempts = 100, delay = 1, tight = False):
   # Tries to save figure; on failure, retries with delay
   # This command can fail with multiprocessing as matplotlib uses threads
   for ii in range(attempts):
      try:
         if tight is True:
            fig.savefig(filePath,dpi=dpi,transparent=False,bbox_inches='tight')
         else:
            fig.savefig(filePath,dpi=dpi,transparent=False)
      except (SyntaxError, RuntimeError) as s:
         if ii == 0:
            print("Plot \"" + os.path.basename(filePath) + "\" failed, beginning reattempt loops")
         if ii < attempts - 1:
            time.sleep(delay)
            continue
         else:
            print("Failed to plot \"" + os.path.basename(filePath) + "\"")
            sys.stdout.flush()
            continue
      if ii > 0:
         print("Plot \"" + os.path.basename(filePath) + "\" generated on attempt " + str(ii+1))
         sys.stdout.flush()
      break

def snap_lims(old_min, old_max, locator = None, major_ticks = None):
   # Snap given lims to tick marks given by locator, or list of tick points
   # Also returns tick locations inside resulting range
   # If locator does not return range containing requested
   if locator is not None:
      major_ticks = locator.tick_values(old_min, old_max)
      if isinstance(locator,ticker.SymmetricalLogLocator): # SymmetricalLogLocator might not produce ticks beyond the data limits, so extend until it does
         if old_min < major_ticks[0] or old_max > major_ticks[-1]:
            test_max = max(abs(old_min), abs(old_max))
            test_min = -test_max
            major_ticks = locator.tick_values(test_min, test_max)
            while old_min < major_ticks[0] or old_max > major_ticks[-1]:
               test_min *= 10
               test_max *= 10
               major_ticks = locator.tick_values(test_min, test_max)
         
   elif major_ticks is None:
      warnings.warn("No locator or list of tick marks provided to snap_lims", UserWarning)
      return (old_min,old_max),[]
   
   # Find first tick points outside of (or equal to) lims
   new_min,new_max = expand_values(old_min, old_max, major_ticks)

   restricted_ticks = [x for x in major_ticks if x >= new_min and x <= new_max]

   return (new_min,new_max),restricted_ticks

def find_tickStep(maxCrdRange, min_ticks = 5, max_ticks = 15, bias = 'min'):
   # Find good tickstep for plot range
   # If max_ticks/min_ticks < approx. 2.5 then solutions in requested range may not be possible
   # With bias == 'max' finds valid solution with most ticks, 'min' finds least
   ii_start = math.floor(math.log10(min_ticks/abs(maxCrdRange))) - 1
   ii_end = math.floor(math.log10(max_ticks/abs(maxCrdRange))) + 1
   if bias == 'max':
      tick_range = itools.product(range(ii_end,ii_start - 1,-1),(5,2,1))
   else:
      tick_range = itools.product(range(ii_start,ii_end + 1),(1,2,5))
   for ii,jj in tick_range:
      recip_tickStep = jj*10**ii
      NticksMax = abs(maxCrdRange)*recip_tickStep
      if NticksMax >= min_ticks and NticksMax <= max_ticks:
         mantissa = 10//jj # Since jj = 1,2,5 gives first digit exactly
         tickStep = mantissa*10**(-ii - 1) # Constructed directly to avoid float error
         if mantissa == 10:
            mantissa = 1
         return tickStep,mantissa # exit function if good tick step found

   # No suitable tickstep found; treat max_ticks as more important
   # print("WARNING: no good tick step found: NticksMax = " + str(NticksMax) + ", tickStep = " + str(tickStep))
   old_NticksMax = old_jj = old_ii = 0
   for ii,jj in itools.product(range(ii_start,100),(1,2,5)):
      recip_tickStep = jj*10**ii
      NticksMax = abs(maxCrdRange)*recip_tickStep
      if NticksMax >= min_ticks and old_NticksMax >= 1:
         mantissa = 10//old_jj # Since jj = 1,2,5 gives first digit exactly
         tickStep = mantissa*10**(-old_ii - 1) # Constructed directly to avoid float error
         if mantissa == 10:
            mantissa = 1
         return tickStep,mantissa # exit function if good tick step found
      old_NticksMax = NticksMax
      old_jj = jj
      old_ii = ii

   # Still no suitable tickstep found; just find something
   # print("WARNING: no decent tick step found: NticksMax = " + str(NticksMax) + ", tickStep = " + str(tickStep))
   old_NticksMax = old_jj = old_ii = 0
   for ii,jj in itools.product(range(ii_start,100),(1,2,5)):
      recip_tickStep = jj*10**ii
      NticksMax = abs(maxCrdRange)*recip_tickStep
      if NticksMax >= min_ticks:
         mantissa = 10//jj # Since jj = 1,2,5 gives first digit exactly
         tickStep = mantissa*10**(-ii - 1) # Constructed directly to avoid float error
         if mantissa == 10:
            mantissa = 1
         return tickStep,mantissa # exit function if good tick step found

   # Still no suitable tickstep
   print("WARNING: no bad tick step found: NticksMax = " + str(NticksMax) + ", tickStep = " + str(tickStep))
   return 1,1

def plotFigure(t_data, var_data, var_label, unit = None, rescale_x = True, rescale_y = False, log = False):
   # Generate single dataset figure
   fig = Figure(figsize = figureSize, frameon = True, layout = "compressed")
   ax = fig.subplots(nrows = 1, ncols = 1, squeeze = False)[0,0]
   
   xRange = np.array((np.min(t_data).item(),np.max(t_data).item()))
   if log is True:
      yRange = yRange = np.array((np.min(var_data.where(var_data > 0.0)).item(),np.max(var_data.where(var_data > 0.0)).item()))
   else:
      yRange = np.array((np.min(var_data).item(),np.max(var_data).item()))
   
   if any(np.isnan(yRange)) or all(np.abs(yRange) == 0):
      yRange[0] = -0.05
      yRange[1] = 0.05
   elif np.max(np.abs(yRange)) - np.min(np.abs(yRange)) < 1e-8*np.max(np.abs(yRange)):
      mid = np.mean(yRange)
      if mid == 0:
         diff = 0.05
      else:
         diff = 0.05*mid
      yRange[0] = mid - diff*0.05
      yRange[1] = mid + diff*0.05
   
   # x_pts = np.linspace(*xRange)
   # y_pts = 1e-8 * np.exp(0.35*math.sqrt(2*314208961640850*const.e**2/(const.m_e*const.epsilon_0))*x_pts)
         
   if rescale_x is True:
      x_log = math.floor(math.log(np.max(np.abs(xRange)), 1e3))
      t_data = t_data/1e3**x_log
      xRange = xRange/1e3**x_log
      # x_pts /= 1e3**x_log
   else:
      x_log = 0

   if rescale_y is True:
      y_log = math.floor(math.log(np.max(np.abs(yRange)), 1e3))
      var_data = var_data/1e3**y_log
      yRange = yRange/1e3**y_log
      # y_pts /= 1e3**y_log
   else:
      y_log = 0
   
   _,xLocators = get_MultLocators(*xRange)
   if log is True:
      try:
         yRange,yLocators = get_LogLocators(*yRange)
      except ValueError:
         log = False
         yRange,yLocators = get_MultLocators(*yRange)
   else:
      yRange,yLocators = get_MultLocators(*yRange)
   
   scatter = ax.scatter(t_data, var_data, 0.1)
   # line = ax.plot(t_data, var_data, linewidth = 0.5)

   # ax.plot(x_pts, y_pts, markersize = 0, linewidth =  0.5)
   
   ax.xaxis.set_major_locator(xLocators[0])
   ax.xaxis.set_minor_locator(xLocators[1])
   ax.yaxis.set_major_locator(yLocators[0])
   ax.yaxis.set_minor_locator(yLocators[1])

   if log is True:
      ax.set_yscale('log')
   
   ax.set_xlim(xRange)
   ax.set_ylim(yRange)
   
   prefix_list = ["q","r","y","z","a","f","p","n",r"$\mu$","m","","k","M","G","T","P","E","Z","Y","R","Q"]

   prefix_table = dict(zip(range(-10,11),prefix_list))
   
   x_label = "x [" + prefix_table[x_log] + "m]"
   if unit is None:
      y_label = var_label
   else:
      y_label = var_label + " [" + prefix_table[y_log] + unit + "]"
      
   ax.set_xlabel(x_label)
   ax.set_ylabel(y_label)

   return fig

def expand_values(vmin, vmax, locations):
   # Expand vmin and vmax (away from each other) to the first match in list of acceptable values locations
   # Return vmin or vmax if smaller/larger than list
   try:
      new_min = next(x for x in sorted(locations, reverse = True) if x <= vmin)
   except StopIteration:
      new_min = vmin
   try:
      new_max = next(x for x in sorted(locations) if x >= vmax)
   except StopIteration:
      new_max = vmax
   
   return new_min,new_max

def get_MultLocators(vmin, vmax, min_ticks = 2, max_ticks = 9, bias = 'max', snap_minor = False):
   # Return major and minor MultipleLocators for ticks given min and max value of data, and expand min and max to first tick for given locator
   tickStep,factor = find_tickStep(vmax - vmin, min_ticks, max_ticks, bias = bias)
   major_locator = ticker.MultipleLocator(tickStep)
   if factor == 2:
      minor_count = 4
   else:
      minor_count = 5
   minor_locator = ticker.AutoMinorLocator(minor_count)
   
   if snap_minor is True:
      (new_min,new_max),_ = snap_lims(vmin, vmax, locator = minor_locator)
   else:
      (new_min,new_max),_ = snap_lims(vmin, vmax, locator = major_locator)
   
   return (new_min,new_max),(major_locator,minor_locator)

def get_LogLocators(vmin, vmax, vstep = 1, max_ticks = 9, match_unity = True, snap_minor = False):
   # Return major and minor LogLocators for ticks given min and max value of data, and expand min and max to first locator
   # With match_unity adjusts limits and locators so that unity (10⁰) is one of the major tick points
   major_locator = ticker.LogLocator(10**vstep, numticks = max_ticks)
   
   # If log locator does not place ticks at consecutive powers (e.g. it places them at 1, 100, 10000 etc.) then minor ticks should be placed at missing powers, otherwise multiples of the last power
   tickLocs = major_locator.tick_values(vmin, vmax)
   step = round(np.mean(tickLocs[1:]/tickLocs[:-1]))
   if step > 10:
      if match_unity is True:
         major_locator = ticker.LogLocator(step, numticks = max_ticks)
      minor_locator = ticker.LogLocator(10, numticks = max_ticks*9)
   else:
      minor_locator = ticker.LogLocator(10, subs = np.arange(1,10),
                                        numticks = max_ticks*9)

   if snap_minor is True:
      (new_min,new_max),_ = snap_lims(vmin, vmax, locator = minor_locator)
   else:
      (new_min,new_max),_ = snap_lims(vmin, vmax, locator = major_locator)
   
   new_tickLocs = major_locator.tick_values(new_min, new_max)
   new_step = round(np.mean(new_tickLocs[1:]/new_tickLocs[:-1]))
   if new_step != step:
      # Step size has changed, retry recursively until step size does not change
      (new_min,new_max),(major_locator,minor_locator) = get_LogLocators(vmin, vmax, vstep + 1, max_ticks, match_unity, snap_minor)
      
   return (new_min,new_max),(major_locator,minor_locator)

figDpi = 100
figResolutionX = 450
figResolutionY = 450
figureSize = (figResolutionX/figDpi,figResolutionY/figDpi)

data = xr.open_dataset(args.data)

if args.vars is None:
   fig_list = [
      "faceBx",
      "faceBy",
      "faceBz",
      "nodeEx",
      "nodeEy",
      "nodeEz",
      "cellJx",
      "cellJy",
      "cellJz",
      "cellT",
      "cellUx",
      "cellUy",
      "cellUz",
      "cellN",
      "divB",
      "divE"
   ]
else:
   fig_list = args.vars

var_list = data.keys()
pop_list = [x[:-7] for x in var_list if x.endswith("_cellJx")]

figs_field = {}
for fig_name in fig_list:
   match fig_name:
      case "faceBx":
         y_unit = "T"
      case "faceBy":
         y_unit = "T"
      case "faceBz":
         y_unit = "T"
      case "nodeEx":
         y_unit = "V/m"
      case "nodeEy":
         y_unit = "V/m"
      case "nodeEz":
         y_unit = "V/m"
      case "divB":
         y_unit = "T/m"
      case "divE":
         y_unit = r"V/m$^2$"
      case _:
         continue

   dat = data[fig_name].isel(t = -1, y = 0, z = 0).data
   dat[np.isnan(dat)] = 0
   
   figs_field[fig_name] = plotFigure(data.x, dat, fig_name, y_unit)

if args.png:
   for fig_name,fig in figs_field.items():
      saveFigure(fig, fig_name + ".png", figDpi)
else:
   with PdfPages("field_data.pdf", keep_empty = False) as pdf:
      for fig in figs_field.values():
         pdf.savefig(fig, dpi = figDpi, transparent = False, bbox_inches = 'tight')
         plt.close(fig)

figs_pop = {}
for pop in pop_list:
   for fig_name in fig_list:
      match fig_name:
         case "cellUx":
            y_unit = "m/s"
         case "cellUy":
            y_unit = "m/s"
         case "cellUz":
            y_unit = "m/s"
         case "cellJx":
            y_unit = r"A/m$^2$"
         case "cellJy":
            y_unit = r"A/m$^2$"
         case "cellJz":
            y_unit = r"A/m$^2$"
         case _:
            continue
      
      dat = data[pop + "_" + fig_name].isel(t = -1, y = 0, z = 0).data
      dat[np.isnan(dat)] = 0
         
      figs_pop[pop + "_" + fig_name] = plotFigure(data.x, dat, pop + " " + fig_name, y_unit)

full_pop_list = [x[:-2] for x in pop_list if x[-2:] == "_l"]
for pop in full_pop_list:
   for fig_name in fig_list:
      match fig_name:
         case "cellUx":
            y_unit = "m/s"
         case "cellUy":
            y_unit = "m/s"
         case "cellUz":
            y_unit = "m/s"
         case "cellJx":
            y_unit = r"A/m$^2$"
         case "cellJy":
            y_unit = r"A/m$^2$"
         case "cellJz":
            y_unit = r"A/m$^2$"
         case "cellT":
            y_unit = "K"
         case "cellN":
            y_unit = r"m$^-3$"
         case _:
            continue

      dens_l = data[pop + "_l_cellN"].isel(t = -1, y = 0, z = 0).data
      dens_r = data[pop + "_r_cellN"].isel(t = -1, y = 0, z = 0).data
      
      dens_tot = dens_l + dens_r

      dat_l = data[pop + "_l_" + fig_name].isel(t = -1, y = 0, z = 0).data
      dat_l[np.isnan(dat_l)] = 0

      dat_r = data[pop + "_r_" + fig_name].isel(t = -1, y = 0, z = 0).data
      dat_r[np.isnan(dat_r)] = 0
      
      dat = (dat_l*dens_l + dat_r*dens_r)/dens_tot
      
      figs_pop[pop + "_" + fig_name] = plotFigure(data.x, dat, pop + " " + fig_name, y_unit)

if args.png:
   for fig_name,fig in figs_pop.items():
      saveFigure(fig, fig_name + ".png", figDpi)
else:
   with PdfPages("pop_data.pdf", keep_empty = False) as pdf:
      for fig in figs_pop.values():
         pdf.savefig(fig, dpi = figDpi, transparent = False, bbox_inches = 'tight')
         plt.close(fig)
