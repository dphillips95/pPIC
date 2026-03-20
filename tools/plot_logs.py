import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
import xarray as xr

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

def plotFigure(t_data, var_data, var_label):
   # Generate single dataset figure

   fig = Figure(figsize = figureSize, frameon = True, layout = "compressed")

   ax = fig.subplots(nrows = 1, ncols = 1, squeeze = False)[0,0]

   scatter = ax.scatter(t_data, var_data)
   line = ax.plot(t_data, var_data)

   ax.set_xlabel("time [s]")
   ax.set_ylabel(var_label)

   return fig

figDpi = 100
figResolutionX = 450
figResolutionY = 450
figureSize = (figResolutionX/figDpi,figResolutionY/figDpi)

logs = xr.open_dataset("logs.h5")

figs = []
figs.append(plotFigure(logs.t, logs.avgFaceBx, "avg(faceBx) [T]"))
figs.append(plotFigure(logs.t, logs.avgFaceBy, "avg(faceBy) [T]"))
figs.append(plotFigure(logs.t, logs.avgFaceBz, "avg(faceBz) [T]"))
figs.append(plotFigure(logs.t, logs.avgFaceB_mag, "avg(|faceB|) [T]"))
figs.append(plotFigure(logs.t, logs.maxFaceB_mag, "max(|faceB|) [T]"))
figs.append(plotFigure(logs.t, logs.energy_B, "energy(faceB) [T]"))

figs.append(plotFigure(logs.t, logs.avgNodeEx, "avg(nodeEx) [V/m]"))
figs.append(plotFigure(logs.t, logs.avgNodeEy, "avg(nodeEy) [V/m]"))
figs.append(plotFigure(logs.t, logs.avgNodeEz, "avg(nodeEz) [V/m]"))
figs.append(plotFigure(logs.t, logs.avgNodeE_mag, "avg(|nodeE|) [V/m]"))
figs.append(plotFigure(logs.t, logs.maxNodeE_mag, "max(|nodeE|) [V/m]"))
figs.append(plotFigure(logs.t, logs.energy_E, "energy(nodeE) [J]"))

figs.append(plotFigure(logs.t, logs.avgCellRhoQ, "avg(cellRhoQ) [C]"))

figs.append(plotFigure(logs.t, logs.avgCellJx, "avg(cellJx) [A]"))
figs.append(plotFigure(logs.t, logs.avgCellJy, "avg(cellJy) [A]"))
figs.append(plotFigure(logs.t, logs.avgCellJz, "avg(cellJz) [A]"))
figs.append(plotFigure(logs.t, logs.avgCellJ_mag, "avg(|cellJ|) [A]"))
figs.append(plotFigure(logs.t, logs.maxCellJ_mag, "max(|cellJ|) [A]"))

var_list = logs.keys()
pop_list = (x[:-13] for x in var_list if x.endswith("_avgCellU_mag"))

for pop in pop_list:
   figs.append(plotFigure(logs.t, logs[pop + "_Np"], pop + " macroparticles"))
   figs.append(plotFigure(logs.t, logs[pop + "_parts"], pop + " particles"))
   figs.append(plotFigure(logs.t, logs[pop + "_avgCellUx"], pop + " avg(Ux) [m/s]"))
   figs.append(plotFigure(logs.t, logs[pop + "_avgCellUy"], pop + " avg(Uy) [m/s]"))
   figs.append(plotFigure(logs.t, logs[pop + "_avgCellUz"], pop + " avg(Uz) [m/s]"))
   figs.append(plotFigure(logs.t, logs[pop + "_avgCellU_mag"], pop + " avg(|U|) [m/s]"))
   figs.append(plotFigure(logs.t, logs[pop + "_maxCellU_mag"], pop + " max(|U|) [m/s]"))
   figs.append(plotFigure(logs.t, logs[pop + "_avgCellJix"], pop + " avg(cellJx) [V/m]"))
   figs.append(plotFigure(logs.t, logs[pop + "_avgCellJiy"], pop + " avg(cellJy) [V/m]"))
   figs.append(plotFigure(logs.t, logs[pop + "_avgCellJiz"], pop + " avg(cellJz) [V/m]"))
   figs.append(plotFigure(logs.t, logs[pop + "_avgCellJi_mag"], pop + " avg(|cellJ|) [V/m]"))
   figs.append(plotFigure(logs.t, logs[pop + "_maxCellJi_mag"], pop + " max(|cellJ|) [V/m]"))
   figs.append(plotFigure(logs.t, logs[pop + "_KE"], pop + " Kinetic Energy [J]"))

figs.append(plotFigure(logs.t, logs.total_energy, "Total Energy [J]"))

with PdfPages("logs.pdf", keep_empty = False) as pdf:
   for fig in figs:
      pdf.savefig(fig, dpi = figDpi, transparent = False, bbox_inches = 'tight')
      plt.close(fig)
