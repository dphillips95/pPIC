import sys
import os
import math
import numpy as np
import scipy as sp
import scipy.constants as const
from scipy.sparse.linalg import gmres
from timeit import default_timer as timer
import numba
from numba import int64,float64
from numba import types
from numba.typed import Dict as nb_dict
from numba import njit
from numba.experimental import jitclass as jitclass
import configparser
import argparse

from interpolators import face2cell,face2node,face2r,cell2node,cell2face,cell2r,node2face,node2cell,node2r,face2cell_njit,face2node_njit,face2r_njit,cell2node_njit,cell2face_njit,cell2r_njit,node2face_njit,node2cell_njit,node2r_njit

from tools import split_axis

parser = argparse.ArgumentParser()
parser.add_argument("--config", type = str, default = "pPIC.cfg",
                    help = "Configuration file; input \"help\" to this argument to get a decription of all config parameters")
parser.add_argument("--unsafe", action = 'store_const', const = True,
                    help = "Ignore dt and dx bounds")
parser.add_argument("--steps", type = int,
                    help = "Number of full time steps to run")
parser.add_argument("--dt", type = float,
                    help = "Time step size (s). If this violates stability bounds then code will abort unless \"unsafe\" option is used")
parser.add_argument("--mass-ratio", type = float,
                    help = "Ratio of proton to electron mass. Defaults to physical value")
parser.add_argument("--theta", type = float,
                    help = "Theta parameter for maxwell integration; must be theta >= 0.5, theta = 0.5 is 2nd order, theta > 0.5 suppresses some oscillations")
parser.add_argument("--rtol", type = float,
                    help = "gmres relative tolerance, norm(b - A @ x) <= rtol*norm(b)")
parser.add_argument("--atol", type = float,
                    help = "gmres absolute tolerance, norm(b - A @ x) <= atol")
parser.add_argument("--seed", type = int, help = "Rng seed")
args = parser.parse_args()

def configHelp():
   # Help with configuration file
   print("Configuration file help")
   print("")
   print("IMPORTANT NOTE: To leave an option blank, omit the equals '=' sign, do NOT delete from config file.")
   print("")
   print("")
   print("[main] options:")
   print("")
   print("   unsafe (bool): Ignore dt and dx bounds")
   print("")
   print("   seed (int): RNG seed")
   print("")
   print("   dimensions (int): Restrict to n dimensions; forces higher dimensions to single box size, dimensions dk = dx")
   print("")
   print("   1v (bool): Restrict system to 1D velocity; automatically enforces 1D domain (dimensions = 1)")
   print("")
   print("")
   print("[simulation] options:")
   print("")
   print("   steps (int): Number of full time steps to run")
   print("")
   print("   dt (s): Time step size. If this violates stability bounds then code will abort unless \"unsafe\" option is used")
   print("")
   print("   theta (float): Theta parameter for maxwell integration; must be 0.5 <= theta <= 1.0, theta = 0.5 is 2nd order, theta > 0.5 suppresses some oscillations")
   print("")
   print("   species_list (str ...): List of species names, separated by spaces. Each species should have its own section of the configuration file [<Species_Name>]. One species must be \"e-\" (electrons).")
   print("")
   print("   mass_ratio (float): Ratio of proton to electron mass")
   print("")
   print("   rtol (float): gmres relative tolerance, norm(b - A @ x) <= rtol*norm(b)")
   print("")
   print("   atol (float): gmres absolute tolerance, norm(b - A @ x) <= atol")
   print("")
   print("   Vdt_dx_cap (float): Maximum ratio of time step to dx/maxV, i.e. caps particle movement to given fraction of cell width, default is 0.5")
   print("")
   print("")
   print("[domain] options:")
   print("   x_min (m): Minimum x-coordinate")
   print("")
   print("   x_max (m): Maximum x-coordinate")
   print("")
   print("   y_min (m): Minimum y-coordinate, ignored if dimensions < 2. If this and y_max are both unset then places boxes around y = 0, i.e. y_min = -y_size*dx/2")
   print("")
   print("   y_max (m): Maximum y-coordinate, ignored if dimensions < 2. If this and y_min are both unset then places boxes around y = 0, i.e. y_max = y_size*dx/2")
   print("")
   print("   z_min (m): Minimum z-coordinate, ignored if dimensions < 3. If this and z_max are both unset then places boxes around z = 0, i.e. z_min = -z_size*dx/2")
   print("")
   print("   z_max (m): Maximum z-coordinate, ignored if dimensions < 2. If this and z_min are both unset then places boxes around z = 0, i.e. z_max = z_size*dx/2")
   print("")
   print("   x_size (int): Number of cells in x-dimension")
   print("")
   print("   y_size (int): Number of cells in y-dimension, ignored if dimensions < 2")
   print("")
   print("   z_size (int): Number of cells in z-dimension, ignored if dimensions < 3")
   print("")
   print("[<Species_Name>] options (all species follow this format), one of these must be \"e-\"")
   print("")
   print("   Electron (bool): If this particle is an electron/electron scaled, set this to True. Sets mass scaling according to \"mass_ratio\" in \"[simulation]\"")
   print("")
   print("   mass (m_p/m_e): Mass of particle species, in units of protons, or electrons if \"Electron\" is True")
   print("")
   print("   charge (e): Charge of particle species in units of elementary charge (i.e. protons = +1 and electrons = -1")
   print("")
   print("   temperature (K): Species initial temperature")
   print("")
   print("   velocity (m/s): Species initial bulk velocity")
   print("")
   print("   density (#/m^3): Species initial number density. For electrons, leave blank to match local charge density and ensure quasi-neutrality")
   print("")
   print("   macroparticles_per_cell (#): Species number of macroparticles per cell")

def parse_multiarg_config(config, section, param_name, type_fun = str, divider = " "):
   # Parse parameter from config file with multiple arguments
   # "type_fun" is applied to each argument; e.g. if parameters are integers use "type_fun = int" to make arguments integers. default is string - i.e. do nothing
   # Use divider to change divider symbol; default is space
   tmp = config.get(section, param_name, fallback = "")
   if tmp is None:
      return None
   tmp = tmp.split(divider)
   if tmp[0] == "":
      return None
   return [type_fun(x) for x in tmp]

def readConfig(fName, args):
   # Parse config file
   config = configparser.ConfigParser(allow_no_value = True)
   config.read_file(open(fName))

   args_dict = {key:val for key,val in vars(args).items() if val is not None}
   try:
      args.unsafe = config.getboolean("main", "unsafe", vars = args_dict,
                                      fallback = False)
   except AttributeError:
      args.unsafe = False
   
   try:
      args.seed = config.getint("main", "seed", vars = args_dict, fallback = 0)
   except TypeError:
      args.seed = 0
   
   try:
      args.steps = config.getint("simulation", "steps", vars = args_dict,
                                 fallback = 100)
   except TypeError:
      args.steps = 100

   try:
      args.mass_ratio = config.getfloat("simulation", "mass_ratio",
                                        vars = args_dict,
                                        fallback = const.m_p/const.m_e)
   except TypeError:
      args.mass_ratio = const.m_p/const.m_e
   
   args.dt = config.get("simulation", "dt", vars = args_dict)
   if args.dt is not None:
      args.dt = float(args.dt)

   args.theta = config.get("simulation", "theta", vars = args_dict)
   if args.theta is None:
      args.theta = 0.5
   else:
      args.theta = float(args.theta)
   
   args.rtol = config.get("simulation", "rtol", vars = args_dict)
   if args.rtol is None:
      args.rtol = 1e-6
   else:
      args.rtol = float(args.rtol)
   
   args.atol = config.get("simulation", "atol", vars = args_dict)
   if args.atol is None:
      args.atol = 0.0
   else:
      args.atol = float(args.atol)
   
   return args,config

class my_timers:
   def __init__(self):
      self.timers = {
         'init':0.0,
         'alpha':0.0,
         'current':0.0,
         'mass matrices':0.0,
         'maxwell':0.0,
         'build A':0.0,
         'build b':0.0,
         'gmres':0.0,
         'mover':0.0,
         'lorentz':0.0,
         'total':0.0
         }

   def tic(self, _timer_):
      self.timers[_timer_] -= timer()
   
   def toc(self, _timer_):
      self.timers[_timer_] += timer()
   
   def reset(self, _timer_):
      self.timers[_timer_] = 0.0

dims_spec = [
   ("x_min", float64),
   ("x_max", float64),
   ("y_min", float64),
   ("y_max", float64),
   ("z_min", float64),
   ("z_max", float64),
   ("dx", float64),
   ("dy", float64),
   ("dz", float64),
   ("vol", float64),
   ("x_size", int64),
   ("y_size", int64),
   ("z_size", int64),
   ("Ncells_total", int64),
   ("dim_scalar", int64[:]),
   ("dim_vector", int64[:]),
   ("x_range", int64[:]),
   ("y_range", int64[:]),
   ("z_range", int64[:])
]

@jitclass(dims_spec)
class Dims:
   def __init__(self, lims, spacing, sizes):
      # Pseudo-dict containg dimension params
      (self.x_min,self.x_max),(self.y_min,self.y_max),(self.z_min,self.z_max) = lims
      self.dx,self.dy,self.dz = spacing
      self.vol = self.dx*self.dy*self.dz
      self.x_size,self.y_size,self.z_size = sizes
      self.Ncells_total = self.x_size*self.y_size*self.z_size
      self.dim_scalar = np.array([self.z_size,self.y_size,self.x_size], dtype = int64)
      self.dim_vector = np.array([self.z_size,self.y_size,self.x_size,3], dtype = int64)
      self.x_range = np.array(list(range(self.x_size)), dtype = int64)
      self.y_range = np.array(list(range(self.y_size)), dtype = int64)
      self.z_range = np.array(list(range(self.z_size)), dtype = int64)

class Pop:
   def __init__(self, name, q, m, w, electron, Np, T = None, v = None):
      # Initial fill of particle population
      # If Np = 0 skips particle creation

      # name = population name
      # q = species charge (multiple of elementary charge)
      # m = species mass (multiple of proton mass, including for electrons)
      # w = species macroparticle weight
      # electron = True if species is electrons, else protons
      # Np = number of macroparticles to generate on population generation
      # T = population temperature (dist. assumed Maxwellian)
      # v = population bulk velocity
      
      self.name = name
      self.q = q
      self.m = m
      self.w = w
      
      if Np > 0:
         vth = math.sqrt(const.k*T/self.m)
         self.uniform_injector(Np, vth, v)
      
      self.accumulators()

   def uniform_injector(self, Np, vth, v):
      # Inject Np particles between xmin and xmax, bulk velocity v, and thermal velocity vth

      self.r = np.zeros((Np*dims.Ncells_total,3))
      self.v = np.zeros((Np*dims.Ncells_total,3))

      for ii in range(dims.Ncells_total):
         zi,yi,xi = np.unravel_index(ii, dims.dim_scalar)
         
         cell_x_min = dims.x_min + xi*dims.dx
         cell_x_max = dims.x_max - (dims.x_size - 1 - xi)*dims.dx
         cell_y_min = dims.y_min + yi*dims.dy
         cell_y_max = dims.y_max - (dims.y_size - 1 - yi)*dims.dy
         cell_z_min = dims.z_min + zi*dims.dz
         cell_z_max = z_max - (dims.z_size - 1 - zi)*dims.dz
         
         self.r[ii*Np:(ii+1)*Np,0] = rng.uniform(cell_x_min,cell_x_max, Np)
         if oneV is True:
            self.r[ii*Np:(ii+1)*Np,1] = (cell_y_max + cell_y_min)/2
            self.r[ii*Np:(ii+1)*Np,2] = (cell_z_max + cell_z_min)/2
         else:
            self.r[ii*Np:(ii+1)*Np,1] = rng.uniform(cell_y_min,cell_y_max, Np)
            self.r[ii*Np:(ii+1)*Np,2] = rng.uniform(cell_z_min,cell_z_max, Np)
         self.v[ii*Np:(ii+1)*Np,0] = rng.normal(v[0], vth, Np)
         if oneV is False:
            self.v[ii*Np:(ii+1)*Np,1] = rng.normal(v[1], vth, Np)
            self.v[ii*Np:(ii+1)*Np,2] = rng.normal(v[2], vth, Np)

      self.Np = self.r.shape[0]
   
   def apply_boundaries(self):
      # Apply boundary conditions to particles
      # Periodic boundaries wrap locations
      # Non-periodic delete particles

      lims = ((dims.x_min,dims.x_max),(dims.y_min,dims.y_max),(dims.z_min,z_max))
      sizes = (dims.x_size,dims.y_size,dims.z_size)
      dgrid = (dims.dx,dims.dy,dims.dz)

      del_list = np.repeat(False, self.Np)
      
      for ii,(lim,Ng,dg,rep) in enumerate(zip(lims,sizes,dgrid,period)):
         cond = self.r[:,ii] < lim[0]
         if rep:
            while any(cond):
               self.r[cond,ii] += Ng*dg
               cond = self.r[:,ii] < lim[0]
         else:
            del_list |= cond

         cond = self.r[:,ii] >= lim[1]
         if rep:
            while any(cond):
               self.r[cond,ii] -= Ng*dg
               cond = self.r[:,ii] >= lim[1]
            if any(self.r[:,ii] < lim[0]):
               raise ValueError("Particle could not be moved back inside domain")
         else:
            del_list |= cond
      
      if any(del_list):
         self.removeParticle(del_list)
   
   def removeParticle(self, to_delete):
      # Delete particle from population
      to_keep = np.logical_not(to_delete)
      self.r = self.r[to_keep]
      self.v = self.v[to_keep]
      self.alpha = self.alpha[to_keep]
      self.Np = np.sum(to_keep)
   
   def accumulators(self):
      # Accumulate macroparticles into cell-centred charge and current densities
      x_locs = np.round((self.r[:,0] - dims.x_min)/dims.dx).astype(int)
      y_locs = np.round((self.r[:,1] - dims.y_min)/dims.dy).astype(int)
      z_locs = np.round((self.r[:,2] - dims.z_min)/dims.dz).astype(int)

      if x_periodic:
         x_locs = np.mod(x_locs, dims.x_size)
      if y_periodic:
         y_locs = np.mod(y_locs, dims.y_size)
      if z_periodic:
         z_locs = np.mod(z_locs, dims.z_size)
      
      self.rhoq = np.zeros(dims.dim_scalar, dtype = float)
      self.cellJi = np.zeros(dims.dim_vector, dtype = float)
      
      x0 = (x_locs - 1 + 0.5)*dims.dx + dims.x_min
      y0 = (y_locs - 1 + 0.5)*dims.dy + dims.y_min
      z0 = (z_locs - 1 + 0.5)*dims.dz + dims.z_min
      
      x_w1 = (self.r[:,0] - x0)/dims.dx
      y_w1 = (self.r[:,1] - y0)/dims.dy
      z_w1 = (self.r[:,2] - z0)/dims.dz

      if x_periodic:
         x_w1 = np.mod(x_w1, 1)
      if y_periodic:
         y_w1 = np.mod(y_w1, 1)
      if z_periodic:
         z_w1 = np.mod(z_w1, 1)
      
      x_w0 = 1 - x_w1
      y_w0 = 1 - y_w1
      z_w0 = 1 - z_w1
      
      x_ind0 = x_locs - 1
      if x_periodic:
         x_ind0 = np.mod(x_ind0, dims.x_size)
      x_ind1 = x_locs

      y_ind0 = y_locs - 1
      if y_periodic:
         y_ind0 = np.mod(y_ind0, dims.y_size)
      y_ind1 = y_locs

      z_ind0 = z_locs - 1
      if z_periodic:
         z_ind0 = np.mod(z_ind0, dims.z_size)
      z_ind1 = z_locs
      
      # CIC weight factors
      w_000 = x_w0 * y_w0 * z_w0
      w_001 = x_w0 * y_w0 * z_w1
      w_010 = x_w0 * y_w1 * z_w0
      w_011 = x_w0 * y_w1 * z_w1
      w_100 = x_w1 * y_w0 * z_w0
      w_101 = x_w1 * y_w0 * z_w1
      w_110 = x_w1 * y_w1 * z_w0
      w_111 = x_w1 * y_w1 * z_w1
      
      for x_ind_l,x_ind_r,y_ind_l,y_ind_r,z_ind_l,z_ind_r,w_lll,w_llr,w_lrl,w_lrr,w_rll,w_rlr,w_rrl,w_rrr,v in zip(x_ind0,x_ind1,y_ind0,y_ind1,z_ind0,z_ind1,w_000,w_001,w_010,w_011,w_100,w_101,w_110,w_111,self.v):
         # Charge density
         self.rhoq[z_ind_l,y_ind_l,x_ind_l] += w_lll
         self.rhoq[z_ind_r,y_ind_l,x_ind_l] += w_rll
         self.rhoq[z_ind_l,y_ind_r,x_ind_l] += w_lrl
         self.rhoq[z_ind_l,y_ind_l,x_ind_r] += w_llr
         self.rhoq[z_ind_r,y_ind_r,x_ind_l] += w_rrl
         self.rhoq[z_ind_r,y_ind_l,x_ind_r] += w_rlr
         self.rhoq[z_ind_l,y_ind_r,x_ind_r] += w_lrr
         self.rhoq[z_ind_r,y_ind_r,x_ind_r] += w_rrr
         
         # (cell-centred) current density
         self.cellJi[z_ind_l,y_ind_l,x_ind_l,:] += w_lll*v
         self.cellJi[z_ind_r,y_ind_l,x_ind_l,:] += w_rll*v
         self.cellJi[z_ind_l,y_ind_r,x_ind_l,:] += w_lrl*v
         self.cellJi[z_ind_l,y_ind_l,x_ind_r,:] += w_llr*v
         self.cellJi[z_ind_r,y_ind_r,x_ind_l,:] += w_rrl*v
         self.cellJi[z_ind_r,y_ind_l,x_ind_r,:] += w_rlr*v
         self.cellJi[z_ind_l,y_ind_r,x_ind_r,:] += w_lrr*v
         self.cellJi[z_ind_r,y_ind_r,x_ind_r,:] += w_rrr*v

      self.rhoq *= self.q*self.w/dims.vol
      self.cellJi *= self.q*self.w/dims.vol

   def compute_alpha(self, faceB):
      # Compute alpha matrix for all particles in population
      rB = face2r_njit(faceB, self.r, period, dims)

      beta = (self.q*dt)/(2*self.m)

      bx,by,bz = split_axis(rB*beta, axis = 1)
      
      if oneV is True:
         self.alpha = np.zeros((self.Np,3,3))
         
         self.alpha[:,0,0] = 1
      else:
         factor = 1/(1 + bx**2 + by**2 + bz**2)
         
         self.alpha = np.zeros((self.Np,3,3))
         self.alpha[:,0,0] = 1 + bx**2
         self.alpha[:,0,1] = 1 + bz + bx*by
         self.alpha[:,0,2] = 1 - by + bx*bz
         self.alpha[:,1,0] = 1 - bz + bx*by
         self.alpha[:,1,1] = 1 + by**2
         self.alpha[:,1,2] = 1 + bx + by*bz
         self.alpha[:,2,0] = 1 + by + bx*bz
         self.alpha[:,2,1] = 1 - bx + by*bz
         self.alpha[:,2,2] = 1 + bz**2

         self.alpha *= factor.reshape(self.Np,1,1)

   def moveParticles(self, dstep):
      # Pushes particle positions by time dstep
      self.r += self.v * dstep
      
      self.apply_boundaries()

   def Lorentz(self, midNodeE):
      # Accelerate particles via Lorentz force
      rE = node2r_njit(midNodeE, self.r, period, dims)
      
      beta = (self.q*dt)/(2*self.m)
      
      for ii,(r,v,alpha,E) in enumerate(zip(self.r,self.v,self.alpha,rE)):
         new_v = 2*alpha@(v + beta*E) - v
         
         self.v[ii] = new_v
      
   def nodeU(self):
      # Computes bulk velocity at nodes
      nodeU = np.zeros(dims.dim_vector)
      node_w = np.zeros(dims.dim_vector)
      
      for r,v in zip(self.r,self.v):
         x_locs = np.floor((r[0] - dims.x_min)/dims.dx).astype(int)
         y_locs = np.floor((r[1] - dims.y_min)/dims.dy).astype(int)
         z_locs = np.floor((r[2] - dims.z_min)/dims.dz).astype(int)
         
         x0 = x_locs*dims.dx + dims.x_min
         y0 = y_locs*dims.dy + dims.y_min
         z0 = z_locs*dims.dz + dims.z_min
         
         x_w1 = (r[0] - x0)/dims.dx
         y_w1 = (r[1] - y0)/dims.dy
         z_w1 = (r[2] - z0)/dims.dz
         
         x_w0 = 1 - x_w1
         y_w0 = 1 - y_w1
         z_w0 = 1 - z_w1
         
         x_ind0 = x_locs
         x_ind1 = x_locs + 1
         if x_periodic:
            x_ind1 = np.mod(x_ind1, dims.x_size)
         
         y_ind0 = y_locs
         y_ind1 = y_locs + 1
         if y_periodic:
            y_ind1 = np.mod(y_ind1, dims.y_size)
         
         z_ind0 = z_locs
         z_ind1 = z_locs + 1
         if z_periodic:
            z_ind1 = np.mod(z_ind1, dims.z_size)
         
         if oneV is True:
            nodeU[z_ind0,y_ind0,x_ind0,:] += x_w0*v
            nodeU[z_ind0,y_ind0,x_ind1,:] += x_w1*v
            
            node_w[z_ind0,y_ind0,x_ind0,:] += x_w0
            node_w[z_ind0,y_ind0,x_ind1,:] += x_w1
         else:
            w_lll = x_w0 * y_w0 * z_w0
            w_rll = x_w1 * y_w0 * z_w0
            w_lrl = x_w0 * y_w1 * z_w0
            w_llr = x_w0 * y_w0 * z_w1
            w_rrl = x_w1 * y_w1 * z_w0
            w_rlr = x_w1 * y_w0 * z_w1
            w_lrr = x_w0 * y_w1 * z_w1
            w_rrr = x_w1 * y_w1 * z_w1
            
            nodeU[z_ind0,y_ind0,x_ind0,:] += w_lll*v
            nodeU[z_ind1,y_ind0,x_ind0,:] += w_rll*v
            nodeU[z_ind0,y_ind1,x_ind0,:] += w_lrl*v
            nodeU[z_ind0,y_ind0,x_ind1,:] += w_llr*v
            nodeU[z_ind1,y_ind1,x_ind0,:] += w_rrl*v
            nodeU[z_ind1,y_ind0,x_ind1,:] += w_rlr*v
            nodeU[z_ind0,y_ind1,x_ind1,:] += w_lrr*v
            nodeU[z_ind1,y_ind1,x_ind1,:] += w_rrr*v
            
            node_w[z_ind0,y_ind0,x_ind0,:] += w_lll
            node_w[z_ind1,y_ind0,x_ind0,:] += w_rll
            node_w[z_ind0,y_ind1,x_ind0,:] += w_lrl
            node_w[z_ind0,y_ind0,x_ind1,:] += w_llr
            node_w[z_ind1,y_ind1,x_ind0,:] += w_rrl
            node_w[z_ind1,y_ind0,x_ind1,:] += w_rlr
            node_w[z_ind0,y_ind1,x_ind1,:] += w_lrr
            node_w[z_ind1,y_ind1,x_ind1,:] += w_rrr

      nodeU /= node_w
      
      return nodeU
   
   # def compute_rotated_current(self):
   #    # Computes current due to B-rotation
   #    # Current is stored at nodes to match E
   #    # This requires an interpolation after accumulating to cells
   #    cellJ = np.zeros(dims.dim_vector)
   #    
   #    for r,v,alpha in zip(self.r,self.v,self.alpha):
   #       x_locs = np.round((r[0] - dims.x_min)/dims.dx).astype(int)
   #       y_locs = np.round((r[1] - dims.y_min)/dims.dy).astype(int)
   #       z_locs = np.round((r[2] - dims.z_min)/dims.dz).astype(int)
   #       
   #       if x_periodic:
   #          x_locs = np.mod(x_locs, dims.x_size)
   #       if y_periodic:
   #          y_locs = np.mod(y_locs, dims.y_size)
   #       if z_periodic:
   #          z_locs = np.mod(z_locs, dims.z_size)
   #       
   #       x0 = (x_locs - 1 + 0.5)*dims.dx + dims.x_min
   #       y0 = (y_locs - 1 + 0.5)*dims.dy + dims.y_min
   #       z0 = (z_locs - 1 + 0.5)*dims.dz + dims.z_min
   #       
   #       x_w1 = (r[0] - x0)/dims.dx
   #       y_w1 = (r[1] - y0)/dims.dy
   #       z_w1 = (r[2] - z0)/dims.dz
   #       
   #       if x_periodic:
   #          x_w1 = np.mod(x_w1, 1)
   #       if y_periodic:
   #          y_w1 = np.mod(y_w1, 1)
   #       if z_periodic:
   #          z_w1 = np.mod(z_w1, 1)
   #       
   #       x_w0 = 1 - x_w1
   #       y_w0 = 1 - y_w1
   #       z_w0 = 1 - z_w1
   #       
   #       x_ind0 = x_locs - 1
   #       if x_periodic:
   #          x_ind0 = np.mod(x_ind0, dims.x_size)
   #       x_ind1 = x_locs
   #       
   #       y_ind0 = y_locs - 1
   #       if y_periodic:
   #          y_ind0 = np.mod(y_ind0, dims.y_size)
   #       y_ind1 = y_locs
   #       
   #       z_ind0 = z_locs - 1
   #       if z_periodic:
   #          z_ind0 = np.mod(z_ind0, dims.z_size)
   #       z_ind1 = z_locs
   #       
   #       if oneV is True:
   #          alpha_v = alpha@v
   #          
   #          # (cell-centred) current density
   #          cellJ[z_ind0,y_ind0,x_ind0,:] += x_w0*alpha_v
   #          cellJ[z_ind0,y_ind0,x_ind1,:] += x_w1*alpha_v
   #       else:
   #          w_lll = x_w0 * y_w0 * z_w0
   #          w_rll = x_w1 * y_w0 * z_w0
   #          w_lrl = x_w0 * y_w1 * z_w0
   #          w_llr = x_w0 * y_w0 * z_w1
   #          w_rrl = x_w1 * y_w1 * z_w0
   #          w_rlr = x_w1 * y_w0 * z_w1
   #          w_lrr = x_w0 * y_w1 * z_w1
   #          w_rrr = x_w1 * y_w1 * z_w1
   #          
   #          alpha_v = alpha@v
   #          
   #          # (cell-centred) current density
   #          cellJ[z_ind0,y_ind0,x_ind0,:] += w_lll*alpha_v
   #          cellJ[z_ind1,y_ind0,x_ind0,:] += w_rll*alpha_v
   #          cellJ[z_ind0,y_ind1,x_ind0,:] += w_lrl*alpha_v
   #          cellJ[z_ind0,y_ind0,x_ind1,:] += w_llr*alpha_v
   #          cellJ[z_ind1,y_ind1,x_ind0,:] += w_rrl*alpha_v
   #          cellJ[z_ind1,y_ind0,x_ind1,:] += w_rlr*alpha_v
   #          cellJ[z_ind0,y_ind1,x_ind1,:] += w_lrr*alpha_v
   #          cellJ[z_ind1,y_ind1,x_ind1,:] += w_rrr*alpha_v
   #          
   #    cellJ *= self.q*self.w/dims.vol
   #    
   #    nodeJ = cell2node(cellJ, period)
   #    
   #    return nodeJ

   def compute_rotated_current(self):
      # Computes current due to B-rotation
      # Current is stored at nodes to match E
      # This is accumulated directly to the nodes
      nodeJ = np.zeros(dims.dim_vector)
      
      for r,v,alpha in zip(self.r,self.v,self.alpha):
         x_locs = np.floor((r[0] - dims.x_min)/dims.dx).astype(int)
         y_locs = np.floor((r[1] - dims.y_min)/dims.dy).astype(int)
         z_locs = np.floor((r[2] - dims.z_min)/dims.dz).astype(int)
         
         x0 = x_locs*dims.dx + dims.x_min
         y0 = y_locs*dims.dy + dims.y_min
         z0 = z_locs*dims.dz + dims.z_min
         
         x_w1 = (r[0] - x0)/dims.dx
         y_w1 = (r[1] - y0)/dims.dy
         z_w1 = (r[2] - z0)/dims.dz
         
         x_w0 = 1 - x_w1
         y_w0 = 1 - y_w1
         z_w0 = 1 - z_w1
         
         x_ind0 = x_locs
         x_ind1 = x_locs + 1
         if x_periodic:
            x_ind1 = np.mod(x_ind1, dims.x_size)
         
         y_ind0 = y_locs
         y_ind1 = y_locs + 1
         if y_periodic:
            y_ind1 = np.mod(y_ind1, dims.y_size)
         
         z_ind0 = z_locs
         z_ind1 = z_locs + 1
         if z_periodic:
            z_ind1 = np.mod(z_ind1, dims.z_size)
         
         if oneV is True:
            alpha_v = alpha@v

            # (cell-centred) current density
            nodeJ[z_ind0,y_ind0,x_ind0,:] += x_w0*alpha_v
            nodeJ[z_ind0,y_ind0,x_ind1,:] += x_w1*alpha_v
         else:
            w_lll = x_w0 * y_w0 * z_w0
            w_rll = x_w1 * y_w0 * z_w0
            w_lrl = x_w0 * y_w1 * z_w0
            w_llr = x_w0 * y_w0 * z_w1
            w_rrl = x_w1 * y_w1 * z_w0
            w_rlr = x_w1 * y_w0 * z_w1
            w_lrr = x_w0 * y_w1 * z_w1
            w_rrr = x_w1 * y_w1 * z_w1
            
            alpha_v = alpha@v
            
            # (cell-centred) current density
            nodeJ[z_ind0,y_ind0,x_ind0,:] += w_lll*alpha_v
            nodeJ[z_ind1,y_ind0,x_ind0,:] += w_rll*alpha_v
            nodeJ[z_ind0,y_ind1,x_ind0,:] += w_lrl*alpha_v
            nodeJ[z_ind0,y_ind0,x_ind1,:] += w_llr*alpha_v
            nodeJ[z_ind1,y_ind1,x_ind0,:] += w_rrl*alpha_v
            nodeJ[z_ind1,y_ind0,x_ind1,:] += w_rlr*alpha_v
            nodeJ[z_ind0,y_ind1,x_ind1,:] += w_lrr*alpha_v
            nodeJ[z_ind1,y_ind1,x_ind1,:] += w_rrr*alpha_v

      nodeJ *= self.q*self.w/dims.vol
      
      return nodeJ
   
   def compute_mass_matrices(self):
      # Compute mass matrices
      if oneV:
         M = np.zeros((1,1,dims.Ncells_total,dims.Ncells_total))
      else:
         M = np.zeros((3,3,dims.Ncells_total,dims.Ncells_total))
      
      for r,alpha in zip(self.r,self.alpha):
         x_locs = np.floor((r[0] - dims.x_min)/dims.dx).astype(int)
         y_locs = np.floor((r[1] - dims.y_min)/dims.dy).astype(int)
         z_locs = np.floor((r[2] - dims.z_min)/dims.dz).astype(int)

         x0 = x_locs*dims.dx + dims.x_min
         y0 = y_locs*dims.dy + dims.y_min
         z0 = z_locs*dims.dz + dims.z_min
         
         x_w1 = (r[0] - x0)/dims.dx
         y_w1 = (r[1] - y0)/dims.dy
         z_w1 = (r[2] - z0)/dims.dz
         
         x_w0 = 1 - x_w1
         y_w0 = 1 - y_w1
         z_w0 = 1 - z_w1
         
         x_ind0 = x_locs
         x_ind1 = x_locs + 1
         if x_periodic:
            x_ind1 = np.mod(x_ind1, dims.x_size)
         
         y_ind0 = y_locs
         y_ind1 = y_locs + 1
         if y_periodic:
            y_ind1 = np.mod(y_ind1, dims.y_size)
         
         z_ind0 = z_locs
         z_ind1 = z_locs + 1
         if z_periodic:
            z_ind1 = np.mod(z_ind1, dims.z_size)
         
         x_ind = (x_ind0, x_ind1)
         y_ind = (y_ind0, y_ind1)
         z_ind = (z_ind0, z_ind1)
         
         x_w = (x_w0,x_w1)
         y_w = (y_w0,y_w1)
         z_w = (z_w0,z_w1)

         if oneV is True:
            for xi,x_wi in zip(x_ind,x_w):
               for xj,x_wj in zip(x_ind,x_w):
                  M[0,0,xi,xj] += x_wi*x_wj * alpha[0,0]
         else:
            for xi,x_wi in zip(x_ind,x_w):
               for xj,x_wj in zip(x_ind,x_w):
                  for yi,y_wi in zip(y_ind,y_w):
                     for yj,y_wj in zip(y_ind,y_w):
                        for zi,z_wi in zip(z_ind,z_w):
                           for zj,z_wj in zip(z_ind,z_w):
                              for ii in range(3):
                                 for jj in range(3):
                                    M[ii,jj,(zi*dims.y_size + yi)*dims.x_size + xi,
                                      (zj*dims.y_size + yj)*dims.x_size + xj] += x_wi*x_wj*y_wi*y_wj*z_wi*z_wj * alpha[ii,jj]
      
      beta = (self.q*dt)/(2*self.m)
      M *= beta * self.q*self.w/dims.vol
      
      return M

def apply_boundaries(data):
   # Applies Neumann BCs to given data array
   # All boundary types (cell,node,face) use the same method
   # Data in outermost layer copied from nearest 'real' neighbour
   # Periodic BCs has no boundary layer, so skip for periodic boundaries

   if not x_periodic:
      data[:,:,0,...] = data[:,:,1,...]
      data[:,:,-1,...] = data[:,:,-2,...]
   
   if not y_periodic:
      data[:,0,...] = data[:,1,...]
      data[:,-1,...] = data[:,-1,...]
   
   if not z_periodic:
      data[0,...] = data[1,...]
      data[-1,...] = data[-2,...]
   
   return data

def initialise_populations():
   # Initialise all particle populations
   print("")
   print("Initialising particle populations")
   pop_list = parse_multiarg_config(config, "simulation", "pop_list")

   global pops
   
   pops = {}
   for pop_name in pop_list:
      electron = config.getboolean(pop_name, "electron", fallback = "no")
      mass = config.getfloat(pop_name, "mass", fallback = None)
      charge = config.getfloat(pop_name, "charge", fallback = None)
      temp = config.getfloat(pop_name, "temperature", fallback = None)

      velocity = parse_multiarg_config(config, pop_name, "velocity", type_fun = float)

      density = config.getfloat(pop_name, "density", fallback = None)
      macros = config.getint(pop_name, "macroparticles_per_cell", fallback = None)
      
      weight = density*dims.vol/macros

      if electron:
         mass = const.m_p/args.mass_ratio
      else:
         mass *= const.m_p

      charge *= const.e
      pops[pop_name] = Pop(pop_name, charge, mass, weight, electron, macros, temp, velocity)

def initialise_fields():
   # Initialise all fields
   print("")
   print("Initialising all fields")

   global nodeJ,faceB,nodeE
   
   cellJp = np.sum([x.cellJi for x in pops.values()], axis = 0)

   faceJp = cell2face_njit(cellJp, period)
   
   faceB = np.zeros(dims.dim_vector)
   
   B_types = parse_multiarg_config(config, "magnetic_field", "type")
   Bx = config.getfloat("magnetic_field", "Bx")
   if oneV is True:
      By = Bz = 0
   else:
      By = config.getfloat("magnetic_field", "By")
      Bz = config.getfloat("magnetic_field", "Bz")
   for B_type in B_types:
      if B_type == "uniform":
         faceB[:,:,:,0] += Bx
         faceB[:,:,:,1] += By
         faceB[:,:,:,2] += Bz

   nodeB = face2node_njit(faceB, period)
   nodeUe = pops["e-"].nodeU()

   nodeE = -np.cross(nodeUe, nodeB)

def build_A(mass_matrices):
   # Construct sparse matrix A of Ax = b equation representing Maxwell's equations
   if oneV:
      A = np.zeros((2*dims.Ncells_total,2*dims.Ncells_total))
      A[0:dims.Ncells_total,dims.Ncells_total:2*dims.Ncells_total] -= const.mu_0*theta*mass_matrices[0,0]
      for ii in range(dims.Ncells_total):
         A[ii,ii + dims.Ncells_total] -= 1/(const.c**2*dt)
      
      for ii in range(dims.Ncells_total):
         A[ii + dims.Ncells_total,ii] += 1/dt
   else:
      A = np.zeros((6*dims.Ncells_total,6*dims.Ncells_total))
      
   return A

def build_b(faceB,nodeE,nodeJ_hat,mass_matrices):
   # Construct constant vector b of Ax = b equation representing Maxwell's equations
   Bx,By,Bz = split_axis(faceB, 3)
   Ex,Ey,Ez = split_axis(nodeE, 3)
   Jx,Jy,Jz = split_axis(nodeJ_hat, 3)

   Bx = Bx.flatten()
   By = By.flatten()
   Bz = Bz.flatten()
   Ex = Ex.flatten()
   Ey = Ey.flatten()
   Ez = Ez.flatten()
   Jx = Jx.flatten()
   Jy = Jy.flatten()
   Jz = Jz.flatten()
   
   if oneV:
      b = np.zeros(2*dims.Ncells_total)
      b[:dims.Ncells_total] += -1/(const.c**2*dt)*Ex+const.mu_0*Jx+const.mu_0*(1-theta)*mass_matrices[0,0]@Ex
      
      b[dims.Ncells_total:] += 1/dt*Bx
   else:
      b = np.zeros(6*dims.Ncells_total)
   
   return b
            
def test_interpolators():
   # Test interpolation methods
   Np = 10
   r = rng.uniform((dims.x_min,dims.y_min,dims.z_min), (dims.x_max,dims.y_max,z_max), (Np,3))
   
   cell_s = np.array(range(dims.Ncells_total), dtype = float).reshape(dims.dim_scalar)
   cell_v = np.array(range(dims.Ncells_total*3), dtype = float).reshape(dims.dim_vector)

   # cell_s = rng.uniform(-1, 1, dims.dim_scalar)
   # cell_v = rng.uniform(-1, 1, dims.dim_vector)
   
   cell_to_face = cell2face(cell_v, period)
   cell_to_node_s = cell2node(cell_s, period)
   cell_to_node_v = cell2node(cell_v, period)
   cell_to_r_s = cell2r(cell_s, r, period, dims)
   cell_to_r_v = cell2r(cell_v, r, period, dims)

   cell_to_face_njit = cell2face_njit(cell_v, period)
   cell_to_node_s_njit = cell2node_njit(cell_s, period)
   cell_to_node_v_njit = cell2node_njit(cell_v, period)
   cell_to_r_s_njit = cell2r_njit(cell_s, r, period, dims)
   cell_to_r_v_njit = cell2r_njit(cell_v, r, period, dims)

   node_s = np.array(range(dims.Ncells_total), dtype = float).reshape(dims.dim_scalar)
   node_v = np.array(range(dims.Ncells_total*3), dtype = float).reshape(dims.dim_vector)

   # node_s = rng.uniform(-1, 1, dims.dim_scalar)
   # node_v = rng.uniform(-1, 1, dims.dim_vector)
   
   node_to_face = node2face(node_v, period)
   node_to_cell_s = node2cell(node_s, period)
   node_to_cell_v = node2cell(node_v, period)
   node_to_r_s = node2r(node_s, r, period, dims)
   node_to_r_v = node2r(node_v, r, period, dims)

   node_to_face_njit = node2face_njit(node_v, period)
   node_to_cell_s_njit = node2cell_njit(node_s, period)
   node_to_cell_v_njit = node2cell_njit(node_v, period)
   node_to_r_s_njit = node2r_njit(node_s, r, period, dims)
   node_to_r_v_njit = node2r_njit(node_v, r, period, dims)

   face = np.array(range(dims.Ncells_total*3), dtype = float).reshape(dims.dim_vector)
   
   # face = rng.uniform(-1, 1, dims.dim_vector)

   face_to_cell = face2cell(face, period)
   face_to_node = face2node(face, period)
   face_to_r = face2r(face, r, period, dims)

   face_to_cell_njit = face2cell_njit(face, period)
   face_to_node_njit = face2node_njit(face, period)
   face_to_r_njit = face2r_njit(face, r, period, dims)

   compare("cell2face:        ", cell_to_face, cell_to_face_njit)
   compare("cell2node scalar: ", cell_to_node_s, cell_to_node_s_njit)
   compare("cell2node vector: ", cell_to_node_v, cell_to_node_v_njit)
   compare("cell2r scalar:    ", cell_to_r_s, cell_to_r_s_njit)
   compare("cell2r vector:    ", cell_to_r_v, cell_to_r_v_njit)

   compare("node2face:        ", node_to_face, node_to_face_njit)
   compare("node2cell scalar: ", node_to_cell_s, node_to_cell_s_njit)
   compare("node2cell vector: ", node_to_cell_v, node_to_cell_v_njit)
   compare("node2r scalar:    ", node_to_r_s, node_to_r_s_njit)
   compare("node2r vector:    ", node_to_r_v, node_to_r_v_njit)

   compare("face2cell:        ", face_to_cell, face_to_cell_njit)
   compare("face2node:        ", face_to_node, face_to_node_njit)
   compare("face2r:           ", face_to_r, face_to_r_njit)

def compare(name, first, second):
   test = all(first.flat == second.flat)

   print(name + str(test))
   
def cap_dt(pops):
   # Restrict dt if necessary, or expand
   global dt
   maxV = 0.0
   for pop in pops.values():
      maxV = max(maxV, np.linalg.norm(pop.v, axis = 1).max())
   
   if args.dt*maxV > dims.dx*dt_cap:
      print("")
      print("WARNING: maximum particle motion for given particle velocity exceeded, shrinking time step accordingly")

   if maxV > 0:
      dt = min(dt, dims.dx/maxV*dt_cap)
   else:
      dt = args.dt
   
args,config = readConfig(args.config, args)

dt = args.dt
theta = args.theta

dt_cap = config.getfloat("simulation", "Vdt_dx_cap", fallback = 0.5)

dimensions = config.getint("main", "dimensions", fallback = 3)

oneV = config.getboolean("main", "1v", fallback = False)

if oneV:
   dimensions = 1

x_periodic = config.getboolean("domain", "x_periodic", fallback = True)
y_periodic = config.getboolean("domain", "y_periodic", fallback = True)
z_periodic = config.getboolean("domain", "z_periodic", fallback = True)

rng = np.random.default_rng(args.seed)

x_min = config.getfloat("domain", "x_min", fallback = 0)
x_max = config.getfloat("domain", "x_max", fallback = 1)
x_size = config.getint("domain", "x_size", fallback = 1)
dx = (x_max - x_min)/x_size

if dimensions < 2:
   y_min = -dx/2
   y_max = dx/2
   y_size = 1
   y_periodic = True
   dy = dx
else:
   y_size = config.getint("domain", "y_size", fallback = 1)
   y_min = config.get("domain", "y_min", fallback = None)
   y_max = config.get("domain", "y_max", fallback = None)
   if y_min is None and y_max is None:
      y_min = -y_size*dx/2
      y_max = y_size*dx/2
      dy = dx
   elif y_min is None or y_max is None:
      raise TypeError("One of y_min or y_max not set; either both must be set or neither")
   else:
      y_min = float(y_min)
      y_max = float(y_max)
      dy = (y_max - y_min)/y_size

if dimensions < 3:
   z_min = -dx/2
   z_max = dx/2
   z_size = 1
   z_periodic = True
   dz = dx
else:
   z_size = config.getint("domain", "z_size", fallback = 1)
   z_min = config.get("domain", "z_min", fallback = None)
   z_max = config.get("domain", "z_max", fallback = None)
   if z_min is None and z_max is None:
      z_min = -z_size*dx/2
      z_max = z_size*dx/2
      dz = dx
   elif z_min is None or z_max is None:
      raise TypeError("One of z_min or z_max not set; either both must be set or neither")
   else:
      z_min = float(z_min)
      z_max = float(z_max)
      dz = (z_max - z_min)/z_size

# Extend axes if not periodic, to add ghost cells
if not x_periodic:
   x_size += 2
   x_min -= dx
   x_max += dx

if not y_periodic:
   y_size += 2
   y_min -= dy
   y_max += dy

if not z_periodic:
   z_size += 2
   z_min -= dz
   z_max += dz

period = (z_periodic, y_periodic, x_periodic, True)

lims = ((x_min,x_max),(y_min,y_max),(z_min,z_max))
spacing = (dx,dy,dz)
sizes = (x_size,y_size,z_size)

dims = Dims(lims, spacing, sizes)

test_interpolators()

timers = my_timers()
timers.tic("total")
timers.tic("init")

initialise_populations()

initialise_fields()

cap_dt(pops)

for pop in pops.values():
   print("")
   print("Uncentering particles of population " + pop.name)
   pop.moveParticles(-dt/2)

tmp_r = pops["e-"].r.copy()
tmp_v = pops["e-"].v.copy()
   
timers.toc("init")

for jj in range(args.steps):
   print("")
   print("Starting time step " + str(jj + 1))
   
   timers.tic("mover")
   
   print("")
   print("Moving particles")
   for pop in pops.values():
      pop.moveParticles(dt)

   timers.toc("mover")
   timers.tic("alpha")
      
   print("Calculating alpha \"rotation\" matrices")
   for pop in pops.values():
      pop.compute_alpha(faceB)

   timers.toc("alpha")
   timers.tic("current")
      
   print("Computing rotated current")
   nodeJ_hat = np.zeros(dims.dim_vector)
   for pop in pops.values():
      nodeJ_hat += pop.compute_rotated_current()

   timers.toc("current")
   timers.tic("mass matrices")
      
   print("Computing mass matrices")
   if oneV:
      mass_matrices = np.zeros((1,1,dims.Ncells_total,dims.Ncells_total))
   else:
      mass_matrices = np.zeros((3,3,dims.Ncells_total,dims.Ncells_total))
   
   for pop in pops.values():
      mass_matrices += pop.compute_mass_matrices()

   timers.toc("mass matrices")
   timers.tic("maxwell")

   print("Solving Maxwell's equations")
   timers.tic("build A")
   A = build_A(mass_matrices)
   timers.toc("build A")
   
   timers.tic("build b")
   b = build_b(faceB, nodeE, nodeJ_hat, mass_matrices)
   timers.toc("build b")
   
   timers.tic("gmres")
   info = 1
   rtol = args.rtol
   atol = args.atol
   
   if oneV:
      x0 = np.concatenate((faceB[:,:,:,0].flat,nodeE[:,:,:,0].flat))
   else:
      x0 = np.concatenate((faceB.flat,nodeE.flat))
   
   while info > 0 and rtol < 1e-2 and atol < 1e-2:
      xnext,info = gmres(A, b, rtol = rtol, atol = atol, x0 = x0)
      if info > 0:
         rtol *= 10
         atol *= 10
   
   if info < 0:
      sys.stderr.write("Illegal input in timestep " + str(jj) + "\n")
      sys.exit(1)
   elif info > 0:
      sys.stderr.write("Did not convergence in timestep " + str(jj) + "\n")
      sys.exit(1)
   timers.toc("gmres")
   
   newFaceB = faceB.copy()
   newNodeE = nodeE.copy()

   if oneV:
      newFaceB[:,:,:,0] = xnext[:dims.Ncells_total].reshape(dims.dim_scalar)
      newNodeE[:,:,:,0] = xnext[dims.Ncells_total:].reshape(dims.dim_scalar)
   
   midNodeE = theta*newNodeE + (1-theta)*nodeE

   nodeE = newNodeE
   faceB = newFaceB
   
   timers.toc("maxwell")

   timers.tic("lorentz")

   for pop in pops.values():
      pop.Lorentz(midNodeE)

   timers.toc("lorentz")

timers.toc("total")
   
print("Total time:           " + str(timers.timers["total"]))
print("Initialisation:       " + str(timers.timers["init"]))
print("Particle Mover:       " + str(timers.timers["mover"]))
print("Alpha Computation:    " + str(timers.timers["alpha"]))
print("Current Accumulation: " + str(timers.timers["current"]))
print("Mass Matrices:        " + str(timers.timers["mass matrices"]))
print("Maxwell:              " + str(timers.timers["maxwell"]))
print("   build A:           " + str(timers.timers["build A"]))
print("   build b:           " + str(timers.timers["build b"]))
print("   gmres:             " + str(timers.timers["gmres"]))
print("Lorentz Force update: " + str(timers.timers["lorentz"]))

tmp2 = pops["e-"].v - tmp_v

breakpoint()
