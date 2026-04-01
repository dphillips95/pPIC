import sys
import os
import math
import numpy as np
import scipy as sp
import scipy.constants as const
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import gmres
from timeit import default_timer as timer
import numba
from numba import int64,float64,boolean
from numba.typed import Dict as nb_dict
from numba import njit
from numba.experimental import jitclass as jitclass
import configparser
import argparse
import xarray as xr
import logging
import types

from indexers import split_axis,floatToStr,shift_indices,shift_indices_njit
from interpolators import face2cell,face2node,face2r,cell2node,cell2face,cell2r,node2face,node2cell,node2r,div_face2cell,div_node2cell,curl_face2node,curl_node2face,face2cell_njit,face2node_njit,face2r_njit,cell2node_njit,cell2face_njit,cell2r_njit,node2face_njit,node2cell_njit,node2r_njit,div_face2cell_njit,div_node2cell_njit,curl_face2node_njit,curl_node2face_njit,face2cell_njit_alt,face2node_njit_alt,cell2node_njit_alt,cell2face_njit_alt,node2face_njit_alt,node2cell_njit_alt,div_face2cell_njit_alt,div_node2cell_njit_alt,curl_face2node_njit_alt,curl_node2face_njit_alt,get_operator_curl_face2node,get_operator_curl_node2face,get_operator_coo_curl_face2node,get_operator_coo_curl_node2face
from populations import Pop,compute_alpha,moveParticles,Lorentz,compute_rotated_current,compute_mass_matrices,accumulators,calcNodeData,Pop_njit,compute_alpha_njit,moveParticles_njit,Lorentz_njit,compute_rotated_current_njit,compute_mass_matrices_njit,compute_mass_matrices_alt,compute_mass_matrices_coo,compute_mass_matrices_coo_alt,compute_mass_matrices_coo_njit
from fields import Fields,upwind_fields
from output import save_data

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
parser.add_argument("--phi", type = float,
                    help = "Phi parameter for particle position centering")
parser.add_argument("--rtol", type = float,
                    help = "gmres relative tolerance, norm(b - A @ x) <= rtol*norm(b)")
parser.add_argument("--atol", type = float,
                    help = "gmres absolute tolerance, norm(b - A @ x) <= atol")
parser.add_argument("--seed", type = int, help = "Rng seed")
parser.add_argument("-o", "--out-dir", type = str, default = "./",
                    help = "Output data directory path")
parser.add_argument("--test", type = str, nargs = '*',
                    help = "Test interpolators and quit")
args = parser.parse_args()

if args.config == "help":
   configHelp()

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
   print("   use_nonlinear_r_interpolation (bool): Use distance-weighted interpolation in cell2r and node2r instead of trilinear")
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

   sys.exit()

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

def log_newline(self, how_many_lines=1):
   # Output blank line by switching to blank handler then back again
   self.removeHandler(self.console_handler)
   self.addHandler(self.blank_handler)
   for i in range(how_many_lines):
      self.info('')
   
   self.removeHandler(self.blank_handler)
   self.addHandler(self.console_handler)

def create_logger():
   # Create a logger with two handles - normal and blank lines
   console_handler = logging.FileHandler('logfile.txt')
   console_handler.setLevel(logging.INFO)
   console_handler.setFormatter(logging.Formatter(fmt="%(asctime)-15s %(levelname)-8s %(message)s"))
   
   blank_handler = logging.FileHandler('logfile.txt')
   blank_handler.setLevel(logging.DEBUG)
   blank_handler.setFormatter(logging.Formatter(fmt=''))
   
   logger = logging.getLogger('logging_test')
   logger.setLevel(logging.DEBUG)
   logger.addHandler(console_handler)
   
   logger.console_handler = console_handler
   logger.blank_handler = blank_handler
   logger.newline = types.MethodType(log_newline, logger)
   
   return logger

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

   args.phi = config.get("simulation", "phi", vars = args_dict)
   if args.phi is None:
      args.phi = 0.5
   else:
      args.phi = float(args.phi)
   
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
      self.timers = dict()

   def tic(self, _timer_):
      self.timers[_timer_] -= timer()
   
   def toc(self, _timer_):
      self.timers[_timer_] += timer()
   
   def start(self, _timer_):
      self.timers[_timer_] = 0.0

   def reset(self, _timer_ = None):
      if _timer_ == None:
         for key in self.timers.keys():
            self.timers[key] = 0.0
      else:
         self.timers[_timer_] = 0.0

Dims_spec = [
   ("x_min", float64),
   ("x_max", float64),
   ("y_min", float64),
   ("y_max", float64),
   ("z_min", float64),
   ("z_max", float64),
   ("dx", float64),
   ("dy", float64),
   ("dz", float64),
   ("dV", float64),
   ("x_size", int64),
   ("y_size", int64),
   ("z_size", int64),
   ("Ncells_total", int64),
   ("dim_scalar", int64[:]),
   ("dim_vector", int64[:]),
   ("x_range", int64[:]),
   ("y_range", int64[:]),
   ("z_range", int64[:]),
   ("x_range_r2l", int64[:]),
   ("y_range_r2l", int64[:]),
   ("z_range_r2l", int64[:]),
   ("x_range_l2r", int64[:]),
   ("y_range_l2r", int64[:]),
   ("z_range_l2r", int64[:]),
   ("x_locs", float64[:]),
   ("y_locs", float64[:]),
   ("z_locs", float64[:]),
   ("period", boolean[:]),
   ("linear", boolean),
   ("oneV", boolean),
   ("dt", float64),
   ("time", float64),
   ("timestep", int64),
   ("theta", float64),
   ("phi", float64)
]

@jitclass(Dims_spec)
class Dims:
   def __init__(self, lims, dt, theta, phi, sizes, period, linear, oneV, time = 0.0, timestep = 0):
      # Pseudo-dict containg dimension params
      (self.x_min,self.x_max),(self.y_min,self.y_max),(self.z_min,self.z_max) = lims
      self.x_size,self.y_size,self.z_size = sizes
      self.dx = (self.x_max - self.x_min)/self.x_size
      self.dy = (self.y_max - self.y_min)/self.y_size
      self.dz = (self.z_max - self.z_min)/self.z_size
      self.dV = self.dx*self.dy*self.dz
      self.Ncells_total = self.x_size*self.y_size*self.z_size
      
      self.dim_scalar = np.array([self.z_size,self.y_size,self.x_size], dtype = int64)
      self.dim_vector = np.array([self.z_size,self.y_size,self.x_size,3], dtype = int64)

      self.dt = dt
      self.theta = theta
      self.phi = phi
      
      self.x_range = np.array(list(range(self.x_size)), dtype = int64)
      self.y_range = np.array(list(range(self.y_size)), dtype = int64)
      self.z_range = np.array(list(range(self.z_size)), dtype = int64)
      
      self.x_locs = np.linspace(self.x_min + self.dx/2, self.x_max - self.dx/2, self.x_size)
      self.y_locs = np.linspace(self.y_min + self.dy/2, self.y_max - self.dy/2, self.y_size)
      self.z_locs = np.linspace(self.z_min + self.dz/2, self.z_max - self.dz/2, self.z_size)
      
      # self.period = np.array(list(period), dtype = boolean)
      self.period = np.empty(4, dtype = boolean)
      for ii in range(4):
         self.period[ii] = period[ii]
      
      self.x_range_r2l = shift_indices_njit(self.x_range, 1, self.period[2])
      self.y_range_r2l = shift_indices_njit(self.y_range, 1, self.period[1])
      self.z_range_r2l = shift_indices_njit(self.z_range, 1, self.period[0])
      
      self.x_range_l2r = shift_indices_njit(self.x_range, -1, self.period[2])
      self.y_range_l2r = shift_indices_njit(self.y_range, -1, self.period[1])
      self.z_range_l2r = shift_indices_njit(self.z_range, -1, self.period[0])
      
      self.linear = linear
      self.oneV = oneV
      self.time = time
      self.timestep = timestep
      
   def copy(self):
      # Copy dims via reconstruction - returns inputs to Dims() as tuple
      lims = ((self.x_min,self.x_max),
              (self.y_min,self.y_max),
              (self.z_min,self.z_max))
      dt = self.dt
      theta = self.theta
      phi = self.phi
      sizes = (self.x_size,self.y_size,self.z_size)
      period = self.period.copy()
      linear = self.linear
      oneV = self.oneV
      time = self.time
      timestep = self.timestep

      return lims,dt,theta,phi,sizes,period,linear,oneV,time,timestep

   def step_time(self):
      # Increment time and timestep
      self.time += self.dt
      self.timestep += 1

def initialise_populations():
   # Initialise all particle populations
   print("")
   print("Initialising particle populations")
   pop_list = parse_multiarg_config(config, "simulation", "pop_list")

   global pops,pops_njit
   
   pops = {}
   # pops_njit = {}
   if pop_list is not None:
      for pop_name in pop_list:
         electron = config.getboolean(pop_name, "electron", fallback = False)
         mass = config.getfloat(pop_name, "mass", fallback = 0.0)
         charge = config.getfloat(pop_name, "charge", fallback = 0.0)
         temp = config.getfloat(pop_name, "temperature", fallback = 0.0)

         velocity = parse_multiarg_config(config, pop_name, "velocity", type_fun = float)

         density = config.getfloat(pop_name, "density", fallback = 0.0)
         macros = config.getint(pop_name, "macroparticles_per_cell", fallback = 1)
         
         weight = density*dims.dV/macros

         if electron:
            mass = const.m_p/args.mass_ratio
         else:
            mass *= const.m_p

         charge *= const.e

         static = config.getboolean(pop_name, "static", fallback = False)
         
         pops[pop_name] = Pop(pop_name, charge, mass, weight, electron, macros, rng, dims, temp, velocity, static)
      # pops_njit[pop_name] = Pop_njit(charge, mass, weight, electron, macros, rng, dims, temp, velocity)

def build_A(mass_matrices):
   # Construct sparse matrix A of Ax = b equation representing Maxwell's equations
   if dims.oneV:
      A = np.zeros((2*dims.Ncells_total,2*dims.Ncells_total))
      # B-component of Faraday's Law
      A[:dims.Ncells_total,:dims.Ncells_total] = np.identity(dims.Ncells_total)
      # E-component of Faraday's Law not present due to 1V

      # B-component of Ampère's Law not present due to 1V

      # E-component of Ampère's Law
      A[dims.Ncells_total:,dims.Ncells_total:] = np.identity(dims.Ncells_total)
      A[dims.Ncells_total:,dims.Ncells_total:] += dims.dt*dims.theta*mass_matrices[0,0]/const.epsilon_0
   else:
      A = np.zeros((6*dims.Ncells_total,6*dims.Ncells_total))
      # B-component of Faraday's Law
      A[:3*dims.Ncells_total,:3*dims.Ncells_total] = np.identity(3*dims.Ncells_total)
      # E-component of Faraday's Law
      A[:3*dims.Ncells_total,3*dims.Ncells_total:] = get_operator_curl_node2face(dims)*dims.dt*dims.theta

      # B-component of Ampère's Law
      A[3*dims.Ncells_total:,:3*dims.Ncells_total] = -get_operator_curl_face2node(dims)*dims.dt*dims.theta*const.c**2

      # E-component of Ampère's Law
      A[3*dims.Ncells_total:,3*dims.Ncells_total:] = np.identity(3*dims.Ncells_total)

      mass = mass_matrices.transpose((2,1,3,0)).reshape(3*dims.Ncells_total,3*dims.Ncells_total)
      
      A[3*dims.Ncells_total:,3*dims.Ncells_total:] += dims.dt*dims.theta*mass/const.epsilon_0
   return A

def build_b(faceB, nodeE, nodeJ_hat, mass_matrices):
   # Construct constant vector b of Ax = b equation representing Maxwell's equations      
   if dims.oneV:
      Ex = nodeE[:,:,:,0].flatten()

      Jx = nodeJ_hat[:,:,:,0].flatten()
      Jx += (1 - dims.theta)*mass_matrices[0,0]@Ex
      
      b = np.zeros(2*dims.Ncells_total)
      # Constant term of Faraday's Law
      b[:dims.Ncells_total] += faceB[:,:,:,0].flat

      # Constant term of Ampère's Law
      b[dims.Ncells_total:] += Ex-dims.dt*Jx/const.epsilon_0
   else:
      # Jx = np.zeros(Jx.shape)
      # Jy = np.zeros(Jy.shape)
      # Jz = np.zeros(Jz.shape)

      # Jx += (1 - dims.theta)*mass_matrices[0,0]@E_split[0].flat
      # Jx += (1 - dims.theta)*mass_matrices[0,1]@E_split[1].flat
      # Jx += (1 - dims.theta)*mass_matrices[0,2]@E_split[2].flat
      
      # Jy += (1 - dims.theta)*mass_matrices[1,0]@E_split[0].flat
      # Jy += (1 - dims.theta)*mass_matrices[1,1]@E_split[1].flat
      # Jy += (1 - dims.theta)*mass_matrices[1,2]@E_split[2].flat

      # Jz += (1 - dims.theta)*mass_matrices[2,0]@E_split[0].flat
      # Jz += (1 - dims.theta)*mass_matrices[2,1]@E_split[1].flat
      # Jz += (1 - dims.theta)*mass_matrices[2,2]@E_split[2].flat
      
      E_split = split_axis(nodeE, axis = 3)
      
      mass = (1 - dims.theta)*mass_matrices.transpose((1,0,2,3))

      J_mod = np.empty((3,3,dims.Ncells_total), dtype = np.float64)
      
      for ii in range(3):
         J_mod[ii] = np.matvec(mass[ii], E_split[ii].flat)

      J_mod = np.sum(J_mod, axis = 0)
      
      b = np.zeros(6*dims.Ncells_total)

      J_star = nodeJ_hat.copy().flatten()
      J_star += J_mod.transpose().flatten()
      
      # Constant term of Faraday's Law
      b[:3*dims.Ncells_total] += faceB.flat
      b[:3*dims.Ncells_total] -= (curl_node2face(nodeE, dims)*dims.dt*(1-dims.theta)).flat
      
      # Constant term of Ampère's Law
      b[3*dims.Ncells_total:] += nodeE.flat
      b[3*dims.Ncells_total:] -= dims.dt*J_star/const.epsilon_0
      b[3*dims.Ncells_total:] += (curl_face2node(faceB, dims)*const.c**2*dims.dt*(1-dims.theta)).flat
   
   return b

def build_A_coo(mass_matrices):
   # Construct sparse coo matrix A of Ax = b equation representing Maxwell's equations
   if dims.oneV:
      # B-component of Faraday's Law
      FaradayB = sp.sparse.identity(dims.Ncells_total, dtype = np.float64, format = 'coo')

      # E-component of Faraday's Law
      FaradayE = sp.sparse.coo_matrix((dims.Ncells_total,dims.Ncells_total), dtype = np.float64)

      # B-component of Ampère's Law
      AmpereB = sp.sparse.coo_matrix((dims.Ncells_total,dims.Ncells_total), dtype = np.float64)

      # E-component of Ampère's Law
      AmpereE = sp.sparse.identity(dims.Ncells_total, dtype = np.float64, format = 'coo')
      
      AmpereE += dims.dt*dims.theta*mass_matrices/const.epsilon_0
   else:
      block_shape = (3*dims.Ncells_total,3*dims.Ncells_total)
      
      # B-component of Faraday's Law
      FaradayB = sp.sparse.identity(3*dims.Ncells_total, dtype = np.float64, format = 'coo')
      
      # E-component of Faraday's Law
      data,rows,cols = get_operator_coo_curl_node2face(dims)
      FaradayE = coo_matrix((data,(rows,cols)), shape = block_shape)*dims.dt*dims.theta
      FaradayE.sum_duplicates()
      
      # B-component of Ampère's Law
      data,rows,cols = get_operator_coo_curl_face2node(dims)
      AmpereB = -coo_matrix((data,(rows,cols)), shape = block_shape)*dims.dt*dims.theta*const.c**2
      AmpereB.sum_duplicates()

      # E-component of Ampère's Law
      AmpereE = sp.sparse.identity(3*dims.Ncells_total, dtype = np.float64, format = 'coo')

      AmpereE += dims.dt*dims.theta*mass_matrices/const.epsilon_0

      AmpereE.sum_duplicates()

   Faraday = sp.sparse.hstack((FaradayB,FaradayE))
   Ampere = sp.sparse.hstack((AmpereB,AmpereE))
   A = sp.sparse.vstack((Faraday,Ampere))
      
   return A

def build_b_coo(faceB, nodeE, nodeJ_hat, mass_matrices):
   # Construct constant vector b of Ax = b equation representing Maxwell's equations
   if dims.oneV:
      b = np.zeros(2*dims.Ncells_total)
      
      Ex = nodeE[:,:,:,0].flat
      
      J_star = (1 - dims.theta)*mass_matrices@Ex
      
      J_star += nodeJ_hat[:,:,:,0].flat
      
      # Constant term of Faraday's Law
      b[:dims.Ncells_total] += faceB[:,:,:,0].flat

      # Constant term of Ampère's Law
      b[dims.Ncells_total:] += Ex-dims.dt*J_star/const.epsilon_0
   else:
      b = np.zeros(6*dims.Ncells_total)
      
      J_star = (1 - dims.theta)*mass_matrices@nodeE.flat
      
      J_star += nodeJ_hat.flat
      
      # Constant term of Faraday's Law
      b[:3*dims.Ncells_total] += faceB.flat
      b[:3*dims.Ncells_total] -= (curl_node2face(nodeE, dims)*dims.dt*(1-dims.theta)).flat
      
      # Constant term of Ampère's Law
      b[3*dims.Ncells_total:] += nodeE.flat
      b[3*dims.Ncells_total:] -= dims.dt*J_star/const.epsilon_0
      b[3*dims.Ncells_total:] += (curl_face2node(faceB, dims)*const.c**2*dims.dt*(1-dims.theta)).flat
   
   return b

def test_interpolators(function = None):
   # Test interpolation methods
   print("Testing interpolators")
   Np = 100
   
   r = rng.uniform((dims.x_min,dims.y_min,dims.z_min), (dims.x_max,dims.y_max,dims.z_max), (Np,3))
   
   dims_n = Dims(*dims.copy())
   dims_n.linear = False
   
   if not function:
      function = [
         "cell2face",
         "cell2node",
         "cell2r",
         "node2face",
         "node2cell",
         "node2r",
         "face2cell",
         "face2node",
         "face2r",
         "div_face2cell",
         "div_node2cell",
         "curl_face2node",
         "curl_node2face"
         ]
   elif not isinstance(function, list):
      try:
         function = list(function)
      except TypeError:
         function = [function]

   for func in function:
      if func.startswith("cell"):
         # cell_s = np.array(range(dims.Ncells_total), dtype = np.float64).reshape(dims.dim_scalar)
         # cell_v = np.array(range(dims.Ncells_total*3), dtype = np.float64).reshape(dims.dim_vector)

         cell_s = rng.uniform(-1, 1, dims.dim_scalar)
         cell_v = rng.uniform(-1, 1, dims.dim_vector)

         if func == "cell2face":
            c2f = cell2face(cell_v, dims)
            c2f_njit = cell2face_njit(cell_v, dims)
            c2f_njit_alt = cell2face_njit_alt(cell_v, dims)
            compare("cell2face:        ", (c2f,c2f_njit,c2f_njit_alt))
         elif func == "cell2node":
            c2n_s = cell2node(cell_s, dims)
            c2n_v = cell2node(cell_v, dims)
            c2n_s_njit = cell2node_njit(cell_s, dims)
            c2n_v_njit = cell2node_njit(cell_v, dims)
            c2n_s_njit_alt = cell2node_njit_alt(cell_s, dims)
            c2n_v_njit_alt = cell2node_njit_alt(cell_v, dims)
            compare("cell2node scalar: ", (c2n_s,c2n_s_njit,c2n_s_njit_alt))
            compare("cell2node vector: ", (c2n_v,c2n_v_njit,c2n_v_njit_alt))
         elif func == "cell2r":
            c2r_s_l = cell2r(cell_s, r, dims)
            c2r_v_l = cell2r(cell_v, r, dims)
            c2r_s_n = cell2r(cell_s, r, dims_n)
            c2r_v_n = cell2r(cell_v, r, dims_n)
            c2r_s_l_njit = cell2r_njit(cell_s, r, dims)
            c2r_v_l_njit = cell2r_njit(cell_v, r, dims)
            c2r_s_n_njit = cell2r_njit(cell_s, r, dims_n)
            c2r_v_n_njit = cell2r_njit(cell_v, r, dims_n)
            compare("cell2r scalar:    ", (c2r_s_l,c2r_s_l_njit))
            compare("cell2r vector:    ", (c2r_v_l,c2r_v_l_njit))
            compare("cell2r scalar:    ", (c2r_s_n,c2r_s_n_njit))
            compare("cell2r vector:    ", (c2r_v_n,c2r_v_n_njit))

      elif func.startswith("node"):
         # node_s = np.array(range(dims.Ncells_total), dtype = float).reshape(dims.dim_scalar)
         # node_v = np.array(range(dims.Ncells_total*3), dtype = float).reshape(dims.dim_vector)
         
         node_s = rng.uniform(-1, 1, dims.dim_scalar)
         node_v = rng.uniform(-1, 1, dims.dim_vector)

         if func == "node2face":
            n2f = node2face(node_v, dims)
            n2f_njit = node2face_njit(node_v, dims)
            n2f_njit_alt = node2face_njit_alt(node_v, dims)
            compare("node2face:        ", (n2f,n2f_njit,n2f_njit_alt))
         elif func == "node2cell":
            n2c_s = node2cell(node_s, dims)
            n2c_v = node2cell(node_v, dims)
            n2c_s_njit = node2cell_njit(node_s, dims)
            n2c_v_njit = node2cell_njit(node_v, dims)
            n2c_s_njit_alt = node2cell_njit_alt(node_s, dims)
            n2c_v_njit_alt = node2cell_njit_alt(node_v, dims)
            compare("node2cell scalar: ", (n2c_s,n2c_s_njit,n2c_s_njit_alt))
            compare("node2cell vector: ", (n2c_v,n2c_v_njit,n2c_v_njit_alt))
         elif func == "node2r":
            n2r_s_l = node2r(node_s, r, dims)
            n2r_v_l = node2r(node_v, r, dims)
            n2r_s_n = node2r(node_s, r, dims_n)
            n2r_v_n = node2r(node_v, r, dims_n)
            n2r_s_l_njit = node2r_njit(node_s, r, dims)
            n2r_v_l_njit = node2r_njit(node_v, r, dims)
            n2r_s_n_njit = node2r_njit(node_s, r, dims_n)
            n2r_v_n_njit = node2r_njit(node_v, r, dims_n)
            compare("node2r scalar:    ", (n2r_s_l,n2r_s_l_njit))
            compare("node2r vector:    ", (n2r_v_l,n2r_v_l_njit))
            compare("node2r scalar:    ", (n2r_s_n,n2r_s_n_njit))
            compare("node2r vector:    ", (n2r_v_n,n2r_v_n_njit))
   

      elif func.startswith("face"):
         # face = np.array(range(dims.Ncells_total*3), dtype = float).reshape(dims.dim_vector)
         
         face = rng.uniform(-1, 1, dims.dim_vector)

         if func == "face2cell":
            f2c = face2cell(face, dims)
            f2c_njit = face2cell_njit(face, dims)
            f2c_njit_alt = face2cell_njit_alt(face, dims)
            compare("face2cell:        ", (f2c,f2c_njit,f2c_njit_alt))
         elif func == "face2node":   
            f2n = face2node(face, dims)
            f2n_njit = face2node_njit(face, dims)
            f2n_njit_alt = face2node_njit_alt(face, dims)
            compare("face2node:        ", (f2n,f2n_njit,f2n_njit_alt))
         elif func == "face2r":
            f2r = face2r(face, r, dims)
            f2r_njit = face2r_njit(face, r, dims)
            compare("face2r:           ", (f2r,f2r_njit))

      elif func.startswith("div"):
         # face = np.array(range(dims.Ncells_total*3), dtype = float).reshape(dims.dim_vector)
         face = rng.uniform(-1, 1, dims.dim_vector)

         # node = np.array(range(dims.Ncells_total*3), dtype = float).reshape(dims.dim_vector)
         node = rng.uniform(-1, 1, dims.dim_vector)

         if func == "div_face2cell":
            div_f2c = div_face2cell(face, dims)
            div_f2c_njit = div_face2cell_njit(face, dims)
            div_f2c_njit_alt = div_face2cell_njit_alt(face, dims)
            compare("div_face2cell:    ",
                    (div_f2c,div_f2c_njit,div_f2c_njit_alt))
         elif func == "div_node2cell":
            div_n2c = div_node2cell(node, dims)
            div_n2c_njit = div_node2cell_njit(node, dims)
            div_n2c_njit_alt = div_node2cell_njit_alt(node, dims)
            compare("div_node2cell:    ",
                    (div_n2c,div_n2c_njit,div_n2c_njit_alt))
            
      elif func.startswith("curl"):
         # face = np.array(range(dims.Ncells_total*3), dtype = float).reshape(dims.dim_vector)
         face = rng.uniform(-1, 1, dims.dim_vector)

         # node = np.array(range(dims.Ncells_total*3), dtype = float).reshape(dims.dim_vector)
         node = rng.uniform(-1, 1, dims.dim_vector)

         if func == "curl_face2node":
            curl_f2n = curl_face2node(face, dims)
            curl_f2n_njit = curl_face2node_njit(face, dims)
            curl_f2n_njit_alt = curl_face2node_njit_alt(face, dims)
            operator = get_operator_curl_face2node(dims)
            curl_f2n_op = (operator@face.flat).reshape(dims.dim_vector)
            data,rows,cols = get_operator_coo_curl_face2node(dims)
            operator_coo = coo_matrix((data,(rows,cols)), shape = (3*dims.Ncells_total,3*dims.Ncells_total))
            curl_f2n_op_coo = (operator_coo@face.flat).reshape(dims.dim_vector)
            compare("curl_face2node:   ",
                    (curl_f2n,curl_f2n_njit,curl_f2n_njit_alt,curl_f2n_op,curl_f2n_op_coo))
         elif func == "curl_node2face":
            curl_n2f = curl_node2face(node, dims)
            curl_n2f_njit = curl_node2face_njit(node, dims)
            curl_n2f_njit_alt = curl_node2face_njit_alt(node, dims)
            operator = get_operator_curl_node2face(dims)
            curl_n2f_op = (operator@node.flat).reshape(dims.dim_vector)
            data,rows,cols = get_operator_coo_curl_node2face(dims)
            operator_coo = coo_matrix((data,(rows,cols)), shape = (3*dims.Ncells_total,3*dims.Ncells_total))
            curl_n2f_op_coo = (operator_coo@node.flat).reshape(dims.dim_vector)
            compare("curl_node2face:   ",
                    (curl_n2f,curl_n2f_njit,curl_n2f_njit_alt,curl_n2f_op,curl_n2f_op_coo))
   
   print("")
   
   interp_timers = my_timers()
   interp_timers_njit = my_timers()
   interp_timers_njit_alt = my_timers()
   tests_dict = {
      "cell2face":[("c2f",     (cell2face,cell2face_njit,cell2face_njit_alt), "vector", None)],
      "cell2node":[
         ("c2n_s",   (cell2node,cell2node_njit,cell2node_njit_alt), "scalar", None),
         ("c2n_v",   (cell2node,cell2node_njit,cell2node_njit_alt), "vector", None)
         ],
      "cell2r":[
         ("c2r_s_l", (cell2r,cell2r_njit),    "scalar", True),
         ("c2r_v_l", (cell2r,cell2r_njit),    "vector", True),
         ("c2r_s_n", (cell2r,cell2r_njit),    "scalar", False),
         ("c2r_v_n", (cell2r,cell2r_njit),    "vector", False),
      ],
      "node2face":[("n2f",     (node2face,node2face_njit,node2face_njit_alt), "vector", None)],
      "node2cell":[
         ("n2c_s",   (node2cell,node2cell_njit,node2cell_njit_alt), "scalar", None),
         ("n2c_v",   (node2cell,node2cell_njit,node2cell_njit_alt), "vector", None),
      ],
      "node2r":[
         ("n2r_s_l", (node2r,node2r_njit),    "scalar", True),
         ("n2r_v_l", (node2r,node2r_njit),    "vector", True),
         ("n2r_s_n", (node2r,node2r_njit),    "scalar", False),
         ("n2r_v_n", (node2r,node2r_njit),    "vector", False),
      ],
      "face2cell":[("f2c",     (face2cell,face2cell_njit,face2cell_njit_alt), "vector", None)],
      "face2node":[("f2n",     (face2node,face2node_njit,face2node_njit_alt), "vector", None)],
      "face2r":[
         ("f2r_l",   (face2r,face2r_njit),    "vector", True),
         ("f2r_n",   (face2r,face2r_njit),    "vector", False),
      ],
      "div_face2cell":[("d_f2c",   (div_face2cell,div_face2cell_njit,div_face2cell_njit_alt), "vector", None)],
      "div_node2cell":[("d_n2c",   (div_node2cell,div_node2cell_njit,div_node2cell_njit_alt), "vector", None)],
      "curl_face2node":[("c_f2n",   (curl_face2node,curl_face2node_njit,curl_face2node_njit_alt), "vector", None)],
      "curl_node2face":[("c_n2f",   (curl_node2face,curl_node2face_njit,curl_node2face_njit_alt), "vector", None)]
   }

   tests = []
   for func in function:
      tests.extend(tests_dict[func])
   
   for timer_name,*_ in tests:
      interp_timers.start(timer_name)
      interp_timers_njit.start(timer_name)
      interp_timers_njit_alt.start(timer_name)

   loops = 100
   
   Np_test = 100
   
   x_min_test = -1
   x_max_test = 1
   y_min_test = -1
   y_max_test = 1
   z_min_test = -1
   z_max_test = 1

   x_size_test = 10
   y_size_test = 10
   z_size_test = 10

   x_periodic_test = True
   y_periodic_test = True
   z_periodic_test = True
   
   lims = ((x_min_test,x_max_test),(y_min_test,y_max_test),(z_min_test,z_max_test))
   sizes = (x_size_test,y_size_test,z_size_test)
   period = (z_periodic_test, y_periodic_test, x_periodic_test, True)

   oneV = True
   
   test_dims = Dims(lims, dt, dims.theta, dims.phi, sizes, period, True, oneV)
   test_dims_n = Dims(*test_dims.copy())
   test_dims_n.linear = False
   
   for name,functions,array_type,linear in tests:
      for ii in range(loops):
         if array_type == "scalar":
            array_dim = test_dims.dim_scalar
         elif array_type == "vector":
            array_dim = test_dims.dim_vector
         array = rng.uniform(-1, 1, array_dim)

         no_alt = True
         if functions[0].__name__.endswith("r"):
            function,function_njit = functions
            r = rng.uniform((test_dims.x_min,test_dims.y_min,test_dims.z_min), (test_dims.x_max,test_dims.y_max,test_dims.z_max), (Np_test,3))
            if function.__name__.startswith("face"):
               interp_timers.tic(name)
               res = function(array, r, test_dims)
               interp_timers.toc(name)

               interp_timers_njit.tic(name)
               res_njit = function_njit(array, r, test_dims)
               interp_timers_njit.toc(name)
            else:
               if linear:
                  interp_timers.tic(name)
                  res = function(array, r, test_dims)
                  interp_timers.toc(name)
                  
                  interp_timers_njit.tic(name)
                  res_njit = function_njit(array, r, test_dims)
                  interp_timers_njit.toc(name)
               else:
                  interp_timers.tic(name)
                  res = function(array, r, test_dims_n)
                  interp_timers.toc(name)
                  
                  interp_timers_njit.tic(name)
                  res_njit = function_njit(array, r, test_dims_n)
                  interp_timers_njit.toc(name)
         else:
            no_alt = False
            function,function_njit,function_njit_alt = functions
            interp_timers.tic(name)
            res = function(array, test_dims)
            interp_timers.toc(name)
            
            interp_timers_njit.tic(name)
            res_njit = function_njit(array, test_dims)
            interp_timers_njit.toc(name)

            interp_timers_njit_alt.tic(name)
            res_njit_alt = function_njit_alt(array, test_dims)
            interp_timers_njit_alt.toc(name)
      
      print_string = function.__name__
      print_string_njit = function_njit.__name__
      if no_alt:
         print_string_njit_alt = None
      else:
         print_string_njit_alt = function_njit_alt.__name__
      if "face" not in function.__name__:
         print_string += " " + array_type
         print_string_njit += " " + array_type
         if not no_alt:
            print_string_njit_alt += " " + array_type
      if no_alt:
         if linear:
            print_string += " linear"
            print_string_njit += " linear"
         else:
            print_string += " non-linear"
            print_string_njit += " non-linear"

      print_ratio = print_string + " njit ratio:"
      if not no_alt:
         print_ratio_alt = print_string + " njit_alt ratio:"
            
      print_string += ": "
      print_string_njit += ": "
      if not no_alt:
         print_string_njit_alt += ": "
            
      print_ratio = print_ratio.ljust(38)
      print_string = print_string.ljust(38)
      print_string_njit = print_string_njit.ljust(38)
      if not no_alt:
         print_ratio_alt = print_ratio_alt.ljust(38)
         print_string_njit_alt = print_string_njit_alt.ljust(38)

      print_string += str(interp_timers.timers[name])
      print_string_njit += str(interp_timers_njit.timers[name])
      print_ratio += str(interp_timers.timers[name] / interp_timers_njit.timers[name])
      if not no_alt:
         print_string_njit_alt += str(interp_timers_njit_alt.timers[name])
         print_ratio_alt += str(interp_timers.timers[name] / interp_timers_njit_alt.timers[name])
      
      print(print_string)
      print(print_string_njit)
      if not no_alt:
         print(print_string_njit_alt)
      print(print_ratio)
      if not no_alt:
         print(print_ratio_alt)
      print("")

   exit()
      
def compare(name, items, tol = 1e-15):
   # Compare two or more ndarrays in items
   test = True
   for dat in items:
      test &= all(dat.flat == 0)
      
   if test:
      print(name + str([Test]*(len(items) - 1)))
   else:
      base_max = np.max(np.abs(items[0]))
      test = np.empty(len(items) - 1, dtype = np.bool)
      for ii,dat in enumerate(items[1:]):
         local_max = max(base_max, np.max(np.abs(dat)))
         test[ii] = all((np.abs(items[0] - dat)/local_max).flat < tol)

      print(name + str(test))
   
def cap_dt(pops):
   # Restrict dt if necessary, or expand
   maxV = 0.0
   for pop in pops.values():
      maxV = max(maxV, np.linalg.norm(pop.v, axis = 1).max())

   n_e = 0
   
   for pop in pops.values():
      electron = config.getboolean(pop.name, "electron", fallback = False)
      if electron:
         n_e += config.getfloat(pop.name, "density", fallback = 0.0)
      
   plasma_freq = math.sqrt(n_e*const.e**2/(const.m_e*const.epsilon_0))

   maxB = np.max(np.linalg.norm(fields.faceB, axis = -1))
   
   electron_gyro = const.e*maxB/const.m_e
   
   dt_pf = 1/plasma_freq if plasma_freq > 0 else np.inf
   dt_eg = 1/electron_gyro if electron_gyro > 0 else np.inf
   dt_part = dims.dx/maxV*dt_cap
   dt_yee = 0.9999 * min([dims.dx,dims.dy,dims.dz])/const.c
   
   if args.dt > dt_part:
      logger.info("Shrinking dt to prevent particles from crossing cell")
      print("Shrinking dt to prevent particles from crossing cell")
      dims.dt = dt_part
   else:
      dims.dt = args.dt
   
   logger.info("Time step relative factors")
   logger.info("dt*omega_pe = " + str(dims.dt*plasma_freq))
   logger.info("dt*omega_ce = " + str(dims.dt*electron_gyro))
   logger.info("dt/dt_yee = " + str(dims.dt/dt_yee))
   
args,config = readConfig(args.config, args)

dt = args.dt
theta = args.theta
phi = args.phi

dt_cap = config.getfloat("simulation", "Vdt_dx_cap", fallback = 0.25)

dimensions = config.getint("main", "dimensions", fallback = 3)

oneV = config.getboolean("main", "1v", fallback = False)

if oneV:
   dimensions = 1

x_periodic = config.getboolean("domain", "x_periodic", fallback = True)
y_periodic = config.getboolean("domain", "y_periodic", fallback = True)
z_periodic = config.getboolean("domain", "z_periodic", fallback = True)

rng = np.random.default_rng(args.seed)

x_min = config.getfloat("domain", "x_min", fallback = 0.0)
x_max = config.getfloat("domain", "x_max", fallback = 1.0)
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

lims = ((x_min,x_max),(y_min,y_max),(z_min,z_max))
sizes = (x_size,y_size,z_size)
period = (z_periodic, y_periodic, x_periodic, True)

dims = Dims(lims, dt, theta, phi, sizes, period, not config.getboolean("main", "use_nonlinear_r_interpolation", fallback = False), oneV)

save_steps = config.getint("simulation", "save_steps", fallback = 1)

if args.test is not None:
   test_interpolators(args.test)

timers = my_timers()

timer_list = [
   'init',
   'alpha',
   'current',
   'mass matrices',
   'maxwell',
   'build A',
   'build b',
   'gmres',
   'locs',
   'field solver',
   'lorentz',
   'extra',
   'output',
   'total'
]

if __name__ == '__main__':
   logger = create_logger()
   
   for timer_name in timer_list:
      timers.start(timer_name)

   timers.tic("total")
   timers.tic("init")

   initialise_populations()

   B_types = parse_multiarg_config(config, "magnetic_field", "type")

   B_init = (
      config.getfloat("magnetic_field", "Bx"),
      config.getfloat("magnetic_field", "By"),
      config.getfloat("magnetic_field", "Bz")
   )

   E_types = parse_multiarg_config(config, "electric_field", "type")

   E_init = (
      config.getfloat("electric_field", "Ex"),
      config.getfloat("electric_field", "Ey"),
      config.getfloat("electric_field", "Ez")
   )

   fields = Fields(pops, B_types, B_init, E_types, E_init, rng, dims)

   cap_dt(pops)

   logger.newline()
   for name,pop in pops.items():
      if pop.static is False:
         logger.info("Uncentering particles of population " + name)
         moveParticles(pop, -dims.dt*(1-dims.phi), dims)

   timers.toc("init")
   
   if os.path.isfile(args.out_dir + "/fields.h5") or os.path.isfile(args.out_dir + "/pops.h5") or os.path.isfile(args.out_dir + "/logs.h5" or os.path.isfile(args.out_dir + "/logfile.txt")):
      print("Output file(s) already exist, aborting")
      sys.exit()

   save_data(args.out_dir, fields, pops, dims)
   
   tolerance_error = False
   
   print("")
   print("Starting iterations")
   for jj in range(1, args.steps + 1):
      logger.newline()
      logger.info("Starting time step " + str(jj))
      if jj%25 == 0:
         print("Starting time step " + str(jj))

      timers.tic("locs")

      logger.info("Moving particles")
      for pop in pops.values():
         if pop.static is False:
            moveParticles(pop, dims.dt, dims)

      timers.toc("locs")
      timers.tic("alpha")
      timers.tic("field solver")

      logger.info("Calculating alpha \"rotation\" matrices")
      for pop in pops.values():
         compute_alpha(pop, fields.faceB, dims)

      timers.toc("alpha")
      timers.tic("current")

      logger.info("Computing rotated current")
      nodeJ_hat = np.zeros(dims.dim_vector)
      for pop in pops.values():
         if pop.static is False:
            nodeJ_hat += compute_rotated_current(pop, dims)

      timers.toc("current")
      timers.tic("mass matrices")

      logger.info("Computing mass matrices")
      # if dims.oneV:
      #    mass_matrices = np.zeros((1,1,dims.Ncells_total,dims.Ncells_total))
      # else:
      #    mass_matrices = np.zeros((3,3,dims.Ncells_total,dims.Ncells_total))
      # 
      # for pop in pops.values():
      #    if pop.static is False:
      #       mass_matrices += compute_mass_matrices_njit(
      #          pop.r, pop.alpha, pop.m, pop.q, pop.w, dims)
      
      data_M = []
      rows_M = []
      cols_M = []
      
      for pop in pops.values():
         if pop.static is False:
            dat,row,col = compute_mass_matrices_coo(pop, dims)
            data_M.append(dat)
            rows_M.append(row)
            cols_M.append(col)
      
      data_M = np.concatenate(data_M)
      rows_M = np.concatenate(rows_M)
      cols_M = np.concatenate(cols_M)

      if oneV:
         block_shape = (dims.Ncells_total,dims.Ncells_total)
      else:
         block_shape = (3*dims.Ncells_total,3*dims.Ncells_total)
      
      mass_matrices_coo = coo_matrix((data_M,(rows_M,cols_M)),
                                     shape = block_shape)
      mass_matrices_coo.sum_duplicates()
      
      timers.toc("mass matrices")
      timers.tic("maxwell")

      logger.info("Solving Maxwell's equations")

      timers.tic("build b")
      
      # b = build_b(fields.faceB, fields.nodeE, nodeJ_hat, mass_matrices)
      b_coo = build_b_coo(fields.faceB, fields.nodeE, nodeJ_hat, mass_matrices_coo)
      
      timers.toc("build b")
      timers.tic("build A")
      
      # A = build_A(mass_matrices)
      A_coo = build_A_coo(mass_matrices_coo)
      
      timers.toc("build A")
      timers.tic("gmres")
      
      info = 1
      rtol = args.rtol
      atol = args.atol

      if dims.oneV:
         x0 = np.concatenate((fields.faceB[:,:,:,0].flat,fields.nodeE[:,:,:,0].flat))
      else:
         x0 = np.concatenate((fields.faceB.flat,fields.nodeE.flat))
      
      while info > 0 and rtol < 1e-2 and atol < 1e-2:
         
         xnext,info = gmres(A_coo, b_coo, rtol = rtol, atol = atol, x0 = x0)
         if info > 0:
            logger.info("GMRES tolerance failure on step " + str(jj) + ", reducing tolerance")
            if not tolerance_error:
               tolerance_error = True
               print("GMRES tolerance failure on step " + str(jj) + ", reducing tolerance")
            rtol *= 10
            atol *= 10
      
      if rtol != args.rtol:
         logger.info("Tolerance reduced to " + str(rtol) + ", " + str(math.floor(math.log(rtol/args.rtol, 10))) + " orders of magnitude")

      if info < 0:
         sys.stderr.write("Illegal input in timestep " + str(jj) + "\n")
         sys.exit(1)
      elif info > 0:
         sys.stderr.write("Did not convergence in timestep " + str(jj) + "\n")
         sys.exit(1)
      
      timers.toc("gmres")

      # cellJ = np.zeros(dims.dim_vector, dtype = np.float64)
      # for pop in pops.values():
      #    cellJ += pop.cellJi

      # nodeJ = cell2node(cellJ, dims)

      # implicitE = fields.nodeE - nodeJ*dims.dt/const.epsilon_0

      # old_nodeE = fields.nodeE.copy()

      fields,midNodeE = upwind_fields(fields, xnext, dims)

      timers.toc("maxwell")
      timers.toc("field solver")
      timers.tic("lorentz")

      for pop in pops.values():
         if pop.static is False:
            Lorentz(pop, midNodeE, dims)
            # pop.v = Lorentz_njit(pop.r, pop.v, pop.alpha, pop.q, pop.m, midNodeE, dims)

      timers.toc("lorentz")

      # Increment time
      dims.step_time()

      if (jj+1)%save_steps == 0:
         timers.tic("extra")
         
         for pop in pops.values():
            accumulators(pop, dims)
            calcNodeData(pop, dims)
         fields.update_fields(pops, dims)
         
         timers.toc("extra")
         timers.tic("output")
         
         save_data(args.out_dir, fields, pops, dims)
         
         timers.toc("output")

   logger.newline()
   logger.info("Saving final state")
         
   timers.tic("extra")
   
   for pop in pops.values():
      accumulators(pop, dims)
      calcNodeData(pop, dims)
   fields.update_fields(pops, dims)
   
   timers.toc("extra")
   timers.tic("output")
   
   save_data(args.out_dir, fields, pops, dims)
   
   timers.toc("output")
         
   timers.toc("total")

   logger.newline()
   logger.info("Simulation complete")
   
   print("")
   print("Time per timestep and cell [ms/(cell timestep)]:")
   print("Total time:           " + floatToStr(1000*timers.timers["total"]/(dims.timestep*dims.Ncells_total), decimals = 3))
   print("Initialisation:       " + floatToStr(1000*timers.timers["init"]/(dims.timestep*dims.Ncells_total), decimals = 3))
   print("Particle locs update: " + floatToStr(1000*timers.timers["locs"]/(dims.timestep*dims.Ncells_total), decimals = 3))
   print("Alpha Computation:    " + floatToStr(1000*timers.timers["alpha"]/(dims.timestep*dims.Ncells_total), decimals = 3))
   print("Current Accumulation: " + floatToStr(1000*timers.timers["current"]/(dims.timestep*dims.Ncells_total), decimals = 3))
   print("Mass Matrices:        " + floatToStr(1000*timers.timers["mass matrices"]/(dims.timestep*dims.Ncells_total), decimals = 3))
   print("Maxwell:              " + floatToStr(1000*timers.timers["maxwell"]/(dims.timestep*dims.Ncells_total), decimals = 3))
   print("   build A:           " + floatToStr(1000*timers.timers["build A"]/(dims.timestep*dims.Ncells_total), decimals = 3))
   print("   build b:           " + floatToStr(1000*timers.timers["build b"]/(dims.timestep*dims.Ncells_total), decimals = 3))
   print("   gmres:             " + floatToStr(1000*timers.timers["gmres"]/(dims.timestep*dims.Ncells_total), decimals = 3))
   print("Lorentz Force update: " + floatToStr(1000*timers.timers["lorentz"]/(dims.timestep*dims.Ncells_total), decimals = 3))
   print("Unused fields update: " + floatToStr(1000*timers.timers["extra"]/(dims.timestep*dims.Ncells_total), decimals = 3))
   print("Field Solver:         " + floatToStr(1000*timers.timers["field solver"]/(dims.timestep*dims.Ncells_total), decimals = 3))
   print("Particle Mover:       " + floatToStr(1000*(timers.timers["locs"] + timers.timers["lorentz"])/(dims.timestep*dims.Ncells_total), decimals = 3))
   print("Data saving:          " + floatToStr(1000*timers.timers["output"]/(dims.timestep*dims.Ncells_total), decimals = 3))
