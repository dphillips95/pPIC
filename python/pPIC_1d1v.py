import sys
import os
import math
import numpy as np
import scipy as sp
import scipy.constants as const
from scipy.sparse.linalg import gmres
from timeit import default_timer as timer
import numba
from numba import int64,float64,boolean
from numba import types
from numba.typed import Dict as nb_dict
from numba import njit
from numba.experimental import jitclass as jitclass
import configparser
import argparse
import xarray as xr

from tools import split_axis,floatToStr
from interpolators import face2cell,face2node,face2r,cell2node,cell2face,cell2r,node2face,node2cell,node2r,face2cell_njit,face2node_njit,face2r_njit,cell2node_njit,cell2face_njit,cell2r_njit,node2face_njit,node2cell_njit,node2r_njit
from populations import Pop,compute_alpha,moveParticles,Lorentz,compute_rotated_current,compute_mass_matrices,accumulators,calcNodeData,Pop_njit,compute_alpha_njit,moveParticles_njit,Lorentz_njit,compute_rotated_current_njit,compute_mass_matrices_njit
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
parser.add_argument("--rtol", type = float,
                    help = "gmres relative tolerance, norm(b - A @ x) <= rtol*norm(b)")
parser.add_argument("--atol", type = float,
                    help = "gmres absolute tolerance, norm(b - A @ x) <= atol")
parser.add_argument("--seed", type = int, help = "Rng seed")
parser.add_argument("-o", "--out-dir", type = str, default = "./",
                    help = "Output data directory path")
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
   ("x_locs", float64[:]),
   ("y_locs", float64[:]),
   ("z_locs", float64[:]),
   ("period", boolean[:]),
   ("linear", boolean),
   ("oneV", boolean),
   ("dt", float64),
   ("time", float64),
   ("timestep", int64),
   ("theta", float64)
]

@jitclass(Dims_spec)
class Dims:
   def __init__(self, lims, dt, theta, sizes, period, linear, oneV, time = 0.0, timestep = 0):
      # Pseudo-dict containg dimension params
      (self.x_min,self.x_max),(self.y_min,self.y_max),(self.z_min,self.z_max) = lims
      self.x_size,self.y_size,self.z_size = sizes
      self.dx = (self.x_max - self.x_min)/self.x_size
      self.dy = (self.y_max - self.y_min)/self.y_size
      self.dz = (self.z_max - self.z_min)/self.z_size
      self.dV = self.dx*self.dy*self.dz
      self.dt = dt
      self.theta = theta
      self.Ncells_total = self.x_size*self.y_size*self.z_size
      self.dim_scalar = np.array([self.z_size,self.y_size,self.x_size], dtype = int64)
      self.dim_vector = np.array([self.z_size,self.y_size,self.x_size,3], dtype = int64)
      self.x_range = np.array(list(range(self.x_size)), dtype = int64)
      self.y_range = np.array(list(range(self.y_size)), dtype = int64)
      self.z_range = np.array(list(range(self.z_size)), dtype = int64)
      self.x_locs = np.linspace(self.x_min + self.dx/2, self.x_max - self.dx/2, self.x_size)
      self.y_locs = np.linspace(self.y_min + self.dy/2, self.y_max - self.dy/2, self.y_size)
      self.z_locs = np.linspace(self.z_min + self.dz/2, self.z_max - self.dz/2, self.z_size)
      self.period = np.array(list(period), dtype = boolean)
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
      sizes = (self.x_size,self.y_size,self.z_size)
      period = self.period.copy()
      linear = self.linear
      oneV = self.oneV

      return lims,dt,theta,sizes,period,linear,oneV,time,timestep

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
   for pop_name in pop_list:
      electron = config.getboolean(pop_name, "electron", fallback = "no")
      mass = config.getfloat(pop_name, "mass", fallback = None)
      charge = config.getfloat(pop_name, "charge", fallback = None)
      temp = config.getfloat(pop_name, "temperature", fallback = None)

      velocity = parse_multiarg_config(config, pop_name, "velocity", type_fun = float)

      density = config.getfloat(pop_name, "density", fallback = None)
      macros = config.getint(pop_name, "macroparticles_per_cell", fallback = None)
      
      weight = density*dims.dV/macros

      if electron:
         mass = const.m_p/args.mass_ratio
      else:
         mass *= const.m_p
      
      charge *= const.e
      pops[pop_name] = Pop(pop_name, charge, mass, weight, electron, macros, rng, dims, temp, velocity)
      # pops_njit[pop_name] = Pop_njit(charge, mass, weight, electron, macros, rng, dims, temp, velocity)

def build_A(mass_matrices):
   # Construct sparse matrix A of Ax = b equation representing Maxwell's equations
   if dims.oneV:
      A = np.zeros((2*dims.Ncells_total,2*dims.Ncells_total))
      A[:dims.Ncells_total,dims.Ncells_total:] += dims.dt*const.mu_0*dims.theta*mass_matrices[0,0]

      A[:dims.Ncells_total,dims.Ncells_total:] += np.identity(dims.Ncells_total)/const.c**2

      # for ii in range(dims.Ncells_total):
      #    A[ii,ii + dims.Ncells_total] += 1/const.c**2

      A[dims.Ncells_total:,:dims.Ncells_total] = np.identity(dims.Ncells_total)
      
      # for ii in range(dims.Ncells_total):
      #    A[ii + dims.Ncells_total,ii] += 1
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
   
   if dims.oneV:
      b = np.zeros(2*dims.Ncells_total)
      b[:dims.Ncells_total] += 1/const.c**2*Ex-dims.dt*const.mu_0*Jx-dims.dt*const.mu_0*(1-dims.theta)*mass_matrices[0,0]@Ex
      
      b[dims.Ncells_total:] += Bx
   else:
      b = np.zeros(6*dims.Ncells_total)
   
   return b
            
def test_interpolators():
   # Test interpolation methods
   Np = 10
   r = rng.uniform((dims.x_min,dims.y_min,dims.z_min), (dims.x_max,dims.y_max,dims.z_max), (Np,3))

   dims_n = Dims(*dims.copy())
   dims_n.linear = False
   
   # cell_s = np.array(range(dims.Ncells_total), dtype = np.float64).reshape(dims.dim_scalar)
   # cell_v = np.array(range(dims.Ncells_total*3), dtype = np.float64).reshape(dims.dim_vector)
   
   cell_s = rng.uniform(-1, 1, dims.dim_scalar)
   cell_v = rng.uniform(-1, 1, dims.dim_vector)

   cell_to_face = cell2face(cell_v, dims)
   cell_to_node_s = cell2node(cell_s, dims)
   cell_to_node_v = cell2node(cell_v, dims)
   cell_to_r_s_l = cell2r(cell_s, r, dims)
   cell_to_r_v_l = cell2r(cell_v, r, dims)
   cell_to_r_s_n = cell2r(cell_s, r, dims_n)
   cell_to_r_v_n = cell2r(cell_v, r, dims_n)
   
   cell_to_face_njit = cell2face_njit(cell_v, dims)
   cell_to_node_s_njit = cell2node_njit(cell_s, dims)
   cell_to_node_v_njit = cell2node_njit(cell_v, dims)
   cell_to_r_s_l_njit = cell2r_njit(cell_s, r, dims)
   cell_to_r_v_l_njit = cell2r_njit(cell_v, r, dims)
   cell_to_r_s_n_njit = cell2r_njit(cell_s, r, dims_n)
   cell_to_r_v_n_njit = cell2r_njit(cell_v, r, dims_n)

   # node_s = np.array(range(dims.Ncells_total), dtype = float).reshape(dims.dim_scalar)
   # node_v = np.array(range(dims.Ncells_total*3), dtype = float).reshape(dims.dim_vector)

   node_s = rng.uniform(-1, 1, dims.dim_scalar)
   node_v = rng.uniform(-1, 1, dims.dim_vector)
   
   node_to_face = node2face(node_v, dims)
   node_to_cell_s = node2cell(node_s, dims)
   node_to_cell_v = node2cell(node_v, dims)
   node_to_r_s_l = node2r(node_s, r, dims)
   node_to_r_v_l = node2r(node_v, r, dims)
   node_to_r_s_n = node2r(node_s, r, dims_n)
   node_to_r_v_n = node2r(node_v, r, dims_n)
   
   node_to_face_njit = node2face_njit(node_v, dims)
   node_to_cell_s_njit = node2cell_njit(node_s, dims)
   node_to_cell_v_njit = node2cell_njit(node_v, dims)
   node_to_r_s_l_njit = node2r_njit(node_s, r, dims)
   node_to_r_v_l_njit = node2r_njit(node_v, r, dims)
   node_to_r_s_n_njit = node2r_njit(node_s, r, dims_n)
   node_to_r_v_n_njit = node2r_njit(node_v, r, dims_n)
   
   # face = np.array(range(dims.Ncells_total*3), dtype = float).reshape(dims.dim_vector)
   
   face = rng.uniform(-1, 1, dims.dim_vector)

   face_to_cell = face2cell(face, dims)
   face_to_node = face2node(face, dims)
   face_to_r = face2r(face, r, dims)

   face_to_cell_njit = face2cell_njit(face, dims)
   face_to_node_njit = face2node_njit(face, dims)
   face_to_r_njit = face2r_njit(face, r, dims)

   compare("cell2face:        ", cell_to_face, cell_to_face_njit)
   compare("cell2node scalar: ", cell_to_node_s, cell_to_node_s_njit)
   compare("cell2node vector: ", cell_to_node_v, cell_to_node_v_njit)
   compare("cell2r scalar:    ", cell_to_r_s_l, cell_to_r_s_l_njit)
   compare("cell2r vector:    ", cell_to_r_v_l, cell_to_r_v_l_njit)
   compare("cell2r scalar:    ", cell_to_r_s_n, cell_to_r_s_n_njit)
   compare("cell2r vector:    ", cell_to_r_v_n, cell_to_r_v_n_njit)

   compare("node2face:        ", node_to_face, node_to_face_njit)
   compare("node2cell scalar: ", node_to_cell_s, node_to_cell_s_njit)
   compare("node2cell vector: ", node_to_cell_v, node_to_cell_v_njit)
   compare("node2r scalar:    ", node_to_r_s_l, node_to_r_s_l_njit)
   compare("node2r vector:    ", node_to_r_v_l, node_to_r_v_l_njit)
   compare("node2r scalar:    ", node_to_r_s_n, node_to_r_s_n_njit)
   compare("node2r vector:    ", node_to_r_v_n, node_to_r_v_n_njit)

   compare("face2cell:        ", face_to_cell, face_to_cell_njit)
   compare("face2node:        ", face_to_node, face_to_node_njit)
   compare("face2r:           ", face_to_r, face_to_r_njit)

   print("")
   
   interp_timers = my_timers()
   interp_timers_njit = my_timers()
   tests = [
      ("c2f",     cell2face, cell2face_njit, "vector", None),
      ("c2n_s",   cell2node, cell2node_njit, "scalar", None),
      ("c2n_v",   cell2node, cell2node_njit, "vector", None),
      ("c2r_s_l", cell2r,    cell2r_njit,    "scalar", True),
      ("c2r_v_l", cell2r,    cell2r_njit,    "vector", True),
      ("c2r_s_n", cell2r,    cell2r_njit,    "scalar", False),
      ("c2r_v_n", cell2r,    cell2r_njit,    "vector", False),
      ("n2f",     node2face, node2face_njit, "vector", None),
      ("n2c_s",   node2cell, node2cell_njit, "scalar", None),
      ("n2c_v",   node2cell, node2cell_njit, "vector", None),
      ("n2r_s_l", node2r,    node2r_njit,    "scalar", True),
      ("n2r_v_l", node2r,    node2r_njit,    "vector", True),
      ("n2r_s_n", node2r,    node2r_njit,    "scalar", False),
      ("n2r_v_n", node2r,    node2r_njit,    "vector", False),
      ("f2c",     face2cell, face2cell_njit, "vector", None),
      ("f2n",     face2node, face2node_njit, "vector", None),
      ("f2r_l",   face2r,    face2r_njit,    "vector", True),
      ("f2r_n",   face2r,    face2r_njit,    "vector", False)
   ]

   for timer_name,*_ in tests:
      interp_timers.start(timer_name)
      interp_timers_njit.start(timer_name)

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
   
   test_dims = Dims(lims, dt, dims.theta, sizes, period, True, oneV)
   test_dims_n = Dims(*test_dims.copy())
   test_dims_n.linear = False
   
   dim_s = (10,10,10)
   dim_v = (10,10,10,3)
   
   for name,function,function_njit,array_type,linear in tests:
      for ii in range(loops):
         if array_type == "scalar":
            array_dim = test_dims.dim_scalar
         elif array_type == "vector":
            array_dim = test_dims.dim_vector
         array = rng.uniform(-1, 1, array_dim)

         if function.__name__.endswith("r"):
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
            interp_timers.tic(name)
            res = function(array, test_dims)
            interp_timers.toc(name)
            
            interp_timers_njit.tic(name)
            res_njit = function_njit(array, test_dims)
            interp_timers_njit.toc(name)
      
      print_string = function.__name__
      print_string_njit = function_njit.__name__
      if not function.__name__.endswith("face"):
         print_string += " " + array_type
         print_string_njit += " " + array_type
      if function.__name__.endswith("r"):
         if linear:
            print_string += " linear"
            print_string_njit += " linear"
         else:
            print_string += " non-linear"
            print_string_njit += " non-linear"

      print_ratio = print_string + " ratio:"
            
      print_string += ": "
      print_string_njit += ": "
      
      print_ratio = print_ratio.ljust(32)
      print_string = print_string.ljust(32)
      print_string_njit = print_string_njit.ljust(32)

      print_string += str(interp_timers.timers[name])
      print_string_njit += str(interp_timers_njit.timers[name])
      print_ratio += str(interp_timers.timers[name] / interp_timers_njit.timers[name])
      
      print(print_string)
      print(print_string_njit)
      print(print_ratio)
      print("")

   exit()
      
def compare(name, first, second):
   test = all(first.flat == second.flat)

   print(name + str(test))
   
def cap_dt(pops):
   # Restrict dt if necessary, or expand
   maxV = 0.0
   for pop in pops.values():
      maxV = max(maxV, np.linalg.norm(pop.v, axis = 1).max())
   
   if args.dt*maxV > dims.dx*dt_cap:
      print("")
      print("WARNING: maximum particle motion for given particle velocity exceeded, shrinking time step accordingly")

   if maxV > 0:
      dims.dt = min(args.dt, dims.dx/maxV*dt_cap)
   else:
      dims.dt = args.dt
   
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

lims = ((x_min,x_max),(y_min,y_max),(z_min,z_max))
sizes = (x_size,y_size,z_size)
period = (z_periodic, y_periodic, x_periodic, True)

dims = Dims(lims, dt, theta, sizes, period, config.getboolean("main", "use_nonlinear_r_interpolation", fallback = False), oneV)

# test_interpolators()

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

fields = Fields(pops, B_types, B_init, dims)

cap_dt(pops)

for name,pop in pops.items():
   print("")
   print("Uncentering particles of population " + name)
   moveParticles(pop, -dims.dt/2, dims)
   
timers.toc("init")

if os.path.isfile(args.out_dir + "/fields.h5") or os.path.isfile(args.out_dir + "/pops.h5") or os.path.isfile(args.out_dir + "/logs.h5"):
   print("Output file(s) already exist, aborting")
   sys.exit()

save_data(args.out_dir, fields, pops, dims)

for jj in range(args.steps):
   print("")
   print("Starting time step " + str(jj + 1))
   
   timers.tic("locs")
   
   print("")
   print("Moving particles")
   for pop in pops.values():
      moveParticles(pop, dims.dt, dims)

   timers.toc("locs")
   timers.tic("alpha")
   timers.tic("field solver")
   
   print("Calculating alpha \"rotation\" matrices")
   for pop in pops.values():
      compute_alpha(pop, fields.faceB, dims)
   
   timers.toc("alpha")
   timers.tic("current")
      
   print("Computing rotated current")
   for pop in pops.values():
      fields.nodeJ_hat += compute_rotated_current(pop, dims)
   
   timers.toc("current")
   timers.tic("mass matrices")
   
   print("Computing mass matrices")
   if dims.oneV:
      mass_matrices = np.zeros((1,1,dims.Ncells_total,dims.Ncells_total))
   else:
      mass_matrices = np.zeros((3,3,dims.Ncells_total,dims.Ncells_total))
   
   for pop in pops.values():
      mass_matrices += compute_mass_matrices(pop, dims)
   
   timers.toc("mass matrices")
   # breakpoint()
   timers.tic("maxwell")
   
   print("Solving Maxwell's equations")

   timers.tic("build A")
   
   A = build_A(mass_matrices)

   timers.toc("build A")
   timers.tic("build b")

   b = build_b(fields.faceB, fields.nodeE, fields.nodeJ_hat, mass_matrices)
   
   timers.toc("build b")
   timers.tic("gmres")

   info = 1
   rtol = args.rtol
   atol = args.atol
   
   if dims.oneV:
      x0 = np.concatenate((fields.faceB[:,:,:,0].flat,fields.nodeE[:,:,:,0].flat))
   else:
      x0 = np.concatenate((fields.faceB.flat,fields.nodeE.flat))
   
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

   cellJ = np.zeros(dims.dim_vector, dtype = np.float64)
   for pop in pops.values():
      cellJ += pop.cellJi

   nodeJ = cell2node(cellJ, dims)
      
   implicitE = fields.nodeE - nodeJ*dims.dt/const.epsilon_0

   old_nodeE = fields.nodeE.copy()
   
   fields,midNodeE = upwind_fields(fields, xnext, dims)

   timers.toc("maxwell")
   timers.toc("field solver")
   timers.tic("lorentz")

   for pop in pops.values():
      Lorentz(pop, midNodeE, dims)
      # pop.v = Lorentz_njit(pop.r, pop.v, pop.alpha, pop.q, pop.m, midNodeE, dims)
   
   timers.toc("lorentz")
   timers.tic("extra")
   
   for pop in pops.values():
      accumulators(pop, dims)
      calcNodeData(pop, dims)
   # fields.update_fields(pops, dims)
   
   timers.toc("extra")

   # Increment time
   dims.step_time()
   
   timers.tic("output")

   save_data(args.out_dir, fields, pops, dims)

   timers.toc("output")

timers.toc("total")

print("")
print("Time per timestep and cell [ms/(timestep cell)]:")
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
