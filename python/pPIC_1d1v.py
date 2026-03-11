import sys
import os
import math
import numpy as np
import scipy as sp
import scipy.constants as const
from scipy.sparse.linalg import gmres
from timeit import default_timer as timer
import configparser
import argparse

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
         self.uniform_injector(x_min, x_max, Np, vth, v)
      
      self.accumulators()

   def uniform_injector(self, x_min, x_max, Np, vth, v):
      # Inject Np particles between xmin and xmax, bulk velocity v, and thermal velocity vth

      self.r = np.zeros((Np,3))
      self.v = np.zeros((Np,3))
      
      self.r[:,0] = rng.uniform(x_min,x_max, Np)
      if oneV is True:
         self.r[:,1] = (y_max + y_min)/2
         self.r[:,2] = (z_max + z_min)/2
      else:
         self.r[:,1] = rng.uniform(y_min,y_max, Np)
         self.r[:,2] = rng.uniform(z_min,z_max, Np)
      self.v[:,0] = rng.normal(v[0], vth, Np)
      if oneV is False:
         self.v[:,1] = rng.normal(v[1], vth, Np)
         self.v[:,2] = rng.normal(v[2], vth, Np)

      self.Np = Np
   
   def apply_boundaries(self):
      # Apply boundary conditions to particles
      # Periodic boundaries wrap locations
      # Non-periodic delete particles

      lims = ((x_min,x_max),(y_min,y_max),(z_min,z_max))
      sizes = (x_size,y_size,z_size)
      dgrid = (dx,dy,dz)

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
   
   def accumulators(self):
      # Accumulate macroparticles into cell-centred charge and current densities
      x_locs = np.round((self.r[:,0] - x_min)/dx).astype(int)
      y_locs = np.round((self.r[:,1] - y_min)/dy).astype(int)
      z_locs = np.round((self.r[:,2] - z_min)/dz).astype(int)

      if x_periodic:
         x_locs = np.mod(x_locs, x_size)
      if y_periodic:
         y_locs = np.mod(y_locs, y_size)
      if z_periodic:
         z_locs = np.mod(z_locs, z_size)
      
      self.rhoq = np.zeros(dim_scalar, dtype = float)
      self.cellJi = np.zeros(dim_vector, dtype = float)
      
      x0 = (x_locs - 1 + 0.5)*dx + x_min
      y0 = (y_locs - 1 + 0.5)*dy + y_min
      z0 = (z_locs - 1 + 0.5)*dz + z_min
      
      x_w1 = (self.r[:,0] - x0)/dx
      y_w1 = (self.r[:,1] - y0)/dy
      z_w1 = (self.r[:,2] - z0)/dz

      if x_periodic:
         x_w1 = np.mod(x_w1, 1)
      if y_periodic:
         y_w1 = np.mod(y_w1, 1)
      if z_periodic:
         z_w1 = np.mod(z_w1, 1)
      
      x_w0 = 1 - x_w1
      y_w0 = 1 - y_w1
      z_w0 = 1 - z_w1
      
      x_ind0 = arr_shift(range(x_size), 1, 0, (x_periodic,))[x_locs]
      x_ind1 = x_locs
      y_ind0 = arr_shift(range(y_size), 1, 0, (y_periodic,))[y_locs]
      y_ind1 = y_locs
      z_ind0 = arr_shift(range(z_size), 1, 0, (z_periodic,))[z_locs]
      z_ind1 = z_locs
      
      for x_ind_l,x_ind_r,x_w_l,x_w_r,y_ind_l,y_ind_r,y_w_l,y_w_r,z_ind_l,z_ind_r,z_w_l,z_w_r,v in zip(x_ind0,x_ind1,x_w0,x_w1,y_ind0,y_ind1,y_w0,y_w1,z_ind0,z_ind1,z_w0,z_w1,self.v):
         # CIC weight factors
         w_lll = x_w_l * y_w_l * z_w_l
         w_rll = x_w_r * y_w_l * z_w_l
         w_lrl = x_w_l * y_w_r * z_w_l
         w_llr = x_w_l * y_w_l * z_w_r
         w_rrl = x_w_r * y_w_r * z_w_l
         w_rlr = x_w_r * y_w_l * z_w_r
         w_lrr = x_w_l * y_w_r * z_w_r
         w_rrr = x_w_r * y_w_r * z_w_r
         
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

      self.rhoq *= self.q*self.w/vol
      self.cellJi *= self.q*self.w/vol

   def compute_alpha(self, faceB):
      # Compute alpha matrix for all particles in population
      rB = face2r(faceB, self.r)

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
      # Pushes particle positions by time dt
      self.r += self.v * dstep
      
      self.apply_boundaries()

   def Lorentz(self, midNodeE):
      # Accelerate particles via Lorentz force
      rE = node2r(midNodeE,self.r)
      
      beta = (self.q*dt)/(2*self.m)
      
      for ii,(r,v,alpha,E) in enumerate(zip(self.r,self.v,self.alpha,rE)):
         new_v = 2*alpha@(v + beta*E) - v
         
         self.v[ii] = new_v
      
   def nodeU(self):
      # Computes bulk velocity at nodes
      nodeU = np.zeros(dim_vector)
      node_w = np.zeros(dim_vector)
      
      for r,v in zip(self.r,self.v):
         x_locs = np.floor((r[0] - x_min)/dx).astype(int)
         y_locs = np.floor((r[1] - y_min)/dy).astype(int)
         z_locs = np.floor((r[2] - z_min)/dz).astype(int)
         
         x0 = x_locs*dx + x_min
         y0 = y_locs*dy + y_min
         z0 = z_locs*dz + z_min
         
         x_w1 = (r[0] - x0)/dx
         y_w1 = (r[1] - y0)/dy
         z_w1 = (r[2] - z0)/dz
         
         x_w0 = 1 - x_w1
         y_w0 = 1 - y_w1
         z_w0 = 1 - z_w1
         
         x_ind0 = x_locs
         x_ind1 = arr_shift(range(x_size), -1, 0, (x_periodic,))[x_locs]
         y_ind0 = y_locs
         y_ind1 = arr_shift(range(y_size), -1, 0, (y_periodic,))[y_locs]
         z_ind0 = z_locs
         z_ind1 = arr_shift(range(z_size), -1, 0, (z_periodic,))[z_locs]
         
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
   #    cellJ = np.zeros(dim_vector)
   #    
   #    for r,v,alpha in zip(self.r,self.v,self.alpha):
   #       x_locs = np.round((r[0] - x_min)/dx).astype(int)
   #       y_locs = np.round((r[1] - y_min)/dy).astype(int)
   #       z_locs = np.round((r[2] - z_min)/dz).astype(int)
   #       
   #       if x_periodic:
   #          x_locs = np.mod(x_locs, x_size)
   #       if y_periodic:
   #          y_locs = np.mod(y_locs, y_size)
   #       if z_periodic:
   #          z_locs = np.mod(z_locs, z_size)
   #       
   #       x0 = (x_locs - 1 + 0.5)*dx + x_min
   #       y0 = (y_locs - 1 + 0.5)*dy + y_min
   #       z0 = (z_locs - 1 + 0.5)*dz + z_min
   #       
   #       x_w1 = (r[0] - x0)/dx
   #       y_w1 = (r[1] - y0)/dy
   #       z_w1 = (r[2] - z0)/dz
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
   #       x_ind0 = arr_shift(range(x_size), 1, 0, (x_periodic,))[x_locs]
   #       x_ind1 = x_locs
   #       y_ind0 = arr_shift(range(y_size), 1, 0, (y_periodic,))[y_locs]
   #       y_ind1 = y_locs
   #       z_ind0 = arr_shift(range(z_size), 1, 0, (z_periodic,))[z_locs]
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
   #    cellJ *= self.q*self.w/vol
   #    
   #    nodeJ = cell2node(cellJ)
   #    
   #    return nodeJ

   def compute_rotated_current(self):
      # Computes current due to B-rotation
      # Current is stored at nodes to match E
      # This is accumulated directly to the nodes
      nodeJ = np.zeros(dim_vector)
      
      for r,v,alpha in zip(self.r,self.v,self.alpha):
         x_locs = np.floor((r[0] - x_min)/dx).astype(int)
         y_locs = np.floor((r[1] - y_min)/dy).astype(int)
         z_locs = np.floor((r[2] - z_min)/dz).astype(int)
         
         x0 = x_locs*dx + x_min
         y0 = y_locs*dy + y_min
         z0 = z_locs*dz + z_min
         
         x_w1 = (r[0] - x0)/dx
         y_w1 = (r[1] - y0)/dy
         z_w1 = (r[2] - z0)/dz
         
         x_w0 = 1 - x_w1
         y_w0 = 1 - y_w1
         z_w0 = 1 - z_w1
         
         x_ind0 = x_locs
         x_ind1 = arr_shift(range(x_size), -1, 0, (x_periodic,))[x_locs]
         y_ind0 = y_locs
         y_ind1 = arr_shift(range(y_size), -1, 0, (y_periodic,))[y_locs]
         z_ind0 = z_locs
         z_ind1 = arr_shift(range(z_size), -1, 0, (z_periodic,))[z_locs]
         
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

      nodeJ *= self.q*self.w/vol
      
      return nodeJ
   
   def compute_mass_matrices(self):
      # Compute mass matrices
      if oneV:
         M = np.zeros((1,1,Ncells_total,Ncells_total))
      else:
         M = np.zeros((3,3,Ncells_total,Ncells_total))
      
      for r,alpha in zip(self.r,self.alpha):
         x_locs = np.floor((r[0] - x_min)/dx).astype(int)
         y_locs = np.floor((r[1] - y_min)/dy).astype(int)
         z_locs = np.floor((r[2] - z_min)/dz).astype(int)

         x0 = x_locs*dx + x_min
         y0 = y_locs*dy + y_min
         z0 = z_locs*dz + z_min
         
         x_w1 = (r[0] - x0)/dx
         y_w1 = (r[1] - y0)/dy
         z_w1 = (r[2] - z0)/dz
         
         x_w0 = 1 - x_w1
         y_w0 = 1 - y_w1
         z_w0 = 1 - z_w1
         
         x_ind = (x_locs,arr_shift(range(x_size), -1, 0, (x_periodic,))[x_locs])
         y_ind = (y_locs,arr_shift(range(y_size), -1, 0, (y_periodic,))[y_locs])
         z_ind = (z_locs,arr_shift(range(z_size), -1, 0, (z_periodic,))[z_locs])

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
                                    M[ii,jj,(zi*y_size + yi)*x_size + xi,
                                      (zj*y_size + yj)*x_size + xj] += x_wi*x_wj*y_wi*y_wj*z_wi*z_wj * alpha[ii,jj]
      
      beta = (self.q*dt)/(2*self.m)
      M *= beta * self.q*self.w/vol
      
      return M

def split_axis(arr, axis, keep_dim = False):
   # Splits numpy array along axis into one array per axis index
   # If keep_dim is False the split axis is dropped from the array dimensions
   # i.e. an array of shape (2,3,4,3) split on axis 1 gives
   # 3 arrays of shape (2,4,3) if keep_dim is False, and
   # 3 arrays of shape (2,1,4,3) if keep_dim is True
   arr_shape = arr.shape
   axis_size = arr.shape[axis]
   
   split_arr = np.split(arr, axis_size, axis = axis)
   
   if keep_dim is False:
      sub_arr_shape = tuple(x for ii,x in enumerate(arr_shape) if ii != axis)
      for ii,sub_arr in enumerate(split_arr):
         split_arr[ii] = sub_arr.reshape(sub_arr_shape)
   
   return split_arr
      
def arr_shift(arr, shift, axis, periodicity, dimension = None):
   # Shift array, if axis is periodic then rolls otherwise uses array_indices
   # periodicity is iterable of bools of length arr.ndim that encodes if each axis is periodic
   # If axis is iterable of ints then rolls for each axis
   # If shift is not iterable of ints then uses same shift for all axes
   # Otherwise shift must be the same length as axis
   # i.e. the same as np.roll, but uses array_indices if not periodic
   if axis is None:
      # Skip function if no axis to shift
      return arr
   
   multiShift = False
   if not isinstance(shift, (int, np.int64, np.uint64)):
      shift = np.fromiter(shift, dtype = int)
      multiShift = True

   multiAxis = False
   if not isinstance(axis, (int, np.int64, np.uint64)):
      axis = np.fromiter(axis, dtype = int)
      multiAxis = True

   periodicity = np.fromiter(periodicity, dtype = bool)
   
   sh_arr = np.array(arr)
   
   if multiAxis:
      if multiShift:
         if axis.size != shift.size:
            raise ValueError("if not a single integer, length of shift must equal length of axis")
         for sh,ax in zip(shift, axis):
            sh_arr = arr_shift(sh_arr, sh, ax, periodicity)
      else:
         for ax in axis:
            sh_arr = arr_shift(sh_arr, shift, ax, periodicity)
   else:
      if shift == 0:
         return sh_arr
      if periodicity[axis]:
         sh_arr = np.roll(sh_arr, shift, axis)
      else:
         shape = sh_arr.shape
         ndim = sh_arr.ndim
         if dimension is None:
            dimension = axis
         if shift == 1:
            shift_indices = array_indices[dimension][:-2]
         elif shift == -1:
            shift_indices = array_indices[dimension][2:]
         else:
            raise ValueError("shift cannot be larger than 1 or smaller than -1 for non-periodic axes")
         shift_indices = shift_indices.reshape(tuple(-1 if jj == axis else 1 for jj in range(ndim)))
         ix = tuple(shift_indices if axis == ii else np.array(range(x)).reshape(tuple(-1 if jj == ii else 1 for jj in range(ndim))) for ii,x in enumerate(shape))
         sh_arr = sh_arr[ix]
   
   return sh_arr
      
def face2cell(face_data):
   # Interpolates data from cell faces to centre
   # Data is assumed to be vector data
   xface_data,yface_data,zface_data = split_axis(face_data, axis = 3)
   
   cell_data = face_data.copy()
   
   cell_data[:,:,:,0] += arr_shift(xface_data, -1, 2, period)
   cell_data[:,:,:,1] += arr_shift(yface_data, -1, 1, period)
   cell_data[:,:,:,2] += arr_shift(zface_data, -1, 0, period)
   
   cell_data *= 0.5
   
   return cell_data

def cell2node(cell_data):
   # Interpolates data from cell centres to nodes
   node_data = cell_data.copy()
   
   shift_0 = arr_shift(cell_data, 1, 0, period)
   shift_1 = arr_shift(cell_data, 1, 1, period)
   shift_2 = arr_shift(cell_data, 1, 2, period)
   shift_01 = arr_shift(shift_0, 1, 1, period)
   shift_02 = arr_shift(shift_2, 1, 0, period)
   shift_12 = arr_shift(shift_1, 1, 2, period)
   shift_012 = arr_shift(shift_01, 1, 2, period)
   
   node_data += shift_0
   node_data += shift_1
   node_data += shift_2
   node_data += shift_01
   node_data += shift_02
   node_data += shift_12
   node_data += shift_012
   
   node_data *= 0.125

   return node_data

def face2node(face_data):
   # Interpolate face data to node
   # Data is assumed to be vector data, face data only stores one component each
   node_data = face_data.copy()
   
   xface_data,yface_data,zface_data = split_axis(face_data, axis = 3)

   shift_0 = arr_shift(xface_data, 1, 0, period)
   shift_1 = arr_shift(xface_data, 1, 1, period)
   shift_01 = arr_shift(shift_0, 1, 1, period)

   node_data[:,:,:,0] += shift_0
   node_data[:,:,:,0] += shift_1
   node_data[:,:,:,0] += shift_01

   shift_0 = arr_shift(yface_data, 1, 0, period)
   shift_2 = arr_shift(yface_data, 1, 2, period)
   shift_02 = arr_shift(shift_0, 1, 2, period)

   node_data[:,:,:,1] += shift_0
   node_data[:,:,:,1] += shift_2
   node_data[:,:,:,1] += shift_02

   shift_1 = arr_shift(zface_data, 1, 1, period)
   shift_2 = arr_shift(zface_data, 1, 2, period)
   shift_12 = arr_shift(shift_1, 1, 2, period)

   node_data[:,:,:,2] += shift_1
   node_data[:,:,:,2] += shift_2
   node_data[:,:,:,2] += shift_12
   
   node_data *= 0.25
   
   return node_data   

def node2cell(node_data):
   # Interpolate node data to cell centres
   cell_data = node_data.copy()
   
   shift_0 = arr_shift(node_data, -1, 0, period)
   shift_1 = arr_shift(node_data, -1, 1, period)
   shift_2 = arr_shift(node_data, -1, 2, period)
   shift_01 = arr_shift(shift_0, -1, 1, period)
   shift_02 = arr_shift(shift_2, -1, 0, period)
   shift_12 = arr_shift(shift_1, -1, 2, period)
   shift_012 = arr_shift(shift_01, -1, 2, period)

   cell_data += shift_0
   cell_data += shift_1
   cell_data += shift_2
   cell_data += shift_01
   cell_data += shift_02
   cell_data += shift_12
   cell_data += shift_012
   
   cell_data *= 0.125
   
   return cell_data

def node2face(node_data):
   # Interpolate node data to face
   # Data is assumed to be vector data, face data only stores one component each
   face_data = np.zeros(dim_vector)

   xnode_data,ynode_data,znode_data = split_axis(node_data, axis = 3)

   face_data = node_data.copy()
   
   shift_0 = arr_shift(xnode_data, -1, 0, period)
   shift_1 = arr_shift(xnode_data, -1, 1, period)
   shift_01 = arr_shift(shift_0, -1, 1, period)
   
   face_data[:,:,:,0] += shift_0
   face_data[:,:,:,0] += shift_1
   face_data[:,:,:,0] += shift_01

   shift_0 = arr_shift(ynode_data, -1, 0, period)
   shift_2 = arr_shift(ynode_data, -1, 2, period)
   shift_02 = arr_shift(shift_0, -1, 2, period)
   
   face_data[:,:,:,1] += shift_0
   face_data[:,:,:,1] += shift_2
   face_data[:,:,:,1] += shift_02

   shift_1 = arr_shift(znode_data, -1, 1, period)
   shift_2 = arr_shift(znode_data, -1, 2, period)
   shift_12 = arr_shift(shift_1, -1, 2, period)
   
   face_data[:,:,:,2] += shift_1
   face_data[:,:,:,2] += shift_2
   face_data[:,:,:,2] += shift_12
   
   face_data *= 0.25
   
   return face_data

def cell2face(cell_data):
   # Interpolates data from cell centres to faces
   # Data is assumed to be vector data, face data only stores one component each
   face_data = cell_data.copy()
   
   for ii in range(3):
      vector_comp = 2 - ii
      face_data[:,:,:,vector_comp] += arr_shift(cell_data[:,:,:,vector_comp], 1, ii, period)

   face_data *= 0.5
   
   return face_data

def face2r(face_data,r):
   # Interpolate face data to arbitrary position(s) r
   # Data is assumed to be vector data, face data only stores one component each
   r = np.array(r).reshape(-1,3)
   r_data = np.zeros(r.shape)

   xface_data,yface_data,zface_data = split_axis(face_data, axis = 3)
   
   x_locs = np.floor((r[:,0] - x_min)/dx).astype(int)
   y_locs = np.floor((r[:,1] - y_min)/dy).astype(int)
   z_locs = np.floor((r[:,2] - z_min)/dz).astype(int)

   x0 = x_locs*dx + x_min
   y0 = y_locs*dy + y_min
   z0 = z_locs*dz + z_min
   x_w1 = (r[:,0] - x0)/dx
   y_w1 = (r[:,1] - y0)/dy
   z_w1 = (r[:,2] - z0)/dz
   x_w0 = 1 - x_w1
   y_w0 = 1 - y_w1
   z_w0 = 1 - z_w1
   
   x_locs_r = arr_shift(range(x_size), -1, 0, (x_periodic,), 2)[x_locs]
   y_locs_r = arr_shift(range(y_size), -1, 0, (y_periodic,), 1)[y_locs]
   z_locs_r = arr_shift(range(z_size), -1, 0, (z_periodic,), 0)[z_locs]
   
   r_data[:,0] += x_w0*xface_data[z_locs,y_locs,x_locs]
   r_data[:,0] += x_w1*xface_data[z_locs,y_locs,x_locs_r]

   r_data[:,1] += y_w0*yface_data[z_locs,y_locs,x_locs]
   r_data[:,1] += y_w1*yface_data[z_locs,y_locs_r,x_locs]

   r_data[:,2] += z_w0*zface_data[z_locs,y_locs,x_locs]
   r_data[:,2] += z_w1*zface_data[z_locs_r,y_locs,x_locs]

   if r.shape[0] == 1:
      r_data = r_data.reshape(3)
   
   return r_data

def node2r(node_data,r):
   # Interpolate node data to arbitrary position(s) r
   r = np.array(r).reshape(-1,3)
   vec = (node_data.ndim == 4)

   if vec:
      r_data = np.zeros(r.shape)
   else:
      r_data = np.zeros(r.shape[0:-1])
   
   x_locs = np.floor((r[:,0] - x_min)/dx).astype(int)
   y_locs = np.floor((r[:,1] - y_min)/dy).astype(int)
   z_locs = np.floor((r[:,2] - z_min)/dz).astype(int)
   
   x0 = x_locs*dx + x_min
   y0 = y_locs*dy + y_min
   z0 = z_locs*dz + z_min
   x_w1 = (r[:,0] - x0)/dx
   y_w1 = (r[:,1] - y0)/dy
   z_w1 = (r[:,2] - z0)/dz
   x_w0 = 1 - x_w1
   y_w0 = 1 - y_w1
   z_w0 = 1 - z_w1
   
   weights = np.array([np.sqrt(x_w0**2 + y_w0**2 + z_w0**2),
                       np.sqrt(x_w1**2 + y_w0**2 + z_w0**2),
                       np.sqrt(x_w0**2 + y_w1**2 + z_w0**2),
                       np.sqrt(x_w1**2 + y_w1**2 + z_w0**2),
                       np.sqrt(x_w0**2 + y_w0**2 + z_w1**2),
                       np.sqrt(x_w1**2 + y_w0**2 + z_w1**2),
                       np.sqrt(x_w0**2 + y_w1**2 + z_w1**2),
                       np.sqrt(x_w1**2 + y_w1**2 + z_w1**2)])
   
   weights /= np.linalg.norm(weights, axis = 0, keepdims = True)
   weights /= np.sum(weights, axis = 0)
   if vec:
      weights = weights.reshape(8,-1,1)
   else:
      weights = weights.reshape(8,-1)

   # r_data += weights[0]*node_data[array_indices[0][z_locs+1],
   #                                array_indices[1][y_locs+1],
   #                                array_indices[2][x_locs+1],...]
   # r_data += weights[1]*node_data[array_indices[0][z_locs+2],
   #                                array_indices[1][y_locs+1],
   #                                array_indices[2][x_locs+1],...]
   # r_data += weights[2]*node_data[array_indices[0][z_locs+1],
   #                                array_indices[1][y_locs+2],
   #                                array_indices[2][x_locs+1],...]
   # r_data += weights[3]*node_data[array_indices[0][z_locs+2],
   #                                array_indices[1][y_locs+2],
   #                                array_indices[2][x_locs+1],...]
   # r_data += weights[4]*node_data[array_indices[0][z_locs+1],
   #                                array_indices[1][y_locs+1],
   #                                array_indices[2][x_locs+2],...]
   # r_data += weights[5]*node_data[array_indices[0][z_locs+2],
   #                                array_indices[1][y_locs+1],
   #                                array_indices[2][x_locs+2],...]
   # r_data += weights[6]*node_data[array_indices[0][z_locs+1],
   #                                array_indices[1][y_locs+2],
   #                                array_indices[2][x_locs+2],...]
   # r_data += weights[7]*node_data[array_indices[0][z_locs+2],
   #                                array_indices[1][y_locs+2],
   #                                array_indices[2][x_locs+2],...]

   x_locs_r = arr_shift(range(x_size), -1, 0, (x_periodic,), 2)[x_locs]
   y_locs_r = arr_shift(range(y_size), -1, 0, (y_periodic,), 1)[y_locs]
   z_locs_r = arr_shift(range(z_size), -1, 0, (z_periodic,), 0)[z_locs]
   
   r_data += weights[0]*node_data[z_locs, y_locs, x_locs,...]
   r_data += weights[1]*node_data[z_locs_r, y_locs, x_locs,...]
   r_data += weights[2]*node_data[z_locs, y_locs_r, x_locs,...]
   r_data += weights[3]*node_data[z_locs_r, y_locs_r, x_locs,...]
   r_data += weights[4]*node_data[z_locs, y_locs, x_locs_r,...]
   r_data += weights[5]*node_data[z_locs_r, y_locs, x_locs_r,...]
   r_data += weights[6]*node_data[z_locs, y_locs_r, x_locs_r,...]
   r_data += weights[7]*node_data[z_locs_r, y_locs_r, x_locs_r,...]

   if r.shape[0] == 1:
      r_data = r_data.reshape(3)
   
   return r_data

def cell2r(cell_data,r):
   # Interpolate cell data to arbitrary position(s) r
   r = np.array(r).reshape(-1,3)
   vec = (cell_data.ndim == 4)

   if vec:
      r_data = np.zeros(r.shape)
   else:
      r_data = np.zeros(r.shape[0:-1])

   x_locs = np.round((r[:,0] - x_min)/dx).astype(int)
   y_locs = np.round((r[:,1] - y_min)/dy).astype(int)
   z_locs = np.round((r[:,2] - z_min)/dz).astype(int)
   
   if x_periodic:
      x_locs = np.mod(x_locs, x_size)
   if y_periodic:
      y_locs = np.mod(y_locs, y_size)
   if z_periodic:
      z_locs = np.mod(z_locs, z_size)
   
   x0 = (x_locs - 1 + 0.5)*dx + x_min
   y0 = (y_locs - 1 + 0.5)*dy + y_min
   z0 = (z_locs - 1 + 0.5)*dz + z_min
   x_w1 = (r[:,0] - x0)/dx
   y_w1 = (r[:,1] - y0)/dy
   z_w1 = (r[:,2] - z0)/dz
   
   if x_periodic:
      x_w1 = np.mod(x_w1, 1)
   if y_periodic:
      y_w1 = np.mod(y_w1, 1)
   if z_periodic:
      z_w1 = np.mod(z_w1, 1)
   
   x_w0 = 1 - x_w1
   y_w0 = 1 - y_w1
   z_w0 = 1 - z_w1
   
   weights = np.array([np.sqrt(x_w0**2 + y_w0**2 + z_w0**2),
                       np.sqrt(x_w1**2 + y_w0**2 + z_w0**2),
                       np.sqrt(x_w0**2 + y_w1**2 + z_w0**2),
                       np.sqrt(x_w1**2 + y_w1**2 + z_w0**2),
                       np.sqrt(x_w0**2 + y_w0**2 + z_w1**2),
                       np.sqrt(x_w1**2 + y_w0**2 + z_w1**2),
                       np.sqrt(x_w0**2 + y_w1**2 + z_w1**2),
                       np.sqrt(x_w1**2 + y_w1**2 + z_w1**2)])
   
   weights /= np.linalg.norm(weights, axis = 0, keepdims = True)
   weights /= np.sum(weights, axis = 0)
   if vec:
      weights = weights.reshape(8,-1,1)
   else:
      weights = weights.reshape(8,-1)
   
   # r_data += weights[0]*cell_data[array_indices[0][z_locs],
   #                                array_indices[1][y_locs],
   #                                array_indices[2][x_locs],...]
   # r_data += weights[1]*cell_data[array_indices[0][z_locs+1],
   #                                array_indices[1][y_locs],
   #                                array_indices[2][x_locs],...]
   # r_data += weights[2]*cell_data[array_indices[0][z_locs],
   #                                array_indices[1][y_locs+1],
   #                                array_indices[2][x_locs],...]
   # r_data += weights[3]*cell_data[array_indices[0][z_locs+1],
   #                                array_indices[1][y_locs+1],
   #                                array_indices[2][x_locs],...]
   # r_data += weights[4]*cell_data[array_indices[0][z_locs],
   #                                array_indices[1][y_locs],
   #                                array_indices[2][x_locs+1],...]
   # r_data += weights[5]*cell_data[array_indices[0][z_locs+1],
   #                                array_indices[1][y_locs],
   #                                array_indices[2][x_locs+1],...]
   # r_data += weights[6]*cell_data[array_indices[0][z_locs],
   #                                array_indices[1][y_locs+1],
   #                                array_indices[2][x_locs+1],...]
   # r_data += weights[7]*cell_data[array_indices[0][z_locs+1],
   #                                array_indices[1][y_locs+1],
   #                                array_indices[2][x_locs+1],...]
   
   x_locs_l = arr_shift(range(x_size), 1, 0, (x_periodic,), 2)[x_locs]
   y_locs_l = arr_shift(range(y_size), 1, 0, (y_periodic,), 1)[y_locs]
   z_locs_l = arr_shift(range(z_size), 1, 0, (z_periodic,), 0)[z_locs]
   
   r_data += weights[0]*cell_data[z_locs_l, y_locs_l, x_locs_l,...]
   r_data += weights[1]*cell_data[z_locs, y_locs_l, x_locs_l,...]
   r_data += weights[2]*cell_data[z_locs_l, y_locs, x_locs_l,...]
   r_data += weights[3]*cell_data[z_locs, y_locs, x_locs_l,...]
   r_data += weights[4]*cell_data[z_locs_l, y_locs_l, x_locs,...]
   r_data += weights[5]*cell_data[z_locs, y_locs_l, x_locs,...]
   r_data += weights[6]*cell_data[z_locs_l, y_locs, x_locs,...]
   r_data += weights[7]*cell_data[z_locs, y_locs, x_locs,...]

   if r.shape[0] == 1:
      r_data = r_data.reshape(3)
   
   return r_data

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
      macros = config.getfloat(pop_name, "macroparticles_per_cell", fallback = None)

      Np = int(macros*x_size)
      weight = density*vol/macros

      if electron:
         mass = const.m_p/args.mass_ratio
      else:
         mass *= const.m_p

      charge *= const.e

      pops[pop_name] = Pop(pop_name, charge, mass, weight, electron, Np, temp, velocity)

def initialise_fields():
   # Initialise all fields
   print("")
   print("Initialising all fields")

   global nodeJ,faceB,nodeE
   
   cellJp = np.sum([x.cellJi for x in pops.values()], axis = 0)

   faceJp = cell2face(cellJp)
   
   faceB = np.zeros(dim_vector)
   
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

   nodeB = face2node(faceB)
   nodeUe = pops["e-"].nodeU()

   nodeE = -np.cross(nodeUe, nodeB)

def build_A(mass_matrices):
   # Construct sparse matrix A of Ax = b equation representing Maxwell's equations
   if oneV:
      A = np.zeros((2*Ncells_total,2*Ncells_total))
      A[0:Ncells_total,Ncells_total:2*Ncells_total] -= theta*mass_matrices[0,0]
      for ii in range(Ncells_total):
         A[ii,ii + Ncells_total] -= 1/(const.c**2*dt)
      
      for ii in range(Ncells_total):
         A[ii + Ncells_total,ii] += 1/dt
   else:
      A = np.zeros((6*Ncells_total,6*Ncells_total))
      
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
      b = np.zeros(2*Ncells_total)
      b[:Ncells_total] = -1/(const.c**2*dt)*Ex+const.mu_0*Jx+(1-theta)*mass_matrices[0,0]@Ex
      
      b[Ncells_total:] = 1/dt*Bx
   else:
      b = np.zeros(6*Ncells_total)
   
   return b
            
def test_interpolators():
   # Test interpolation methods
   Np = 10
   r = rng.uniform((x_min,y_min,z_min), (x_max,y_max,z_max), (Np,3))
   
   cell_s = np.array(range(np.prod(dim_scalar)), dtype = float).reshape(dim_scalar)
   cell_v = np.array(range(np.prod(dim_vector)), dtype = float).reshape(dim_vector)

   # cell_s = rng.uniform(-1, 1, dim_scalar)
   # cell_v = rng.uniform(-1, 1, dim_vector)
   
   cell_to_face = cell2face(cell_v)
   cell_to_node_s = cell2node(cell_s)
   cell_to_node_v = cell2node(cell_v)
   cell_to_r_s = cell2r(cell_s, r)
   cell_to_r_v = cell2r(cell_v, r)

   node_s = np.array(range(np.prod(dim_scalar)), dtype = float).reshape(dim_scalar)
   node_v = np.array(range(np.prod(dim_vector)), dtype = float).reshape(dim_vector)

   # node_s = rng.uniform(-1, 1, dim_scalar)
   # node_v = rng.uniform(-1, 1, dim_vector)
   
   node_to_face = node2face(node_v)
   node_to_cell_s = node2cell(node_s)
   node_to_cell_v = node2cell(node_v)
   node_to_r_s = node2r(node_s, r)
   node_to_r_v = node2r(node_v, r)

   face = np.array(range(np.prod(dim_vector)), dtype = float).reshape(dim_vector)
   
   # face = rng.uniform(-1, 1, dim_vector)

   face_to_cell = face2cell(face)
   face_to_node = face2node(face)
   face_to_r = face2r(face, r)
   
args,config = readConfig(args.config, args)

dt = args.dt
theta = args.theta

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
      
period = (z_periodic, y_periodic, x_periodic, True)
      
dim_scalar = (z_size,y_size,x_size)
dim_vector = (z_size,y_size,x_size,3)

Ncells_total = np.product(dim_scalar)

vol = dx*dy*dz

array_indices = []

tmp = np.array(range(-1, z_size + 1))
tmp[0] = z_size - 1
tmp[-1] = 0
array_indices.append(tmp)

tmp = np.array(range(-1, y_size + 1))
tmp[0] = y_size - 1
tmp[-1] = 0
array_indices.append(tmp)

tmp = np.array(range(-1, x_size + 1))
tmp[0] = x_size - 1
tmp[-1] = 0
array_indices.append(tmp)

test_interpolators()

timers = my_timers()
timers.tic("total")
timers.tic("init")

initialise_populations()

initialise_fields()

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
   nodeJ_hat = np.zeros(dim_vector)
   for pop in pops.values():
      nodeJ_hat += pop.compute_rotated_current()

   timers.toc("current")
   timers.tic("mass matrices")
      
   print("Computing mass matrices")
   if oneV:
      mass_matrices = np.zeros((1,1,Ncells_total,Ncells_total))
   else:
      mass_matrices = np.zeros((3,3,Ncells_total,Ncells_total))
   
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
      newFaceB[:,:,:,0] = xnext[:Ncells_total].reshape(dim_scalar)
      newNodeE[:,:,:,0] = xnext[Ncells_total:].reshape(dim_scalar)
   
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
   
breakpoint()
