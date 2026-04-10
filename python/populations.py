# Module of Population class and population functions, e.g. accumulators, alpha, mass matrices etc.

import math
import functools as ftools
import numpy as np
import scipy as sp
import scipy.constants as const
from numba import njit,int64,float64,guvectorize
from numba.experimental import jitclass

from interpolators import face2r,cell2r,node2r,face2r_njit,cell2r_njit,node2r_njit
from indexers import split_axis,numba_unravel_index,CIC_weights_node,CIC_weights_cell,numba_clip,CIC_weights_node_njit,CIC_weights_cell_njit,get_particle_cellid_njit,get_index_njit

class Pop:
   # Contents:
   # name:     population name
   # q:        individual particle charge
   # m:        individual particle mass
   # w:        macroparticle weight (particles per macroparticle)
   # Np:       total number of macroparticles
   # ID:       id no. of macroparticle
   # r:        macroparticle position
   # v:        macroparticle velocity
   # cids:     macroparticle cellid
   # cellJi:   population cell current density
   # cellRhoQ: population cell charge density
   # nodeU:    population node bulk velocity
   # nodeN:    population node number density
   # static:   if True then population positions and velocities will not be updated, and impact on fields is ignored
   def __init__(self, name, q, m, w, electron, Np, rng, dims, T = 0, v = 0, static = False):
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
      
      self.ID = np.empty((0), dtype = np.int64)
      self.r = np.empty((0,3), dtype = np.float64)
      self.v = np.empty((0,3), dtype = np.float64)
      self.cids = np.empty((0,), dtype = np.float64)
      
      if Np > 0:
         vth = math.sqrt(const.k*T/self.m)
         uniform_injector(self, Np, vth, v, rng, dims)
      
      accumulators(self, dims)
      calcNodeData(self, dims)
      
      self.static = static

      self.sort()

   def sort(self, alpha = False):
      # Sort particles in order of cellid
      # Note: if alpha is False then skips sorting alpha
      # (since they will be recalculated)
      cellid_argsort = self.cids.argsort()

      self.cids = self.cids[cellid_argsort]
      self.r = self.r[cellid_argsort]
      self.v = self.v[cellid_argsort]

   def getParams(self, fields, dims):
      # Compute the average population parameters
      Nparts = self.Np*self.w
      
      dens = Nparts/(dims.Ncells_total*dims.dV)
      meanB_mag = np.mean(np.linalg.norm(fields.faceB, axis = -1))
      meanB = np.mean(fields.faceB, axis = (0,1,2))
      if all(meanB == 0.0):
         unitB = np.array([1,0,0])
      else:
         unitB = meanB/np.linalg.norm(meanB)
      
      omega_p = math.sqrt(dens*self.q**2/(self.m*const.epsilon_0))
      gyro_f = self.q*meanB_mag/self.m
      
      meanV = (self.w*self.v).sum(axis = 0)/Nparts

      meanV_perp = (self.w*np.linalg.norm(self.v - (self.v@meanB)[:,np.newaxis]*unitB, axis = 1)).sum()/Nparts

      gyro_r = meanV_perp/gyro_f

      inertial = const.c/omega_p
      
      vth = np.sqrt((self.w*(self.v - meanV)**2).sum()/Nparts)
      if dims.oneV:
         temp = vth**2*self.m/const.k
      else:
         temp = vth**2*self.m/(3*const.k)

      return dens,gyro_f,gyro_r,inertial,meanV,vth,temp
      
def uniform_injector(pop, Np, vth, v, rng, dims):
   # Inject Np particles between xmin and xmax, bulk velocity v, and thermal velocity vth
   
   r_ins = np.empty((Np*dims.Ncells_total,3), dtype = np.float64)
   v_ins = np.zeros((Np*dims.Ncells_total,3), dtype = np.float64)
   cid_ins = np.empty((Np*dims.Ncells_total), dtype = np.float64)

   ID = np.fromiter(range(Np*dims.Ncells_total), dtype = np.int64)
   
   for ii in range(dims.Ncells_total):
      zi,yi,xi = np.unravel_index(ii, dims.dim_scalar)

      cell_x_min = dims.x_min + xi*dims.dx
      cell_x_max = dims.x_max - (dims.x_size - 1 - xi)*dims.dx
      cell_y_min = dims.y_min + yi*dims.dy
      cell_y_max = dims.y_max - (dims.y_size - 1 - yi)*dims.dy
      cell_z_min = dims.z_min + zi*dims.dz
      cell_z_max = dims.z_max - (dims.z_size - 1 - zi)*dims.dz

      r_ins[ii*Np:(ii+1)*Np,0] = rng.uniform(cell_x_min,cell_x_max, Np)
      if dims.oneV is True:
         r_ins[ii*Np:(ii+1)*Np,1] = (cell_y_max + cell_y_min)/2
         r_ins[ii*Np:(ii+1)*Np,2] = (cell_z_max + cell_z_min)/2
      else:
         r_ins[ii*Np:(ii+1)*Np,1] = rng.uniform(cell_y_min,cell_y_max, Np)
         r_ins[ii*Np:(ii+1)*Np,2] = rng.uniform(cell_z_min,cell_z_max, Np)
      v_ins[ii*Np:(ii+1)*Np,0] = rng.normal(v[0], vth, Np)
      if dims.oneV is False:
         v_ins[ii*Np:(ii+1)*Np,1] = rng.normal(v[1], vth, Np)
         v_ins[ii*Np:(ii+1)*Np,2] = rng.normal(v[2], vth, Np)

      v_ins[ii*Np:(ii+1)*Np,0] += 0.1*vth*np.sin(2*math.pi*3*r_ins[ii*Np:(ii+1)*Np,0]/(dims.x_max - dims.x_min))
      
      cid_ins[ii*Np:(ii+1)*Np] = ii
      
   if pop.ID.size == 0:
      ID_max = -1
   else:
      ID_max = np.max(pop.ID)
      
   pop.ID = np.concatenate((pop.ID,ID + ID_max + 1))
   pop.r = np.concatenate((pop.r,r_ins))
   pop.v = np.concatenate((pop.v,v_ins))
   pop.cids = np.concatenate((pop.cids, cid_ins))
   
   pop.Np = pop.r.shape[0]

def apply_boundaries_parts(pop, dims):
   # Apply boundary conditions to particles
   # Periodic boundaries wrap locations
   # Non-periodic delete particles

   lims = ((dims.x_min,dims.x_max),(dims.y_min,dims.y_max),(dims.z_min,dims.z_max))
   sizes = (dims.x_size,dims.y_size,dims.z_size)
   dgrid = (dims.dx,dims.dy,dims.dz)

   del_list = np.repeat(False, pop.Np)
   
   for ii,(lim,rep) in enumerate(zip(lims,dims.period)):
      if rep:
         pop.r[:,ii] = np.mod(pop.r[:,ii] - lim[0], lim[1] - lim[0]) + lim[0]
      else:
         del_list |= pop.r[:,ii] < lim[0] | pop.r[:,ii] >= lim[1]

   if not all(dims.period) and any(del_list):
      removeParticle(pop, del_list)

def removeParticle(pop, to_delete):
   # Delete particle from population
   to_keep = np.logical_not(to_delete)
   pop.ID = pop.ID[to_keep]
   pop.r = pop.r[to_keep]
   pop.v = pop.v[to_keep]
   pop.alpha = pop.alpha[to_keep]
   pop.Np = np.sum(to_keep)

def accumulators(pop, dims):
   # Accumulate macroparticles into cell-centred charge and current densities
   pop.cellRhoQ = np.zeros(dims.dim_scalar, dtype = float)
   pop.cellJi = np.zeros(dims.dim_vector, dtype = float)
   
   # CIC weight factors
   (x_ind,y_ind,z_ind),(x_w,y_w,z_w) = CIC_weights_cell(pop.r, dims)
   w = z_w*y_w*x_w

   np.add.at(pop.cellRhoQ, (z_ind[0],y_ind[0],x_ind[0]), w[0,0,0])
   np.add.at(pop.cellRhoQ, (z_ind[0],y_ind[0],x_ind[1]), w[0,0,1])
   np.add.at(pop.cellRhoQ, (z_ind[0],y_ind[1],x_ind[0]), w[0,1,0])
   np.add.at(pop.cellRhoQ, (z_ind[0],y_ind[1],x_ind[1]), w[0,1,1])
   np.add.at(pop.cellRhoQ, (z_ind[1],y_ind[0],x_ind[0]), w[1,0,0])
   np.add.at(pop.cellRhoQ, (z_ind[1],y_ind[0],x_ind[1]), w[1,0,1])
   np.add.at(pop.cellRhoQ, (z_ind[1],y_ind[1],x_ind[0]), w[1,1,0])
   np.add.at(pop.cellRhoQ, (z_ind[1],y_ind[1],x_ind[1]), w[1,1,1])
   
   np.add.at(pop.cellJi[:,:,:], (z_ind[0],y_ind[0],x_ind[0]), w[0,0,0].reshape(-1,1)*pop.v)
   np.add.at(pop.cellJi[:,:,:], (z_ind[0],y_ind[0],x_ind[1]), w[0,0,1].reshape(-1,1)*pop.v)
   np.add.at(pop.cellJi[:,:,:], (z_ind[0],y_ind[1],x_ind[0]), w[0,1,0].reshape(-1,1)*pop.v)
   np.add.at(pop.cellJi[:,:,:], (z_ind[0],y_ind[1],x_ind[1]), w[0,1,1].reshape(-1,1)*pop.v)
   np.add.at(pop.cellJi[:,:,:], (z_ind[1],y_ind[0],x_ind[0]), w[1,0,0].reshape(-1,1)*pop.v)
   np.add.at(pop.cellJi[:,:,:], (z_ind[1],y_ind[0],x_ind[1]), w[1,0,1].reshape(-1,1)*pop.v)
   np.add.at(pop.cellJi[:,:,:], (z_ind[1],y_ind[1],x_ind[0]), w[1,1,0].reshape(-1,1)*pop.v)
   np.add.at(pop.cellJi[:,:,:], (z_ind[1],y_ind[1],x_ind[1]), w[1,1,1].reshape(-1,1)*pop.v)
   
   pop.cellRhoQ *= pop.q*pop.w/dims.dV
   pop.cellJi *= pop.q*pop.w/dims.dV

def compute_alpha(pop, faceB, dims):
   # Compute alpha matrix for all particles in population
   rB = face2r_njit(faceB, pop.r, dims)
   
   beta = dims.phi*(pop.q*dims.dt)/pop.m

   bx,by,bz = split_axis(rB*beta, axis = 1)

   pop.alpha = np.zeros((pop.Np,3,3))
   if dims.oneV is True:
      pop.alpha[:,0,0] = 1
   else:
      factor = 1/(1 + bx**2 + by**2 + bz**2)
      
      pop.alpha[:,0,0] = bx*bx + 1
      pop.alpha[:,0,1] = bx*by + bz
      pop.alpha[:,0,2] = bx*bz - by
      pop.alpha[:,1,0] = bx*by - bz
      pop.alpha[:,1,1] = by*by + 1
      pop.alpha[:,1,2] = by*bz + bx
      pop.alpha[:,2,0] = bx*bz + by
      pop.alpha[:,2,1] = by*bz - bx
      pop.alpha[:,2,2] = bz*bz + 1

      pop.alpha *= factor.reshape(pop.Np,1,1)

def getCellids(pop, dims):
   # Get cellid for all particles in population
   mins = np.array([dims.x_min,dims.y_min,dims.z_min])
   ds = np.array([dims.dx,dims.dy,dims.dz])
   grid = np.array([dims.z_size,dims.y_size,dims.x_size])
   
   dim_ids = np.floor((pop.r - mins)/ds).astype(int)
   
   cids = np.ravel_multi_index(np.flip(dim_ids, axis = 1).transpose(), grid)

   return cids
      
def moveParticles(pop, dstep, dims):
   # Pushes particle positions by time dstep
   pop.r += pop.v * dstep
   
   apply_boundaries_parts(pop, dims)
   
   pop.cids = getCellids(pop, dims)
   
def Lorentz(pop, midNodeE, dims):
   # Accelerate particles via Lorentz force
   rE = node2r_njit(midNodeE, pop.r, dims)
   
   beta = dims.phi*(pop.q*dims.dt)/pop.m

   v_phi = np.matvec(pop.alpha, (pop.v + beta*rE))
   
   pop.v = (v_phi - (1-dims.phi)*pop.v)/dims.phi
   
   # for ii,(v,alpha,E) in enumerate(zip(pop.v,pop.alpha,rE)):
   #    new_v = 2*alpha@(v + beta*E) - v

   #    pop.v[ii] = new_v

def calcNodeData(pop, dims):
   # Computes bulk velocity, density, and temperature at nodes
   pop.nodeU = np.zeros(dims.dim_vector)
   pop.nodeN = np.zeros(dims.dim_scalar)
   pop.nodeT = np.zeros(dims.dim_scalar)
   
   (x_ind,y_ind,z_ind),(x_w,y_w,z_w) = CIC_weights_node(pop.r, dims)
   
   v = np.transpose(pop.v)
   
   if dims.oneV is True:
      w = x_w.reshape(2,-1)

      v_0 = v[0].copy()
      
      np.add.at(pop.nodeU[:,:,:,0], (z_ind[0],y_ind[0],x_ind[0]), w[0]*v_0)
      np.add.at(pop.nodeU[:,:,:,0], (z_ind[0],y_ind[0],x_ind[1]), w[1]*v_0)

      np.add.at(pop.nodeN, (z_ind[0],y_ind[0],x_ind[0]), w[0])
      np.add.at(pop.nodeN, (z_ind[0],y_ind[0],x_ind[1]), w[1])
   else:
      w = z_w*y_w*x_w

      for nn in range(3):
         v_nn = v[nn].copy()
         
         np.add.at(pop.nodeU[:,:,:,nn], (z_ind[0],y_ind[0],x_ind[0]), w[0,0,0]*v_nn)
         np.add.at(pop.nodeU[:,:,:,nn], (z_ind[0],y_ind[0],x_ind[1]), w[0,0,1]*v_nn)
         np.add.at(pop.nodeU[:,:,:,nn], (z_ind[0],y_ind[1],x_ind[0]), w[0,1,0]*v_nn)
         np.add.at(pop.nodeU[:,:,:,nn], (z_ind[0],y_ind[1],x_ind[1]), w[0,1,1]*v_nn)
         np.add.at(pop.nodeU[:,:,:,nn], (z_ind[1],y_ind[0],x_ind[0]), w[1,0,0]*v_nn)
         np.add.at(pop.nodeU[:,:,:,nn], (z_ind[1],y_ind[0],x_ind[1]), w[1,0,1]*v_nn)
         np.add.at(pop.nodeU[:,:,:,nn], (z_ind[1],y_ind[1],x_ind[0]), w[1,1,0]*v_nn)
         np.add.at(pop.nodeU[:,:,:,nn], (z_ind[1],y_ind[1],x_ind[1]), w[1,1,1]*v_nn)

      np.add.at(pop.nodeN, (z_ind[0],y_ind[0],x_ind[0]), w[0,0,0])
      np.add.at(pop.nodeN, (z_ind[0],y_ind[0],x_ind[1]), w[0,0,1])
      np.add.at(pop.nodeN, (z_ind[0],y_ind[1],x_ind[0]), w[0,1,0])
      np.add.at(pop.nodeN, (z_ind[0],y_ind[1],x_ind[1]), w[0,1,1])
      np.add.at(pop.nodeN, (z_ind[1],y_ind[0],x_ind[0]), w[1,0,0])
      np.add.at(pop.nodeN, (z_ind[1],y_ind[0],x_ind[1]), w[1,0,1])
      np.add.at(pop.nodeN, (z_ind[1],y_ind[1],x_ind[0]), w[1,1,0])
      np.add.at(pop.nodeN, (z_ind[1],y_ind[1],x_ind[1]), w[1,1,1])
   
   pop.nodeU /= pop.nodeN[:,:,:,np.newaxis]
   pop.nodeN *= pop.w/dims.dV
   
   if dims.oneV is True:
      w = x_w.reshape(2,-1)

      v_0 = (v[0] - pop.nodeU[z_ind[0],y_ind[0],x_ind,0])**2

      v_0 *= w
      
      np.add.at(pop.nodeT, (z_ind[0],y_ind[0],x_ind[0]), v_0[0])
      np.add.at(pop.nodeT, (z_ind[0],y_ind[0],x_ind[1]), v_0[1])
   else:
      w = z_w*y_w*x_w

      x_ind_alt = x_ind.reshape(1,1,2,-1)
      y_ind_alt = y_ind.reshape(1,2,1,-1)
      z_ind_alt = z_ind.reshape(2,1,1,-1)
      
      v_nn = np.sum((pop.v - pop.nodeU[z_ind_alt,y_ind_alt,x_ind_alt])**2, axis = -1)

      v_nn *= w
      
      np.add.at(pop.nodeT, (z_ind[0],y_ind[0],x_ind[0]), v_nn[0,0,0])
      np.add.at(pop.nodeT, (z_ind[0],y_ind[0],x_ind[1]), v_nn[0,0,1])
      np.add.at(pop.nodeT, (z_ind[0],y_ind[1],x_ind[0]), v_nn[0,1,0])
      np.add.at(pop.nodeT, (z_ind[0],y_ind[1],x_ind[1]), v_nn[0,1,1])
      np.add.at(pop.nodeT, (z_ind[1],y_ind[0],x_ind[0]), v_nn[1,0,0])
      np.add.at(pop.nodeT, (z_ind[1],y_ind[0],x_ind[1]), v_nn[1,0,1])
      np.add.at(pop.nodeT, (z_ind[1],y_ind[1],x_ind[0]), v_nn[1,1,0])
      np.add.at(pop.nodeT, (z_ind[1],y_ind[1],x_ind[1]), v_nn[1,1,1])
   
      pop.nodeT *= pop.m/const.k
   
def compute_rotated_current(pop, dims):
   # Computes current due to B-rotation
   # Current is stored at nodes to match E
   # This is accumulated directly to the nodes
   nodeJ = np.zeros(dims.dim_vector)
   
   (x_ind,y_ind,z_ind),(x_w,y_w,z_w) = CIC_weights_node(pop.r, dims)
   
   alpha_v = np.transpose(np.matvec(pop.alpha, pop.v))
   
   if dims.oneV is True:
      w = x_w.reshape(2,-1)
      
      # (nodal) 'rotated' current density
      np.add.at(nodeJ[:,:,:,0], (z_ind[0],y_ind[0],x_ind[0]), w[0]*alpha_v[0])
      np.add.at(nodeJ[:,:,:,0], (z_ind[0],y_ind[0],x_ind[1]), w[1]*alpha_v[0])
   else:
      w = z_w*y_w*x_w
      
      # (nodal) 'rotated' current density
      for nn in range(3):
         alpha_v_nn = alpha_v[nn].copy()
         
         np.add.at(nodeJ[:,:,:,nn], (z_ind[0],y_ind[0],x_ind[0]), w[0,0,0]*alpha_v_nn)
         np.add.at(nodeJ[:,:,:,nn], (z_ind[0],y_ind[0],x_ind[1]), w[0,0,1]*alpha_v_nn)
         np.add.at(nodeJ[:,:,:,nn], (z_ind[0],y_ind[1],x_ind[0]), w[0,1,0]*alpha_v_nn)
         np.add.at(nodeJ[:,:,:,nn], (z_ind[0],y_ind[1],x_ind[1]), w[0,1,1]*alpha_v_nn)
         np.add.at(nodeJ[:,:,:,nn], (z_ind[1],y_ind[0],x_ind[0]), w[1,0,0]*alpha_v_nn)
         np.add.at(nodeJ[:,:,:,nn], (z_ind[1],y_ind[0],x_ind[1]), w[1,0,1]*alpha_v_nn)
         np.add.at(nodeJ[:,:,:,nn], (z_ind[1],y_ind[1],x_ind[0]), w[1,1,0]*alpha_v_nn)
         np.add.at(nodeJ[:,:,:,nn], (z_ind[1],y_ind[1],x_ind[1]), w[1,1,1]*alpha_v_nn)

   nodeJ *= pop.q*pop.w/dims.dV

   return nodeJ

def compute_mass_matrices(pop, dims):
   # Compute mass matrices
   if dims.oneV:
      M = np.zeros((dims.Ncells_total,dims.Ncells_total,1,1))
      alpha = pop.alpha[:,0,0].reshape(-1,1,1)
   else:
      M = np.zeros((dims.Ncells_total,dims.Ncells_total,3,3))
      # alpha = np.transpose(pop.alpha, (1,2,0))
   
   (x_ind,y_ind,z_ind),(x_w,y_w,z_w) = CIC_weights_node(pop.r, dims, False)
   
   if dims.oneV is True:
      # x_w_outer = np.empty((pop.Np,2,2))
      # gu_outer(x_w, x_w, x_w_outer)
      x_w = x_w.reshape(2,-1,1,1)
      
      for xi,x_wi in zip(x_ind,x_w):
         for xj,x_wj in zip(x_ind,x_w):
            np.add.at(M, (xi,xj), x_wi*x_wj*alpha)
   else:
      # y_w = np.hstack((y_w0[:,np.newaxis],y_w1[:,np.newaxis]))
      # z_w = np.hstack((z_w0[:,np.newaxis],z_w1[:,np.newaxis]))

      # x_w_outer = np.empty((pop.Np,2,2))
      # y_w_outer = x_w_outer.copy()
      # z_w_outer = x_w_outer.copy()
      # gu_outer(x_w, x_w, x_w_outer)
      # gu_outer(y_w, y_w, y_w_outer)
      # gu_outer(z_w, z_w, z_w_outer)

      # x_w = np.stack((x_w0,x_w1)).reshape(1,1,2,-1)
      # y_w = np.stack((y_w0,y_w1)).reshape(1,2,1,-1)
      # z_w = np.stack((z_w0,z_w1)).reshape(2,1,1,-1)
      # w = z_w*y_w*x_w

      # ww = w[np.newaxis,np.newaxis,np.newaxis,...]*w[:,:,:,np.newaxis,np.newaxis,np.newaxis,...]

      # ind_sub_x = (z_ind0*dims.y_size + y_ind0)*dims.x_size + x_ind0
      # ind_sub_y = (z_ind0*dims.y_size + y_ind0)*dims.x_size + x_ind0

      # w_sub = (x_w0*x_w0*y_w0*y_w0*z_w0*z_w0)[-1,np.newaxis,np.newaxis] * alpha
      
      x_w = x_w.reshape(2,-1,1,1)
      y_w = y_w.reshape(2,-1,1,1)
      z_w = z_w.reshape(2,-1,1,1)

      y_ind *= dims.x_size
      z_ind *= dims.x_size*dims.y_size
      
      for xi,x_wi in zip(x_ind,x_w):
         for xj,x_wj in zip(x_ind,x_w):
            for yi,y_wi in zip(y_ind,y_w):
               for yj,y_wj in zip(y_ind,y_w):
                  for zi,z_wi in zip(z_ind,z_w):
                     for zj,z_wj in zip(z_ind,z_w):
                        np.add.at(M, (zi + yi + xi,zj + yj + xj),
                                  (x_wi*x_wj*y_wi*y_wj*z_wi*z_wj) * pop.alpha)
   M = np.transpose(M, (2,3,0,1))
   
   beta = dims.phi*(pop.q*dims.dt)/pop.m
   M *= beta * pop.q*pop.w/dims.dV
   
   return M

def compute_mass_matrices_alt(pop, dims):
   # Compute mass matrices
   if dims.oneV:
      M = np.zeros((dims.Ncells_total,dims.Ncells_total,1,1))
      alpha = pop.alpha[:,0,0].reshape(-1,1,1)
   else:
      M = np.zeros((dims.Ncells_total,dims.Ncells_total,3,3))
      # alpha = np.transpose(pop.alpha, (1,2,0))
   
   (x_ind,y_ind,z_ind),(x_w,y_w,z_w) = CIC_weights_node(pop.r, dims, False)
   
   if dims.oneV is True:
      # x_w_outer = np.empty((pop.Np,2,2))
      # gu_outer(x_w, x_w, x_w_outer)
      x_w = x_w.reshape(2,-1,1,1)
      
      for xi,x_wi in zip(x_ind,x_w):
         for xj,x_wj in zip(x_ind,x_w):
            np.add.at(M, (xi,xj), x_wi*x_wj*alpha)
   else:
      # y_w = np.hstack((y_w0[:,np.newaxis],y_w1[:,np.newaxis]))
      # z_w = np.hstack((z_w0[:,np.newaxis],z_w1[:,np.newaxis]))

      # x_w_outer = np.empty((pop.Np,2,2))
      # y_w_outer = x_w_outer.copy()
      # z_w_outer = x_w_outer.copy()
      # gu_outer(x_w, x_w, x_w_outer)
      # gu_outer(y_w, y_w, y_w_outer)
      # gu_outer(z_w, z_w, z_w_outer)

      # x_w = np.stack((x_w0,x_w1)).reshape(1,1,2,-1)
      # y_w = np.stack((y_w0,y_w1)).reshape(1,2,1,-1)
      # z_w = np.stack((z_w0,z_w1)).reshape(2,1,1,-1)
      # w = z_w*y_w*x_w

      # ww = w[np.newaxis,np.newaxis,np.newaxis,...]*w[:,:,:,np.newaxis,np.newaxis,np.newaxis,...]

      # ind_sub_x = (z_ind0*dims.y_size + y_ind0)*dims.x_size + x_ind0
      # ind_sub_y = (z_ind0*dims.y_size + y_ind0)*dims.x_size + x_ind0

      # w_sub = (x_w0*x_w0*y_w0*y_w0*z_w0*z_w0)[-1,np.newaxis,np.newaxis] * alpha
      
      x_w = x_w.reshape(2,-1,1,1)
      y_w = y_w.reshape(2,-1,1,1)
      z_w = z_w.reshape(2,-1,1,1)
      
      for xi,x_wi in zip(x_ind,x_w):
         for xj,x_wj in zip(x_ind,x_w):
            for yi,y_wi in zip(y_ind,y_w):
               for yj,y_wj in zip(y_ind,y_w):
                  for zi,z_wi in zip(z_ind,z_w):
                     for zj,z_wj in zip(z_ind,z_w):
                        rows = np.ravel_multi_index((zi,yi,xi),
                                                    (dims.z_size,
                                                     dims.y_size,
                                                     dims.x_size))

                        cols = np.ravel_multi_index((zj,yj,xj),
                                                    (dims.z_size,
                                                     dims.y_size,
                                                     dims.x_size))
                        
                        np.add.at(M, (rows,cols),
                                  (x_wi*x_wj*y_wi*y_wj*z_wi*z_wj) * pop.alpha)
   M = np.transpose(M, (2,3,0,1))
   
   beta = dims.phi*(pop.q*dims.dt)/pop.m
   M *= beta * pop.q*pop.w/dims.dV
   
   return M

# def compute_mass_matrices_coo(pop, dims):
#    # Compute mass matrices
#    if dims.oneV:
#       data_M = np.empty((1,1,64*pop.Np), dtype = np.float64)
#       rows_M = np.empty((1,1,64*pop.Np), dtype = np.int64)
#       cols_M = np.empty((1,1,64*pop.Np), dtype = np.int64)
      
#       alpha = pop.alpha[:,0,0].reshape(1,1,1,-1)
#    else:
#       data_M = np.empty((3,3,64*pop.Np), dtype = np.float64)
#       rows_M = np.empty((3,3,64*pop.Np), dtype = np.int64)
#       cols_M = np.empty((3,3,64*pop.Np), dtype = np.int64)

#       alpha = pop.alpha.transpose((1,2,0)).reshape(3,3,1,-1)
   
#    (x_ind,y_ind,z_ind),(x_w,y_w,z_w) = CIC_weights_node(pop.r, dims, False)
   
#    if dims.oneV is True:
#       x_w = x_w.reshape(1,1,2,-1)

#       x_wi = np.repeat(x_w, 2, axis = 2)
#       x_wj = np.tile(x_w, (1,1,2,1))
      
#       data_M = (x_wi*x_wj*alpha).reshape(1,1,-1)
#       rows_M = np.repeat(x_ind, 2, axis = 0).flatten()
#       cols_M = np.tile(x_ind, (2,1)).flatten()
#    else:
#       x_w = x_w.reshape(1,1,2,-1)
#       y_w = y_w.reshape(1,1,2,-1)
#       z_w = z_w.reshape(1,1,2,-1)
      
#       y_ind *= dims.x_size
#       z_ind *= dims.x_size*dims.y_size

#       x_wi = np.repeat(x_w, 32, axis = 2)
#       x_wj = np.tile(np.repeat(x_w, 16, axis = 2), (1,1,2,1))
#       y_wi = np.tile(np.repeat(y_w, 8, axis = 2), (1,1,4,1))
#       y_wj = np.tile(np.repeat(y_w, 4, axis = 2), (1,1,8,1))
#       z_wi = np.tile(np.repeat(z_w, 2, axis = 2), (1,1,16,1))
#       z_wj = np.tile(z_w, (1,1,32,1))

#       data_M = (x_wi*x_wj*y_wi*y_wj*z_wi*z_wj*alpha).reshape(3,3,-1)
      
#       xi = np.repeat(x_ind, 32, axis = 0).flatten()
#       xj = np.tile(np.repeat(x_ind, 16, axis = 0), (2,1)).flatten()
#       yi = np.tile(np.repeat(y_ind, 8, axis = 0), (4,1)).flatten()
#       yj = np.tile(np.repeat(y_ind, 4, axis = 0), (8,1)).flatten()
#       zi = np.tile(np.repeat(z_ind, 2, axis = 0), (16,1)).flatten()
#       zj = np.tile(z_ind, (32,1)).flatten()
      
#       rows_M = zi + yi + xi
#       cols_M = zj + yj + xj

#    beta = dims.phi*(pop.q*dims.dt)/pop.m
#    data_M *= beta * pop.q*pop.w/dims.dV
   
#    return data_M,rows_M,cols_M

def compute_mass_matrices_coo(pop, dims):
   # Compute mass matrices
   if dims.oneV:
      data_M = np.empty((4,pop.Np), dtype = np.float64)
      rows_M = np.empty((4,pop.Np), dtype = np.int64)
      cols_M = np.empty((4,pop.Np), dtype = np.int64)
      
      alpha = pop.alpha[:,0,0][np.newaxis,:,np.newaxis,np.newaxis]
   else:
      data_M = np.empty((64,3*3*pop.Np), dtype = np.float64)
      rows_M = np.empty((64,3*3*pop.Np), dtype = np.int64)
      cols_M = np.empty((64,3*3*pop.Np), dtype = np.int64)

      alpha = pop.alpha[np.newaxis,...]
   
   (x_ind,y_ind,z_ind),(x_w,y_w,z_w) = CIC_weights_node(pop.r, dims, False)
   
   if dims.oneV is True:
      for jj,(xi,x_wi) in enumerate(zip(x_ind,x_w)):
         for ii,(xj,x_wj) in enumerate(zip(x_ind,x_w)):
            ind = jj*2 + ii
            
            data_M[ind] = (x_wi*x_wj * alpha).flat
            
            rows_M[ind] = xi
            cols_M[ind] = xj
   else:
      x_w = x_w.reshape(2,-1,1,1)
      y_w = y_w.reshape(2,-1,1,1)
      z_w = z_w.reshape(2,-1,1,1)

      x_ind *= 3
      y_ind *= 3*dims.x_size
      z_ind *= 3*dims.x_size*dims.y_size
      
      for nn,(xi,x_wi) in enumerate(zip(x_ind,x_w)):
         for mm,(xj,x_wj) in enumerate(zip(x_ind,x_w)):
            for ll,(yi,y_wi) in enumerate(zip(y_ind,y_w)):
               for kk,(yj,y_wj) in enumerate(zip(y_ind,y_w)):
                  for jj,(zi,z_wi) in enumerate(zip(z_ind,z_w)):
                     for ii,(zj,z_wj) in enumerate(zip(z_ind,z_w)):
                        ind = ((((nn*2 + mm)*2 + ll)*2 + kk)*2 + jj)*2 + ii
                        
                        data_M[ind] = (x_wi*x_wj*y_wi*y_wj*z_wi*z_wj * alpha).flat
                        
                        row = zi + yi + xi
                        col = zj + yj + xj

                        row = (row[:,np.newaxis] + np.arange(3)[np.newaxis]).flat
                        row = np.repeat(row, 3)
                        
                        col = np.repeat(col, 3)
                        col = (col[:,np.newaxis] + np.arange(3)[np.newaxis]).flat
                        
                        rows_M[ind] = row
                        cols_M[ind] = col

   data_M = data_M.flatten()
   rows_M = data_M.flatten()
   cols_M = data_M.flatten()
   
   beta = dims.phi*(pop.q*dims.dt)/pop.m
   data_M *= beta * pop.q*pop.w/dims.dV
   
   return data_M,rows_M,cols_M

def compute_mass_matrices_coo_alt(pop, dims):
   # Compute mass matrices
   (x_ind,y_ind,z_ind),(x_w,y_w,z_w) = CIC_weights_node(pop.r, dims, False)
   
   if dims.oneV is True:
      alpha = pop.alpha[:,0,0][np.newaxis,:,np.newaxis,np.newaxis]
      
      x_w = x_w.reshape(2,-1)

      x_wi = np.repeat(x_w, 2, axis = 0)
      x_wj = np.tile(x_w, (2,1))
      
      data_M = ((x_wi*x_wj)[:,:,np.newaxis,np.newaxis]*alpha).flatten()
      
      rows_M = np.repeat(x_ind, 2, axis = 0).flatten()
      cols_M = np.tile(x_ind, (2,1)).flatten()
   else:
      alpha = pop.alpha[np.newaxis,...]
      
      x_w = x_w.reshape(2,-1,1,1)
      y_w = y_w.reshape(2,-1,1,1)
      z_w = z_w.reshape(2,-1,1,1)
      
      y_ind *= dims.x_size
      z_ind *= dims.x_size*dims.y_size
      
      x_wi = np.repeat(x_w, 2, axis = 0)
      x_wj = np.tile(x_w, (2,1,1,1))
      x_wij = x_wi*x_wj

      y_wi = np.repeat(y_w, 2, axis = 0)
      y_wj = np.tile(y_w, (2,1,1,1))
      y_wij = y_wi*y_wj

      z_wi = np.repeat(z_w, 2, axis = 0)
      z_wj = np.tile(z_w, (2,1,1,1))
      z_wij = z_wi*z_wj

      xyz_w = np.repeat(x_wij, 16, axis = 0) * np.tile(np.repeat(y_wij, 4, axis = 0), (4,1,1,1)) * np.tile(z_wij, (16,1,1,1))      
      
      data_M = (xyz_w*alpha).flatten()
      
      xi = np.repeat(x_ind, 32, axis = 0).flatten()
      xj = np.tile(np.repeat(x_ind, 16, axis = 0), (2,1)).flatten()
      yi = np.tile(np.repeat(y_ind, 8, axis = 0), (4,1)).flatten()
      yj = np.tile(np.repeat(y_ind, 4, axis = 0), (8,1)).flatten()
      zi = np.tile(np.repeat(z_ind, 2, axis = 0), (16,1)).flatten()
      zj = np.tile(z_ind, (32,1)).flatten()
      
      rows_M = zi + yi + xi
      cols_M = zj + yj + xj

      rows_M *= 3
      cols_M *= 3
      
      rows_M = (rows_M[:,np.newaxis] + np.arange(3)[np.newaxis]).flatten()
      rows_M = np.repeat(rows_M, 3)

      cols_M = np.repeat(cols_M, 3)
      cols_M = (cols_M[:,np.newaxis] + np.arange(3)[np.newaxis]).flatten()

   beta = dims.phi*(pop.q*dims.dt)/pop.m
   data_M *= beta * pop.q*pop.w/dims.dV
   
   return data_M,rows_M,cols_M

Pop_spec = [
   ("q", float64),
   ("m", float64),
   ("w", float64),
   ("r", float64[:,:]),
   ("v", float64[:,:]),
   ("Np", int64),
   ("cellRhoQ", float64[:,:,:]),
   ("cellJi", float64[:,:,:,:]),
   ("alpha", float64[:,:,:])
]

# @jitclass(Pop_spec)
class Pop_njit:
   def __init__(self, q, m, w, electron, Np, rng, dims, T = None, v = None):
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
      
      self.q = q
      self.m = m
      self.w = w
      
      if Np > 0:
         vth = math.sqrt(const.k*T/self.m)
         self.r,self.v = uniform_injector_njit(Np, vth, v, rng, dims)
         self.Np = self.r.shape[0]
         
      self.cellRhoQ,self.cellJi = accumulators_njit(self.r, self.v, self.q, self.w, dims)

@njit(cache = True, fastmath = True)
def uniform_injector_njit(Np, vth, bulk_v, rng, dims):
   # Inject Np particles between xmin and xmax, bulk velocity bulk_v, and thermal velocity vth

   r = np.zeros((Np*dims.Ncells_total,3), dtype = float64)
   v = np.zeros((Np*dims.Ncells_total,3), dtype = float64)

   for ii in range(dims.Ncells_total):
      zi,yi,xi = numba_unravel_index(ii, dims.dim_scalar)

      cell_x_min = dims.x_min + xi*dims.dx
      cell_x_max = dims.x_max - (dims.x_size - 1 - xi)*dims.dx
      cell_y_min = dims.y_min + yi*dims.dy
      cell_y_max = dims.y_max - (dims.y_size - 1 - yi)*dims.dy
      cell_z_min = dims.z_min + zi*dims.dz
      cell_z_max = dims.z_max - (dims.z_size - 1 - zi)*dims.dz
      
      r[ii*Np:(ii+1)*Np,0] = rng.uniform(cell_x_min,cell_x_max, Np)
      if dims.oneV is True:
         r[ii*Np:(ii+1)*Np,1] = (cell_y_max + cell_y_min)/2
         r[ii*Np:(ii+1)*Np,2] = (cell_z_max + cell_z_min)/2
      else:
         r[ii*Np:(ii+1)*Np,1] = rng.uniform(cell_y_min,cell_y_max, Np)
         r[ii*Np:(ii+1)*Np,2] = rng.uniform(cell_z_min,cell_z_max, Np)
      v[ii*Np:(ii+1)*Np,0] = rng.normal(bulk_v[0], vth, Np)
      if dims.oneV is False:
         v[ii*Np:(ii+1)*Np,1] = rng.normal(bulk_v[1], vth, Np)
         v[ii*Np:(ii+1)*Np,2] = rng.normal(bulk_v[2], vth, Np)

   return r,v

@njit(cache = True, fastmath = True)
def apply_boundaries_parts_njit(pop, dims):
   # Apply boundary conditions to particles
   # Periodic boundaries wrap locations
   # Non-periodic delete particles

   lims = ((dims.x_min,dims.x_max),(dims.y_min,dims.y_max),(dims.z_min,dims.z_max))
   sizes = (dims.x_size,dims.y_size,dims.z_size)
   dgrid = (dims.dx,dims.dy,dims.dz)

   del_list = np.repeat(False, pop.Np)

   for nn,(lim,Ng,dg,rep) in enumerate(zip(lims,sizes,dgrid,dims.period)):
      if rep:
         for ii,r in enumerate(pop.r):
            pop.r[ii,nn] = (r[nn] - lim[0] % lim[1] - lim[0]) + lim[0]
      else:
         for ii,r in enumerate(pop.r):
            del_list[ii] |= r[nn] < lim[0] | r[nn] >= lim[1]
   
   if any(del_list):
      removeParticle_njit(pop, del_list)

@njit(cache = True, fastmath = True)
def removeParticle_njit(pop, to_delete):
   # Delete particle from population
   to_keep = np.logical_not(to_delete)
   pop.r = pop.r[to_keep]
   pop.v = pop.v[to_keep]
   pop.alpha = pop.alpha[to_keep]
   pop.Np = np.sum(to_keep)

@njit(cache = True, fastmath = True)
def accumulators_njit(r, v, q, w, dims):
   # Accumulate macroparticles into cell-centred charge and current densities
   for r,v in zip(r,v):
      x_locs = round((r[0] - dims.x_min)/dims.dx)
      y_locs = round((r[1] - dims.y_min)/dims.dy)
      z_locs = round((r[2] - dims.z_min)/dims.dz)

      if dims.period[2]:
         x_locs = x_locs % dims.x_size
      if dims.period[1]:
         y_locs = y_locs % dims.y_size
      if dims.period[0]:
         z_locs = z_locs % dims.z_size

      cellRhoQ = np.zeros((dims.z_size,dims.y_size,dims.x_size), dtype = float64)
      cellJi = np.zeros((dims.z_size,dims.y_size,dims.x_size,3), dtype = float64)

      x0 = (x_locs - 1 + 0.5)*dims.dx + dims.x_min
      y0 = (y_locs - 1 + 0.5)*dims.dy + dims.y_min
      z0 = (z_locs - 1 + 0.5)*dims.dz + dims.z_min

      x_w1 = (r[0] - x0)/dims.dx
      y_w1 = (r[1] - y0)/dims.dy
      z_w1 = (r[2] - z0)/dims.dz

      if dims.period[2]:
         x_w1 = x_w1 % 1
      if dims.period[1]:
         y_w1 = y_w1 % 1
      if dims.period[0]:
         z_w1 = z_w1 % 1

      x_w0 = 1 - x_w1
      y_w0 = 1 - y_w1
      z_w0 = 1 - z_w1

      x_ind_l = x_locs - 1
      if dims.period[2]:
         x_ind_l = x_ind_l % dims.x_size
      else:
         x_ind_l = min(max(x_ind_l, 0), dims.x_size - 1)
      x_ind_r = x_locs

      y_ind_l = y_locs - 1
      if dims.period[1]:
         y_ind_l = y_ind_l % dims.y_size
      else:
         y_ind_l = min(max(y_ind_l, 0), dims.y_size - 1)
      y_ind_r = y_locs

      z_ind_l = z_locs - 1
      if dims.period[0]:
         z_ind_l = z_ind_l % dims.z_size
      else:
         z_ind_l = min(max(z_ind_l, 0), dims.z_size - 1)
      z_ind_r = z_locs

      # CIC weight factors
      x_w = np.stack((x_w0,x_w1)).reshape(1,1,2)
      y_w = np.stack((y_w0,y_w1)).reshape(1,2,1)
      z_w = np.stack((z_w0,z_w1)).reshape(2,1,1)
      w = z_w*y_w*x_w

      # Charge density
      cellRhoQ[z_ind_l,y_ind_l,x_ind_l] += w[0,0,0]
      cellRhoQ[z_ind_l,y_ind_l,x_ind_r] += w[0,0,1]
      cellRhoQ[z_ind_l,y_ind_r,x_ind_l] += w[0,1,0]
      cellRhoQ[z_ind_l,y_ind_r,x_ind_r] += w[0,1,1]
      cellRhoQ[z_ind_r,y_ind_l,x_ind_l] += w[1,0,0]
      cellRhoQ[z_ind_r,y_ind_l,x_ind_r] += w[1,0,1]
      cellRhoQ[z_ind_r,y_ind_r,x_ind_l] += w[1,1,0]
      cellRhoQ[z_ind_r,y_ind_r,x_ind_r] += w[1,1,1]

      # (cell-centred) current density
      for nn in range(3):
         v_nn = v[nn].copy()
         
         cellJi[z_ind_l,y_ind_l,x_ind_l,nn] += w[0,0,0]*v_nn
         cellJi[z_ind_l,y_ind_l,x_ind_r,nn] += w[0,0,1]*v_nn
         cellJi[z_ind_l,y_ind_r,x_ind_l,nn] += w[0,1,0]*v_nn
         cellJi[z_ind_l,y_ind_r,x_ind_r,nn] += w[0,1,1]*v_nn
         cellJi[z_ind_r,y_ind_l,x_ind_l,nn] += w[1,0,0]*v_nn
         cellJi[z_ind_r,y_ind_l,x_ind_r,nn] += w[1,0,1]*v_nn
         cellJi[z_ind_r,y_ind_r,x_ind_l,nn] += w[1,1,0]*v_nn
         cellJi[z_ind_r,y_ind_r,x_ind_r,nn] += w[1,1,1]*v_nn
   
   cellRhoQ *= q*w/dims.dV
   cellJi *= q*w/dims.dV
   
   return cellRhoQ,cellJi

@njit(cache = True, fastmath = True)
def compute_alpha_njit(pop, faceB, dims):
   # Compute alpha matrix for all particles in population
   rB = face2r_njit(faceB, pop.r, dims)

   beta = dims.phi*(pop.q*dims.dt)/pop.m

   bx,by,bz = split_axis(rB*beta, axis = 1)

   if dims.oneV is True:
      pop.alpha = np.zeros((pop.Np,3,3))

      pop.alpha[:,0,0] = 1
   else:
      factor = 1/(1 + bx**2 + by**2 + bz**2)
      
      pop.alpha = np.zeros((pop.Np,3,3))
      
      pop.alpha[:,0,0] = bx*bx + 1
      pop.alpha[:,0,1] = bx*by + bz
      pop.alpha[:,0,2] = bx*bz - by
      pop.alpha[:,1,0] = bx*by - bz
      pop.alpha[:,1,1] = by*by + 1
      pop.alpha[:,1,2] = by*bz + bx
      pop.alpha[:,2,0] = bx*bz + by
      pop.alpha[:,2,1] = by*bz - bx
      pop.alpha[:,2,2] = bz*bz + 1

      pop.alpha *= factor.reshape(pop.Np,1,1)

@njit(cache = True, fastmath = True)
def moveParticles_njit(pop, dstep, dims):
   # Pushes particle positions by time dstep
   pop.r += pop.v * dstep

   apply_boundaries_parts_njit(pop, dims)

@njit(cache = True, fastmath = True)
def Lorentz_njit(r, v, alpha, q, m, midNodeE, dims):
   # Accelerate particles via Lorentz force
   rE = node2r_njit(midNodeE, r, dims)

   beta = dims.phi*(q*dims.dt)/m

   new_v = np.zeros(v.shape, dtype = float64)

   for ii,(r,v,alpha,E) in enumerate(zip(r,v,alpha,rE)):
      new_v[ii] = 2*alpha@(v + beta*E) - v

   return new_v

@njit(cache = True, fastmath = True)
def nodeU_njit(pop, dims):
   # Computes bulk velocity at nodes
   nodeU = np.zeros(dims.dim_vector)
   node_w = np.zeros(dims.dim_vector)

   for r,v in zip(pop.r,pop.v):
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
      if dims.period[2]:
         x_ind1 = np.mod(x_ind1, dims.x_size)
      else:
         x_ind1 = np.clip(x_ind1, 0, dims.x_size - 1)

      y_ind0 = y_locs
      y_ind1 = y_locs + 1
      if dims.period[1]:
         y_ind1 = np.mod(y_ind1, dims.y_size)
      else:
         y_ind1 = np.clip(y_ind1, 0, dims.y_size - 1)

      z_ind0 = z_locs
      z_ind1 = z_locs + 1
      if dims.period[0]:
         z_ind1 = np.mod(z_ind1, dims.z_size)
      else:
         z_ind1 = np.clip(z_ind1, 0, dims.z_size - 1)

      if dims.oneV is True:
         for nn in range(3):
            v_nn = v[nn].copy()
            
            nodeU[z_ind0,y_ind0,x_ind0,nn] += x_w0*v_nn
            nodeU[z_ind0,y_ind0,x_ind1,nn] += x_w1*v_nn

         node_w[z_ind0,y_ind0,x_ind0] += x_w0
         node_w[z_ind0,y_ind0,x_ind1] += x_w1
      else:
         x_w = np.stack((x_w0,x_w1)).reshape(1,1,2)
         y_w = np.stack((y_w0,y_w1)).reshape(1,2,1)
         z_w = np.stack((z_w0,z_w1)).reshape(2,1,1)
         w = z_w*y_w*x_w

         for nn in range(3):
            v_nn = v[nn].copy()
            
            nodeU[z_ind0,y_ind0,x_ind0,nn] += w[0,0,0]*v_nn
            nodeU[z_ind0,y_ind0,x_ind1,nn] += w[0,0,1]*v_nn
            nodeU[z_ind0,y_ind1,x_ind0,nn] += w[0,1,0]*v_nn
            nodeU[z_ind0,y_ind1,x_ind1,nn] += w[0,1,1]*v_nn
            nodeU[z_ind1,y_ind0,x_ind0,nn] += w[1,0,0]*v_nn
            nodeU[z_ind1,y_ind0,x_ind1,nn] += w[1,0,1]*v_nn
            nodeU[z_ind1,y_ind1,x_ind0,nn] += w[1,1,0]*v_nn
            nodeU[z_ind1,y_ind1,x_ind1,nn] += w[1,1,1]*v_nn
         
         node_w[z_ind0,y_ind0,x_ind0] += w[0,0,0]
         node_w[z_ind0,y_ind0,x_ind1] += w[0,0,1]
         node_w[z_ind0,y_ind1,x_ind0] += w[0,1,0]
         node_w[z_ind0,y_ind1,x_ind1] += w[0,1,1]
         node_w[z_ind1,y_ind0,x_ind0] += w[1,0,0]
         node_w[z_ind1,y_ind0,x_ind1] += w[1,0,1]
         node_w[z_ind1,y_ind1,x_ind0] += w[1,1,0]
         node_w[z_ind1,y_ind1,x_ind1] += w[1,1,1]

   nodeU /= node_w

   return nodeU

# @njit(cache = True, fastmath = True)
# def compute_rotated_current_njit(pop, dims):
#    # Computes current due to B-rotation
#    # Current is stored at nodes to match E
#    # This requires an interpolation after accumulating to cells
#    cellJ = np.zeros(dims.dim_vector)
#    
#    for r,v,alpha in zip(pop.r,pop.v,pop.alpha):
#       x_locs = np.round((r[0] - dims.x_min)/dims.dx).astype(int)
#       y_locs = np.round((r[1] - dims.y_min)/dims.dy).astype(int)
#       z_locs = np.round((r[2] - dims.z_min)/dims.dz).astype(int)
#       
#       if dims.period[2]:
#          x_locs = np.mod(x_locs, dims.x_size)
#       if dims.period[1]:
#          y_locs = np.mod(y_locs, dims.y_size)
#       if dims.period[0]:
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
#       if dims.period[2]:
#          x_w1 = np.mod(x_w1, 1)
#       if dims.period[1]:
#          y_w1 = np.mod(y_w1, 1)
#       if dims.period[0]:
#          z_w1 = np.mod(z_w1, 1)
#       
#       x_w0 = 1 - x_w1
#       y_w0 = 1 - y_w1
#       z_w0 = 1 - z_w1
#       
#       x_ind0 = x_locs - 1
#       if dims.period[2]:
#          x_ind0 = np.mod(x_ind0, dims.x_size)
#       else:
#          x_ind0 = np.clip(x_ind0, 0, dims.x_size - 1)
#       x_ind1 = x_locs
#       
#       y_ind0 = y_locs - 1
#       if dims.period[1]:
#          y_ind0 = np.mod(y_ind0, dims.y_size)
#       else:
#          y_ind0 = np.clip(y_ind0, 0, dims.y_size - 1)
#       y_ind1 = y_locs
#       
#       z_ind0 = z_locs - 1
#       if dims.period[0]:
#          z_ind0 = np.mod(z_ind0, dims.z_size)
#       else:
#          z_ind0 = np.clip(z_ind0, 0, dims.z_size - 1)
#       z_ind1 = z_locs
#       
#       if dims.oneV is True:
#          alpha_v = alpha@v
#          
#          # (cell-centred) current density
#          cellJ[z_ind0,y_ind0,x_ind0,:] += x_w0*alpha_v
#          cellJ[z_ind0,y_ind0,x_ind1,:] += x_w1*alpha_v
#       else:
#          x_w = np.stack((x_w0,x_w1)).reshape(1,1,2)
#          y_w = np.stack((y_w0,y_w1)).reshape(1,2,1)
#          z_w = np.stack((z_w0,z_w1)).reshape(2,1,1)
#          w = z_w*y_w*x_w
#          
#          alpha_v = alpha@v
#          
#          # (cell-centred) current density
#          for nn in range(3):
#             alpha_v_nn = alpha_v[nn].copy()
#             
#             cellJ[z_ind0,y_ind0,x_ind0,nn] += w[0,0,0]*alpha_v_nn
#             cellJ[z_ind0,y_ind0,x_ind1,nn] += w[0,0,1]*alpha_v_nn
#             cellJ[z_ind0,y_ind1,x_ind0,nn] += w[0,1,0]*alpha_v_nn
#             cellJ[z_ind0,y_ind1,x_ind1,nn] += w[0,1,1]*alpha_v_nn
#             cellJ[z_ind1,y_ind0,x_ind0,nn] += w[1,0,0]*alpha_v_nn
#             cellJ[z_ind1,y_ind0,x_ind1,nn] += w[1,0,1]*alpha_v_nn
#             cellJ[z_ind1,y_ind1,x_ind0,nn] += w[1,1,0]*alpha_v_nn
#             cellJ[z_ind1,y_ind1,x_ind1,nn] += w[1,1,1]*alpha_v_nn
#          
#    cellJ *= pop.q*pop.w/dims.dV
#    
#    nodeJ = cell2node(cellJ, dims)
#    
#    return nodeJ

@njit(cache = True, fastmath = True)
def compute_rotated_current_njit(pop_r, pop_v, pop_alpha, q, w, dims):
   # Computes current due to B-rotation
   # Current is stored at nodes to match E
   # This is accumulated directly to the nodes
   nodeJ = np.zeros(dims.dim_vector)
   
   for r,v,alpha in zip(pop_r,pop_v,pop_alpha):
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
      if dims.period[2]:
         x_ind1 = np.mod(x_ind1, dims.x_size)
      else:
         x_ind1 = np.clip(x_ind1, 0, dims.x_size - 1)

      y_ind0 = y_locs
      y_ind1 = y_locs + 1
      if dims.period[1]:
         y_ind1 = np.mod(y_ind1, dims.y_size)
      else:
         y_ind1 = np.clip(y_ind1, 0, dims.y_size - 1)

      z_ind0 = z_locs
      z_ind1 = z_locs + 1
      if dims.period[0]:
         z_ind1 = np.mod(z_ind1, dims.z_size)
      else:
         z_ind1 = np.clip(z_ind1, 0, dims.z_size - 1)

      if dims.oneV is True:
         alpha_v = alpha@v

         # (cell-centred) current density
         nodeJ[z_ind0,y_ind0,x_ind0,:] += x_w0*alpha_v
         nodeJ[z_ind0,y_ind0,x_ind1,:] += x_w1*alpha_v
      else:
         x_w = np.stack((x_w0,x_w1)).reshape(1,1,2,-1)
         y_w = np.stack((y_w0,y_w1)).reshape(1,2,1,-1)
         z_w = np.stack((z_w0,z_w1)).reshape(2,1,1,-1)
         w = z_w*y_w*x_w
         
         alpha_v = alpha@v

         # (cell-centred) current density
         for nn in range(3):
            alpha_v_nn = alpha_v[nn].copy()
            
            nodeJ[z_ind0,y_ind0,x_ind0,nn] += w[0,0,0]*alpha_v_nn
            nodeJ[z_ind0,y_ind0,x_ind1,nn] += w[0,0,1]*alpha_v_nn
            nodeJ[z_ind0,y_ind1,x_ind0,nn] += w[0,1,0]*alpha_v_nn
            nodeJ[z_ind0,y_ind1,x_ind1,nn] += w[0,1,1]*alpha_v_nn
            nodeJ[z_ind1,y_ind0,x_ind0,nn] += w[1,0,0]*alpha_v_nn
            nodeJ[z_ind1,y_ind0,x_ind1,nn] += w[1,0,1]*alpha_v_nn
            nodeJ[z_ind1,y_ind1,x_ind0,nn] += w[1,1,0]*alpha_v_nn
            nodeJ[z_ind1,y_ind1,x_ind1,nn] += w[1,1,1]*alpha_v_nn
            
   nodeJ *= q*w/dims.dV

   return nodeJ

@njit(cache = True, fastmath = True)
def compute_mass_matrices_njit(pop_r, pop_alpha, m, q, w, dims):
   # Compute mass matrices
   if dims.oneV:
      M = np.zeros((1,1,dims.Ncells_total,dims.Ncells_total))
   else:
      M = np.zeros((3,3,dims.Ncells_total,dims.Ncells_total))

   for r,alpha in zip(pop_r,pop_alpha):
      x_locs = math.floor((r[0] - dims.x_min)/dims.dx)
      y_locs = math.floor((r[1] - dims.y_min)/dims.dy)
      z_locs = math.floor((r[2] - dims.z_min)/dims.dz)

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
      if dims.period[2]:
         x_ind1 = x_ind1 % dims.x_size
      else:
         x_ind1 = min(max(x_ind1, 0), dims.x_size - 1)

      y_ind0 = y_locs
      y_ind1 = y_locs + 1
      if dims.period[1]:
         y_ind1 = y_ind1 % dims.y_size
      else:
         y_ind1 = min(max(y_ind1, 0), dims.y_size - 1)

      z_ind0 = z_locs
      z_ind1 = z_locs + 1
      if dims.period[0]:
         z_ind1 = z_ind1 % dims.z_size
      else:
         z_ind1 = min(max(z_ind1, 0), dims.z_size - 1)

      y_ind0 *= dims.x_size
      y_ind1 *= dims.x_size
      z_ind0 *= dims.y_size*dims.x_size
      z_ind1 *= dims.y_size*dims.x_size
         
      x_ind = (x_ind0, x_ind1)
      y_ind = (y_ind0, y_ind1)
      z_ind = (z_ind0, z_ind1)

      x_w = (x_w0,x_w1)
      y_w = (y_w0,y_w1)
      z_w = (z_w0,z_w1)
      
      if dims.oneV is True:
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
                                 M[ii,jj,zi+yi+xi,zj+yj+xj] += x_wi*x_wj*y_wi*y_wj*z_wi*z_wj * alpha[ii,jj]

   beta = dims.phi*(q*dims.dt)/m
   M *= beta * q*w/dims.dV

   return M

@njit(cache = True, fastmath = True)
def compute_mass_matrices_coo_njit_alt(pop_r, pop_alpha, m, q, w, dims):
   # Compute mass matrices as coo matrices
   if dims.oneV:
      data_M = np.empty((pop.Np,4,1), dtype = np.float64)
      rows_M = np.empty((pop.Np,4,1), dtype = np.int64)
      cols_M = np.empty((pop.Np,4,1), dtype = np.int64)
   else:
      data_M = np.empty((dims.Ncells_total,64,3,3), dtype = np.float64)
      rows_M = np.empty((dims.Ncells_total,64,3,3), dtype = np.int64)
      cols_M = np.empty((dims.Ncells_total,64,3,3), dtype = np.int64)
   
   for pp in range(pop.Np):
      r = pop_r[pp]
      index,weights = CIC_weights_node_njit(r, dims)
      if dims.oneV:
         alpha = pop_alpha[pp,0,0]
         
         for jj in range(2):
            xi = index[0,jj]
            x_wi = weights[0,jj]
            for ii in range(2):
               xj = index[0,ii]
               x_wj = weights[0,ii]
               
               ind = jj*2 + ii

               data_M[pp,ind] = x_wi*x_wj * alpha
               
               rows_M[pp,ind] = xi
               cols_M[pp,ind] = xj
      else:
         alpha = pop_alpha[pp].flatten()
         
         x_ind = index[0]*3
         y_ind = index[1]*3*dims.x_size
         z_ind = index[2]*3*dims.x_size*dims.y_size
         
         for nn in range(2):
            xi = x_ind[nn]
            x_wi = weights[0][nn]
            for mm in range(2):
               xj = x_ind[mm]
               x_wj = weights[0][mm]
               for ll in range(2):
                  yi = y_ind[ll]
                  y_wi = weights[1][ll]
                  for kk in range(2):
                     yj = y_ind[kk]
                     y_wj = weights[1][kk]
                     for jj in range(2):
                        zi = z_ind[jj]
                        z_wi = weights[2][jj]
                        for ii in range(2):
                           zj = z_ind[ii]
                           z_wj = weights[2][ii]
                           
                           ind = ((((nn*2 + mm)*2 + ll)*2 + kk)*2 + jj)*2 + ii
                           
                           data_M[pp,ind] = x_wi*x_wj*y_wi*y_wj*z_wi*z_wj * alpha
                           
                           row = zi + yi + xi
                           col = zj + yj + xj

                           rows_M[pp,ind] = np.array(
                              [row+0,row+0,row+0,
                               row+1,row+1,row+1,
                               row+2,row+2,row+2])
                           
                           cols_M[pp,ind] = np.array(
                              [col+0,col+1,col+2,
                               col+0,col+1,col+2,
                               col+0,col+1,col+2])
         
   data_M = data_M.flatten()
   rows_M = rows_M.flatten()
   cols_M = cols_M.flatten()
                           
   beta = dims.phi*(q*dims.dt)/m
   data_M *= beta * q*w/dims.dV

   return data_M,rows_M,cols_M

@ftools.lru_cache(maxsize = 128)
@njit(cache = True, fastmath = True)
def compute_mass_matrices_coo_njit_rowCol(dims):
   # Returns row and column indices for mass matrices coo matrices
   if dims.oneV:
      rows_M = np.empty((dims.Ncells_total,2,2,1,1), dtype = np.int64)
      cols_M = np.empty((dims.Ncells_total,2,2,1,1), dtype = np.int64)
   else:
      rows_M = np.empty((dims.Ncells_total,8,8,3,3), dtype = np.int64)
      cols_M = np.empty((dims.Ncells_total,8,8,3,3), dtype = np.int64)
   
   if dims.oneV:
      for xi in range(dims.x_size):
         cellid = xi
         
         cells = np.array((xi,xi+1))
         
         if dims.period[2]:
            cells[1] = cells[1]%dims.x_size
         else:
            cells[1] = min(max(cells[1], 0), dims.x_size - 1)

         for ii in range(2):
            cell_i = cells[ii]
            for jj in range(2):
               cell_j = cells[jj]
               rows_M[cellid,ii,jj,0,0] = cell_i
               cols_M[cellid,ii,jj,0,0] = cell_j
   else:
      relative_step = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1],
                                [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
      for zi in range(dims.z_size):
         for yi in range(dims.y_size):
            for xi in range(dims.x_size):
               cellid = (zi*dims.y_size + yi)*dims.x_size + xi
               
               cells = get_index_njit(np.array((zi,yi,xi), dtype = np.int64), relative_step, dims)
               cells = ((cells[:,0]*dims.y_size + cells[:,1])*dims.x_size + cells[:,2])*3
               
               for ll in range(8):
                  for kk in range(8):
                     for jj in range(3):
                        for ii in range(3):
                           rows_M[cellid,ll,kk,jj,ii] = cells[ll] + ii
                           cols_M[cellid,ll,kk,jj,ii] = cells[kk] + jj
   
   rows_M = rows_M.flatten()
   cols_M = cols_M.flatten()
   
   return rows_M,cols_M   

@njit(cache = True, fastmath = True)
def compute_mass_matrices_coo_njit(pop_r, pop_alpha, m, q, w, dims):
   # Compute mass matrices as coo matrices
   if dims.oneV:
      data_M = np.zeros((dims.Ncells_total,2,2,1,1), dtype = np.float64)
   else:
      data_M = np.zeros((dims.Ncells_total,8,8,3,3), dtype = np.float64)
   
   for pp in range(pop_r.shape[0]):
      r = pop_r[pp]
      index,weights = CIC_weights_node_njit(r, dims)
      
      if dims.oneV:
         alpha = pop_alpha[pp,0,0]

         weights = weights[0]

         cellid = index[0,0]

         for ll in range(2):
            for kk in range(2):
               data_M[cellid,ll,kk,0,0] += weights[ll]*weights[kk]*alpha
         
      else:
         alpha = pop_alpha[pp]
         
         multi_weights = weights[0].reshape(1,1,2)*weights[1].reshape(1,2,1)*weights[2].reshape(2,1,1)

         multi_weights = multi_weights.flatten()
         
         cellid = (index[2,0]*dims.y_size + index[1,0])*dims.x_size + index[0,0]

         for ll in range(8):
            for kk in range(8):
               for jj in range(3):
                  for ii in range(3):
                     data_M[cellid,ll,kk,jj,ii] += multi_weights[ll]*multi_weights[kk]*alpha[jj,ii]
   
   data_M = data_M.flatten()
                           
   beta = dims.phi*(q*dims.dt)/m
   data_M *= beta * q*w/dims.dV

   return data_M

@ftools.lru_cache(maxsize = 128)
@njit(cache = True, fastmath = True)
def compute_mass_matrices_coo_njit_alt2_rowCol(dims):
   # Returns row and column indices for mass matrices coo matrices alt2
   if dims.oneV:
      rows_M = np.empty((dims.Ncells_total,3,1,1), dtype = np.int64)
      cols_M = np.empty((dims.Ncells_total,3,1,1), dtype = np.int64)
   else:
      rows_M = np.empty((dims.Ncells_total,27,3,3), dtype = np.int64)
      cols_M = np.empty((dims.Ncells_total,27,3,3), dtype = np.int64)
   
   if dims.oneV:
      for xi in range(dims.x_size):
         nodeid = xi
         
         relative_step = np.array([[0,0,-1], [0,0,0], [0,0,1]])

         nodes = np.array((xi-1,xi,xi+1))

         if dims.period[2]:
            nodes[0] %= dims.x_size
            nodes[2] %= dims.x_size
         else:
            nodes[0] = min(max(nodes[0], 0), dims.x_size - 1)
            nodes[2] = min(max(nodes[2], 0), dims.x_size - 1)
            
         for kk in range(3):
            rows_M[nodeid,kk,0,0] = nodeid
            cols_M[nodeid,kk,0,0] = nodes[kk]
   else:
      for zi in range(dims.z_size):
         for yi in range(dims.y_size):
            for xi in range(dims.x_size):
               nodeid = (zi*dims.y_size + yi)*dims.x_size + xi
               
               relative_step = np.array([[-1,-1,-1], [-1,-1,0], [-1,-1,1],
                                         [-1,0,-1], [-1,0,0], [-1,0,1],
                                         [-1,1,-1], [-1,1,0], [-1,1,1],
                                         [0,-1,-1], [0,-1,0], [0,-1,1],
                                         [0,0,-1], [0,0,0], [0,0,1],
                                         [0,1,-1], [0,1,0], [0,1,1],
                                         [1,-1,-1], [1,-1,0], [1,-1,1],
                                         [1,0,-1], [1,0,0], [1,0,1],
                                         [1,1,-1], [1,1,0], [1,1,1]])
               
               nodes = get_index_njit(np.array((zi,yi,xi), dtype = np.int64), relative_step, dims)
               nodes = (nodes[:,0]*dims.y_size + nodes[:,1])*dims.x_size + nodes[:,2]
               
               for kk in range(27):
                  for jj in range(3):
                     for ii in range(3):
                        rows_M[nodeid,kk,jj,ii] = 3*nodeid + ii
                        cols_M[nodeid,kk,jj,ii] = 3*nodes[kk] + jj
   
   rows_M = rows_M.flatten()
   cols_M = cols_M.flatten()
   
   return rows_M,cols_M

@njit(cache = True, fastmath = True)
def compute_mass_matrices_coo_njit_alt2(pop_r, pop_alpha, pop_cids, m, q, w, dims):
   # Compute mass matrices as coo matrices
   if dims.oneV:
      data_M = np.zeros((dims.Ncells_total,3,1,1), dtype = np.float64)
   else:
      data_M = np.zeros((dims.Ncells_total,27,3,3), dtype = np.float64)
   
   for pp in range(pop_r.shape[0]):
      r = pop_r[pp]
      index,weights = CIC_weights_node_njit(r, dims)
      
      if dims.oneV:
         alpha = pop_alpha[pp,0,0]
         cell_loc = pop_cids[pp]
         
         weights = weights[0]
         
         relative_step = np.array([[0,0,0], [0,0,1]])

         nodeids = np.array([cell_loc, cell_loc+1])

         if dims.period[2]:
            nodeids[1] %= dims.x_size
         else:
            nodeids[1] = min(max(nodeids[1], 0), dims.x_size - 1)
         
         for ll in range(2):
            for kk in range(2):
               tmp = kk - ll + 1
               data_M[nodeids[ll],tmp,0,0] += weights[ll]*weights[kk]*alpha
      else:
         alpha = pop_alpha[pp]
         cellid = pop_cids[pp]
         cell_loc = np.array([int(cellid/(dims.y_size*dims.x_size)),
                              int(cellid/dims.x_size)%dims.y_size,
                              cellid%dims.x_size])
         
         multi_weights = weights[0].reshape(1,1,2)*weights[1].reshape(1,2,1)*weights[2].reshape(2,1,1)

         multi_weights = multi_weights.flatten()
         
         relative_step = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1],
                                   [1,0,0], [1,0,1], [1,1,0], [1,1,1]], dtype = np.int64)
         
         nodeids = get_index_njit(cell_loc, relative_step, dims)
         nodeids = (nodeids[:,0]*dims.y_size + nodeids[:,1])*dims.x_size + nodeids[:,2]

         for ll in range(8):
            for kk in range(8):
               tmp = relative_step[kk] - relative_step[ll] + 1
               tmp = (tmp[0]*3 + tmp[1])*3 + tmp[2]
               for jj in range(3):
                  for ii in range(3):
                     data_M[nodeids[ll],tmp,jj,ii] += multi_weights[ll]*multi_weights[kk]*alpha[jj,ii]
         
   data_M = data_M.flatten()
                           
   beta = dims.phi*(q*dims.dt)/m
   data_M *= beta * q*w/dims.dV

   return data_M
