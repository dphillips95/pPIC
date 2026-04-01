# Module of Population class and population functions, e.g. accumulators, alpha, mass matrices etc.

import math
import numpy as np
import scipy as sp
import scipy.constants as const
from numba import njit,int64,float64,guvectorize
from numba.experimental import jitclass

from interpolators import face2r,cell2r,node2r,face2r_njit,cell2r_njit,node2r_njit
from indexers import split_axis,numba_unravel_index,CIC_weights_node,CIC_weights_cell,numba_clip

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
      
      if Np > 0:
         if dims.oneV:
            vth = math.sqrt(const.k*T/self.m)
         else:
            vth = math.sqrt(3*const.k*T/self.m)
         uniform_injector(self, Np, vth, v, rng, dims)
      
      accumulators(self, dims)
      calcNodeData(self, dims)
      
      self.static = static
      
def uniform_injector(pop, Np, vth, v, rng, dims):
   # Inject Np particles between xmin and xmax, bulk velocity v, and thermal velocity vth
   
   ID = np.zeros((Np*dims.Ncells_total), dtype = np.int64)
   r_ins = np.zeros((Np*dims.Ncells_total,3), dtype = np.float64)
   v_ins = np.zeros((Np*dims.Ncells_total,3), dtype = np.float64)

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

   if pop.ID.size == 0:
      ID_max = -1
   else:
      ID_max = np.max(pop.ID)
   
   pop.ID = np.concatenate((pop.ID,ID + ID_max + 1))
   pop.r = np.concatenate((pop.r,r_ins))
   pop.v = np.concatenate((pop.v,v_ins))
   
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

   if any(del_list):
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

def moveParticles(pop, dstep, dims):
   # Pushes particle positions by time dstep
   pop.r += pop.v * dstep
   
   apply_boundaries_parts(pop, dims)

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
   # Computes bulk velocity and density at nodes
   pop.nodeU = np.zeros(dims.dim_vector)
   pop.nodeN = np.zeros(dims.dim_scalar)

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
      data_M = np.empty((1,1,64*pop.Np), dtype = np.float64)
      rows_M = np.empty((1,1,64*pop.Np), dtype = np.int64)
      cols_M = np.empty((1,1,64*pop.Np), dtype = np.int64)
      
      alpha = pop.alpha[:,0,0].reshape(1,1,1,-1)
   else:
      data_M = np.empty((3*3*64*pop.Np), dtype = np.float64)
      rows_M = np.empty((3*3*64*pop.Np), dtype = np.int64)
      cols_M = np.empty((3*3*64*pop.Np), dtype = np.int64)

      alpha = pop.alpha
   
   (x_ind,y_ind,z_ind),(x_w,y_w,z_w) = CIC_weights_node(pop.r, dims, False)
   
   if dims.oneV is True:
      x_w = x_w.reshape(1,1,2,-1)

      x_wi = np.repeat(x_w, 2, axis = 2)
      x_wj = np.tile(x_w, (1,1,2,1))
      
      data_M = (x_wi*x_wj*alpha).reshape(1,1,-1)
      rows_M = np.repeat(x_ind, 2, axis = 0).flatten()
      cols_M = np.tile(x_ind, (2,1)).flatten()
   else:
      x_w = x_w.reshape(2,-1)
      y_w = y_w.reshape(2,-1)
      z_w = z_w.reshape(2,-1)
      
      y_ind *= dims.x_size
      z_ind *= dims.x_size*dims.y_size

      x_wi = np.repeat(x_w, 32, axis = 0)
      x_wj = np.tile(np.repeat(x_w, 16, axis = 0), (2,1))
      y_wi = np.tile(np.repeat(y_w, 8, axis = 0), (4,1))
      y_wj = np.tile(np.repeat(y_w, 4, axis = 0), (8,1))
      z_wi = np.tile(np.repeat(z_w, 2, axis = 0), (16,1))
      z_wj = np.tile(z_w, (32,1))
      
      data_M = ((x_wi*x_wj*y_wi*y_wj*z_wi*z_wj)[:,:,np.newaxis,np.newaxis]*alpha[np.newaxis,...]).flatten()
      
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

def compute_mass_matrices_coo_alt(pop, dims):
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
def compute_mass_matrices_coo_njit(pop_r, pop_alpha, m, q, w, dims):
   # Compute mass matrices as coo matrices
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
