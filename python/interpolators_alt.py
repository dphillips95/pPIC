# Module of alternative interpolators
import math
import numpy as np
import functools as ftools
import scipy.constants as const
from numba import njit
# from numba import int64,float64
from numpy import int64,float64

from indexers import arr_shift,arr_diff,split_axis,arr_shift_njit,arr_diff_njit,CIC_weights_node,CIC_weights_cell,get_index,get_index_njit,numba_ravel_multi_index

def face2cloud(face_data, r, dims):
   # Interpolate face data to cloud located at arbitrary position(s) r
   # Data is assumed to be vector data, face data only stores one component each
   r = np.array(r).reshape(-1,3)
   r_data = np.zeros(r.shape)
   
   xface_data,yface_data,zface_data = split_axis(face_data, axis = 3)

   (x_ind,y_ind,z_ind),(x_w,y_w,z_w) = CIC_weights_node(r, dims, False)
   
   r_data[:,0] += x_w[0]*xface_data[z_ind[0],y_ind[0],x_ind[0]]
   r_data[:,0] += x_w[1]*xface_data[z_ind[0],y_ind[0],x_ind[1]]

   r_data[:,1] += y_w[0]*yface_data[z_ind[0],y_ind[0],x_ind[0]]
   r_data[:,1] += y_w[1]*yface_data[z_ind[0],y_ind[1],x_ind[0]]

   r_data[:,2] += z_w[0]*zface_data[z_ind[0],y_ind[0],x_ind[0]]
   r_data[:,2] += z_w[1]*zface_data[z_ind[1],y_ind[0],x_ind[0]]
   
   return r_data

@njit(cache = True, fastmath = True)
def face2cloud_njit(face_data, r, dims):
   # Interpolate face data to cloud located at arbitrary position(s) r
   # Data is assumed to be vector data, face data only stores one component each
   if r.ndim == 1:
      r = r.reshape(-1,3)
   
   p_count = r.shape[0]

   r_data = np.zeros((p_count,3))
   
   for ii in range(p_count):
      x_locs_l = math.floor((r[ii,0] - dims.x_min)/dims.dx)
      y_locs_l = math.floor((r[ii,1] - dims.y_min)/dims.dy)
      z_locs_l = math.floor((r[ii,2] - dims.z_min)/dims.dz)
      
      x0 = x_locs_l*dims.dx + dims.x_min
      y0 = y_locs_l*dims.dy + dims.y_min
      z0 = z_locs_l*dims.dz + dims.z_min
      x_w1 = (r[ii,0] - x0)/dims.dx
      y_w1 = (r[ii,1] - y0)/dims.dy
      z_w1 = (r[ii,2] - z0)/dims.dz
      x_w0 = 1 - x_w1
      y_w0 = 1 - y_w1
      z_w0 = 1 - z_w1
      
      x_locs_r = dims.x_range_l2r[x_locs_l]
      y_locs_r = dims.y_range_l2r[y_locs_l]
      z_locs_r = dims.z_range_l2r[z_locs_l]
      
      r_data[ii,0] += x_w0*face_data[z_locs_l,y_locs_l,x_locs_l,0]
      r_data[ii,0] += x_w1*face_data[z_locs_l,y_locs_l,x_locs_r,0]
      
      r_data[ii,1] += y_w0*face_data[z_locs_l,y_locs_l,x_locs_l,1]
      r_data[ii,1] += y_w1*face_data[z_locs_l,y_locs_r,x_locs_l,1]
      
      r_data[ii,2] += z_w0*face_data[z_locs_l,y_locs_l,x_locs_l,2]
      r_data[ii,2] += z_w1*face_data[z_locs_r,y_locs_l,x_locs_l,2]
   
   return r_data

def node2cloud(node_data, r, dims):
   # Interpolate node data to cloud located at arbitrary position(s) r
   r = np.array(r).reshape(-1,3)
   vec = (node_data.ndim == 4)
   
   if vec:
      r_data = np.zeros(r.shape)
   else:
      r_data = np.zeros(r.shape[0])
   
   x_ind0 = np.round((r[:,0] - dims.x_min)/dims.dx).astype(int)
   y_ind0 = np.round((r[:,1] - dims.y_min)/dims.dy).astype(int)
   z_ind0 = np.round((r[:,2] - dims.z_min)/dims.dz).astype(int)
   
   x0 = x_ind0*dims.dx + dims.x_min
   y0 = y_ind0*dims.dy + dims.y_min
   z0 = z_ind0*dims.dz + dims.z_min

   r_x = (r[:,0] - x0)/dims.dx
   r_y = (r[:,1] - y0)/dims.dy
   r_z = (r[:,2] - z0)/dims.dz

   x_ind = np.array((x_ind0 - 1, x_ind0, x_ind0 + 1))
   y_ind = np.array((y_ind0 - 1, y_ind0, y_ind0 + 1))
   z_ind = np.array((z_ind0 - 1, z_ind0, z_ind0 + 1))
   
   if dims.period[2]:
      x_ind = np.mod(x_ind, dims.x_size)
   else:
      x_ind = np.clip(x_ind, 0, dims.x_size - 1)
   
   if dims.period[1]:
      y_ind = np.mod(y_ind, dims.y_size)
   else:
      y_ind = np.clip(y_ind, 0, dims.y_size - 1)
   
   if dims.period[0]:
      z_ind = np.mod(z_ind, dims.z_size)
   else:
      z_ind = np.clip(z_ind, 0, dims.z_size - 1)

   x_w = np.array((0.5*(0.5-r_x)**2, 0.75-r_x**2, 0.5*(0.5+r_x)**2))
   y_w = np.array((0.5*(0.5-r_y)**2, 0.75-r_y**2, 0.5*(0.5+r_y)**2))
   z_w = np.array((0.5*(0.5-r_z)**2, 0.75-r_z**2, 0.5*(0.5+r_z)**2))

   x_w = x_w.reshape(1,1,3,-1)
   y_w = y_w.reshape(1,3,1,-1)
   z_w = z_w.reshape(3,1,1,-1)
   
   weights = x_w*y_w*z_w
   
   if vec:
      weights = weights[:,:,:,:,np.newaxis]

   r_data += weights[0,0,0]*node_data[z_ind[0], y_ind[0], x_ind[0]]
   r_data += weights[0,0,1]*node_data[z_ind[0], y_ind[0], x_ind[1]]
   r_data += weights[0,0,2]*node_data[z_ind[0], y_ind[0], x_ind[2]]
   r_data += weights[0,1,0]*node_data[z_ind[0], y_ind[1], x_ind[0]]
   r_data += weights[0,1,1]*node_data[z_ind[0], y_ind[1], x_ind[1]]
   r_data += weights[0,1,2]*node_data[z_ind[0], y_ind[1], x_ind[2]]
   r_data += weights[0,2,0]*node_data[z_ind[0], y_ind[2], x_ind[0]]
   r_data += weights[0,2,1]*node_data[z_ind[0], y_ind[2], x_ind[1]]
   r_data += weights[0,2,2]*node_data[z_ind[0], y_ind[2], x_ind[2]]
   r_data += weights[1,0,0]*node_data[z_ind[1], y_ind[0], x_ind[0]]
   r_data += weights[1,0,1]*node_data[z_ind[1], y_ind[0], x_ind[1]]
   r_data += weights[1,0,2]*node_data[z_ind[1], y_ind[0], x_ind[2]]
   r_data += weights[1,1,0]*node_data[z_ind[1], y_ind[1], x_ind[0]]
   r_data += weights[1,1,1]*node_data[z_ind[1], y_ind[1], x_ind[1]]
   r_data += weights[1,1,2]*node_data[z_ind[1], y_ind[1], x_ind[2]]
   r_data += weights[1,2,0]*node_data[z_ind[1], y_ind[2], x_ind[0]]
   r_data += weights[1,2,1]*node_data[z_ind[1], y_ind[2], x_ind[1]]
   r_data += weights[1,2,2]*node_data[z_ind[1], y_ind[2], x_ind[2]]
   r_data += weights[2,0,0]*node_data[z_ind[2], y_ind[0], x_ind[0]]
   r_data += weights[2,0,1]*node_data[z_ind[2], y_ind[0], x_ind[1]]
   r_data += weights[2,0,2]*node_data[z_ind[2], y_ind[0], x_ind[2]]
   r_data += weights[2,1,0]*node_data[z_ind[2], y_ind[1], x_ind[0]]
   r_data += weights[2,1,1]*node_data[z_ind[2], y_ind[1], x_ind[1]]
   r_data += weights[2,1,2]*node_data[z_ind[2], y_ind[1], x_ind[2]]
   r_data += weights[2,2,0]*node_data[z_ind[2], y_ind[2], x_ind[0]]
   r_data += weights[2,2,1]*node_data[z_ind[2], y_ind[2], x_ind[1]]
   r_data += weights[2,2,2]*node_data[z_ind[2], y_ind[2], x_ind[2]]
   
   if r.shape[0] == 1:
      r_data = r_data.reshape(3)
   
   return r_data

@njit(cache = True, fastmath = True)
def node2cloud_njit(node_data, r, dims):
   # Interpolate node data to cloud located at arbitrary position(s) r
   if r.ndim == 1:
      r = r.reshape(-1,3)
   vec = (node_data.ndim == 4)
   
   p_count = r.shape[0]

   if vec:
      r_data = np.zeros((p_count,3))
   else:
      r_data = np.zeros(p_count)
   
   for ii in range(p_count):
      x_locs_l = math.floor((r[ii,0] - dims.x_min)/dims.dx)
      y_locs_l = math.floor((r[ii,1] - dims.y_min)/dims.dy)
      z_locs_l = math.floor((r[ii,2] - dims.z_min)/dims.dz)
      
      x0 = x_locs_l*dims.dx + dims.x_min
      y0 = y_locs_l*dims.dy + dims.y_min
      z0 = z_locs_l*dims.dz + dims.z_min
      x_w1 = r[ii,0] - x0
      y_w1 = r[ii,1] - y0
      z_w1 = r[ii,2] - z0

      if dims.linear:
         x_w1 /= dims.dx
         y_w1 /= dims.dy
         z_w1 /= dims.dz
         x_w0 = 1 - x_w1
         y_w0 = 1 - y_w1
         z_w0 = 1 - z_w1
         weights = np.array([x_w0*y_w0*z_w0,x_w1*y_w0*z_w0,
                             x_w0*y_w1*z_w0,x_w1*y_w1*z_w0,
                             x_w0*y_w0*z_w1,x_w1*y_w0*z_w1,
                             x_w0*y_w1*z_w1,x_w1*y_w1*z_w1])
      else:
         x_w0 = dims.dx - x_w1
         y_w0 = dims.dy - y_w1
         z_w0 = dims.dz - z_w1
         weights = np.array([[x_w1,y_w1,z_w1],[x_w0,y_w1,z_w1],
                             [x_w1,y_w0,z_w1],[x_w0,y_w0,z_w1],
                             [x_w1,y_w1,z_w0],[x_w0,y_w1,z_w0],
                             [x_w1,y_w0,z_w0],[x_w0,y_w0,z_w0]])
         weights = np.sqrt((weights**2).sum(axis = 1))
         min_index = weights.argmin()
         if weights[min_index] == 0:
            weights = np.zeros(8, dtype = float64)
            weights[min_index] = 1
         else:
            weights = 1/weights
         weights /= np.sum(weights, axis = 0)
      
      x_locs_r = dims.x_range_l2r[x_locs_l]
      y_locs_r = dims.y_range_l2r[y_locs_l]
      z_locs_r = dims.z_range_l2r[z_locs_l]

      if node_data.ndim == 3:
         r_data[ii] += weights[0]*node_data[z_locs_l, y_locs_l, x_locs_l]
         r_data[ii] += weights[1]*node_data[z_locs_l, y_locs_l, x_locs_r]
         r_data[ii] += weights[2]*node_data[z_locs_l, y_locs_r, x_locs_l]
         r_data[ii] += weights[3]*node_data[z_locs_l, y_locs_r, x_locs_r]
         r_data[ii] += weights[4]*node_data[z_locs_r, y_locs_l, x_locs_l]
         r_data[ii] += weights[5]*node_data[z_locs_r, y_locs_l, x_locs_r]
         r_data[ii] += weights[6]*node_data[z_locs_r, y_locs_r, x_locs_l]
         r_data[ii] += weights[7]*node_data[z_locs_r, y_locs_r, x_locs_r]
      elif node_data.ndim == 4:
         for nn in range(3):
            r_data[ii,nn] += weights[0]*node_data[z_locs_l, y_locs_l, x_locs_l,nn]
            r_data[ii,nn] += weights[1]*node_data[z_locs_l, y_locs_l, x_locs_r,nn]
            r_data[ii,nn] += weights[2]*node_data[z_locs_l, y_locs_r, x_locs_l,nn]
            r_data[ii,nn] += weights[3]*node_data[z_locs_l, y_locs_r, x_locs_r,nn]
            r_data[ii,nn] += weights[4]*node_data[z_locs_r, y_locs_l, x_locs_l,nn]
            r_data[ii,nn] += weights[5]*node_data[z_locs_r, y_locs_l, x_locs_r,nn]
            r_data[ii,nn] += weights[6]*node_data[z_locs_r, y_locs_r, x_locs_l,nn]
            r_data[ii,nn] += weights[7]*node_data[z_locs_r, y_locs_r, x_locs_r,nn]
   
   return r_data

def cell2cloud(cell_data, r, dims):
   # Interpolate cell data to cloud located at arbitrary position(s) r
   r = np.array(r).reshape(-1,3)
   vec = (cell_data.ndim == 4)
   
   if vec:
      r_data = np.zeros(r.shape)
   else:
      r_data = np.zeros(r.shape[0])
   
   if dims.linear:
      (x_ind,y_ind,z_ind),(x_w,y_w,z_w) = CIC_weights_cell(r, dims, True, True)
      weights = x_w*y_w*z_w
   else:
      (x_ind,y_ind,z_ind),(x_w,y_w,z_w) = CIC_weights_cell(r, dims, False, False)
      weights = np.array([[x_w[1],y_w[1],z_w[1]],[x_w[0],y_w[1],z_w[1]],
                          [x_w[1],y_w[0],z_w[1]],[x_w[0],y_w[0],z_w[1]],
                          [x_w[1],y_w[1],z_w[0]],[x_w[0],y_w[1],z_w[0]],
                          [x_w[1],y_w[0],z_w[0]],[x_w[0],y_w[0],z_w[0]]])
      weights = np.linalg.norm(weights, axis = 1)
      with np.errstate(divide = 'raise'):
         # Deal with rare distance == 0 case
         # Usually distance will not be 0 so use try block to
         # attempt normal way first to prevent this from slowing the code
         try:
            weights = 1/weights
         except FloatingPointError:
            for ii in range(r.shape[0]):
               try:
                  weights[:,ii] = 1/weights[:,ii]
               except FloatingPointError:
                  min_index = np.argmin(weights[:,ii])
                  weights[:,ii] = np.zeros(8)
                  weights[min_index,ii] = 1
                  
      weights /= np.sum(weights, axis = 0)
      
   if vec:
      weights = weights.reshape(2,2,2,-1,1)
   else:
      weights = weights.reshape(2,2,2,-1)
   
   r_data += weights[0,0,0]*cell_data[z_ind[0], y_ind[0], x_ind[0]]
   r_data += weights[0,0,1]*cell_data[z_ind[0], y_ind[0], x_ind[1]]
   r_data += weights[0,1,0]*cell_data[z_ind[0], y_ind[1], x_ind[0]]
   r_data += weights[0,1,1]*cell_data[z_ind[0], y_ind[1], x_ind[1]]
   r_data += weights[1,0,0]*cell_data[z_ind[1], y_ind[0], x_ind[0]]
   r_data += weights[1,0,1]*cell_data[z_ind[1], y_ind[0], x_ind[1]]
   r_data += weights[1,1,0]*cell_data[z_ind[1], y_ind[1], x_ind[0]]
   r_data += weights[1,1,1]*cell_data[z_ind[1], y_ind[1], x_ind[1]]

   if r.shape[0] == 1:
      if vec:
         r_data = r_data.reshape(3)
      else:
         r_data = r_data.item()
   
   return r_data

@njit(cache = True, fastmath = True)
def cell2cloud_njit(cell_data, r, dims):
   # Interpolate cell data to cloud located at arbitrary position(s) r
   if r.ndim == 1:
      r = r.reshape(-1,3)
   vec = (cell_data.ndim == 4)

   p_count = r.shape[0]

   if vec:
      r_data = np.zeros((p_count,3))
   else:
      r_data = np.zeros(p_count)
   
   for ii in range(p_count):
      x_locs_r = round((r[ii,0] - dims.x_min)/dims.dx)
      y_locs_r = round((r[ii,1] - dims.y_min)/dims.dy)
      z_locs_r = round((r[ii,2] - dims.z_min)/dims.dz)
      
      # if dims.period[2]:
      #    x_locs_r = x_locs_r % dims.x_size
      # if dims.period[1]:
      #    y_locs_r = y_locs_r % dims.y_size
      # if dims.period[0]:
      #    z_locs_r = z_locs_r % dims.z_size
      
      x0 = (x_locs_r - 1 + 0.5)*dims.dx + dims.x_min
      y0 = (y_locs_r - 1 + 0.5)*dims.dy + dims.y_min
      z0 = (z_locs_r - 1 + 0.5)*dims.dz + dims.z_min
      x_w0 = r[ii,0] - x0
      y_w0 = r[ii,1] - y0
      z_w0 = r[ii,2] - z0
      
      # if dims.period[2]:
      #    x_w1 = x_w1 % 1
      # if dims.period[1]:
      #    y_w1 = y_w1 % 1
      # if dims.period[0]:
      #    z_w1 = z_w1 % 1

      if dims.linear:
         x_w1 = x_w0/dims.dx
         y_w1 = y_w0/dims.dy
         z_w1 = z_w0/dims.dz
         x_w0 = 1 - x_w1
         y_w0 = 1 - y_w1
         z_w0 = 1 - z_w1
         weights = np.array([x_w0*y_w0*z_w0,x_w1*y_w0*z_w0,
                             x_w0*y_w1*z_w0,x_w1*y_w1*z_w0,
                             x_w0*y_w0*z_w1,x_w1*y_w0*z_w1,
                             x_w0*y_w1*z_w1,x_w1*y_w1*z_w1])
      else:
         x_w1 = dims.dx - x_w0
         y_w1 = dims.dy - y_w0
         z_w1 = dims.dz - z_w0
         weights = np.array([[x_w0,y_w0,z_w0],[x_w1,y_w0,z_w0],
                             [x_w0,y_w1,z_w0],[x_w1,y_w1,z_w0],
                             [x_w0,y_w0,z_w1],[x_w1,y_w0,z_w1],
                             [x_w0,y_w1,z_w1],[x_w1,y_w1,z_w1]])
         weights = np.sqrt((weights**2).sum(axis = 1))
         min_index = weights.argmin()
         if weights[min_index] == 0:
            weights = np.zeros(8, dtype = float64)
            weights[min_index] = 1
         else:
            weights = 1/weights
         
         weights /= np.sum(weights)

      if dims.period[2]:
         x_locs_r = x_locs_r % dims.x_size
      if dims.period[1]:
         y_locs_r = y_locs_r % dims.y_size
      if dims.period[0]:
         z_locs_r = z_locs_r % dims.z_size
      
      x_locs_l = dims.x_range_r2l[x_locs_r]
      y_locs_l = dims.y_range_r2l[y_locs_r]
      z_locs_l = dims.z_range_r2l[z_locs_r]

      if vec:
         for nn in range(3):
            r_data[ii,nn] += weights[0]*cell_data[z_locs_l, y_locs_l, x_locs_l,nn]
            r_data[ii,nn] += weights[1]*cell_data[z_locs_l, y_locs_l, x_locs_r,nn]
            r_data[ii,nn] += weights[2]*cell_data[z_locs_l, y_locs_r, x_locs_l,nn]
            r_data[ii,nn] += weights[3]*cell_data[z_locs_l, y_locs_r, x_locs_r,nn]
            r_data[ii,nn] += weights[4]*cell_data[z_locs_r, y_locs_l, x_locs_l,nn]
            r_data[ii,nn] += weights[5]*cell_data[z_locs_r, y_locs_l, x_locs_r,nn]
            r_data[ii,nn] += weights[6]*cell_data[z_locs_r, y_locs_r, x_locs_l,nn]
            r_data[ii,nn] += weights[7]*cell_data[z_locs_r, y_locs_r, x_locs_r,nn]
      else:
         r_data[ii] += weights[0]*cell_data[z_locs_l, y_locs_l, x_locs_l]
         r_data[ii] += weights[1]*cell_data[z_locs_l, y_locs_l, x_locs_r]
         r_data[ii] += weights[2]*cell_data[z_locs_l, y_locs_r, x_locs_l]
         r_data[ii] += weights[3]*cell_data[z_locs_l, y_locs_r, x_locs_r]
         r_data[ii] += weights[4]*cell_data[z_locs_r, y_locs_l, x_locs_l]
         r_data[ii] += weights[5]*cell_data[z_locs_r, y_locs_l, x_locs_r]
         r_data[ii] += weights[6]*cell_data[z_locs_r, y_locs_r, x_locs_l]
         r_data[ii] += weights[7]*cell_data[z_locs_r, y_locs_r, x_locs_r]
   
   return r_data

def div_face2cell(face_data, dims):
   # Compute cell-based divergence from face data
   # N.B.: Since arr_diff subtracts the shifted version from
   # the original, a negative shift subtracts the upper from the lower
   # so to get difference in correct direction flip the sign
   xface_data,yface_data,zface_data = split_axis(face_data, axis = 3)
   
   cell_div = np.zeros(dims.dim_scalar, dtype = np.float64)
   
   cell_div[:,:,:] -= arr_diff(xface_data, -1, 2, dims.period)/dims.dx
   cell_div[:,:,:] -= arr_diff(yface_data, -1, 1, dims.period)/dims.dy
   cell_div[:,:,:] -= arr_diff(zface_data, -1, 0, dims.period)/dims.dz
   
   return cell_div

@njit(cache = True, fastmath = True)
def div_face2cell_njit(face_data, dims):
   # Compute cell-based divergence from face data
   # N.B.: Since arr_diff_njit subtracts the shifted version from
   # the original, a negative shift subtracts the upper from the lower
   # so to get difference in correct direction flip the sign
   dim_scalar = (dims.dim_scalar[0],dims.dim_scalar[1],dims.dim_scalar[2])
   cell_div = np.zeros(dim_scalar, dtype = float64)
   
   xdiff_x = arr_diff_njit(face_data[:,:,:,0], -1, 2, dims.period)/dims.dx
   ydiff_y = arr_diff_njit(face_data[:,:,:,1], -1, 1, dims.period)/dims.dy
   zdiff_z = arr_diff_njit(face_data[:,:,:,2], -1, 0, dims.period)/dims.dz
   for kk in range(dims.z_size):
      for jj in range(dims.y_size):
         for ii in range(dims.x_size):
            cell_div[kk,jj,ii] -= xdiff_x[kk,jj,ii]
            cell_div[kk,jj,ii] -= ydiff_y[kk,jj,ii]
            cell_div[kk,jj,ii] -= zdiff_z[kk,jj,ii]
   
   return cell_div

@njit(cache = True, fastmath = True)
def div_face2cell_njit_alt(face_data, dims):
   # Compute cell-based divergence from face data
   dim_scalar = (dims.dim_scalar[0],dims.dim_scalar[1],dims.dim_scalar[2])
   cell_div = np.zeros(dim_scalar, dtype = float64)
   
   shift = np.array([[0,0,0], [0,0,1], [0,1,0], [1,0,0]], dtype = int64)
   
   for kk in range(dims.z_size):
      for jj in range(dims.y_size):
         for ii in range(dims.x_size):
            base_node = np.array((kk,jj,ii), dtype = int64)
            shift_indices = get_index_njit(base_node, shift, dims)
            sub_data = np.empty((shift.shape[0],3), dtype = float64)
            for nn,curr_ind in enumerate(shift_indices):
               sub_data[nn] = face_data[curr_ind[0],curr_ind[1],curr_ind[2]]
            
            cell_div[kk,jj,ii] += (sub_data[1,0] - sub_data[0,0])/dims.dx
            cell_div[kk,jj,ii] += (sub_data[2,1] - sub_data[0,1])/dims.dy
            cell_div[kk,jj,ii] += (sub_data[3,2] - sub_data[0,2])/dims.dz
   
   return cell_div

def div_node2cell(node_data, dims):
   # Compute cell-based divergence from node data
   # Interpolates first to faces then takes difference
   face_data = node2face(node_data, dims)

   cell_div = div_face2cell(face_data, dims)
   
   return cell_div

@njit(cache = True, fastmath = True)
def div_node2cell_njit(node_data, dims):
   # Compute cell-based divergence from node data
   # Interpolates first to faces then takes difference
   face_data = node2face_njit(node_data, dims)

   cell_div = div_face2cell_njit(face_data, dims)
   
   return cell_div

@njit(cache = True, fastmath = True)
def div_node2cell_njit_alt(node_data, dims):
   # Compute cell-based divergence from node data
   # Interpolates first to faces then takes difference
   face_data = node2face_njit_alt(node_data, dims)

   cell_div = div_face2cell_njit_alt(face_data, dims)
   
   return cell_div

def curl_face2node(face_data, dims):
   # Compute the curl of face data at nodes
   # First computes edge curl then interpolates to node
   node_curl = np.empty(dims.dim_vector, dtype = np.float64)
   
   xface_data,yface_data,zface_data = split_axis(face_data, axis = 3)

   # Diff of y-face data in z-direction
   ydiff_z = arr_diff(yface_data, 1, 0, dims.period)/dims.dz
   # Diff of z-face data in y-direction
   zdiff_y = arr_diff(zface_data, 1, 1, dims.period)/dims.dy

   # Diff of z-face data in x-direction
   zdiff_x = arr_diff(zface_data, 1, 2, dims.period)/dims.dx
   # Diff of x-face data in z-direction
   xdiff_z = arr_diff(xface_data, 1, 0, dims.period)/dims.dz   
   
   # Diff of x-face data in y-direction
   xdiff_y = arr_diff(xface_data, 1, 1, dims.period)/dims.dy
   # Diff of y-face data in x-direction
   ydiff_x = arr_diff(yface_data, 1, 2, dims.period)/dims.dx
   
   edgeJx = zdiff_y - ydiff_z
   edgeJx += arr_shift(edgeJx, 1, 2, dims.period)

   edgeJy = xdiff_z - zdiff_x
   edgeJy += arr_shift(edgeJy, 1, 1, dims.period)

   edgeJz = ydiff_x - xdiff_y
   edgeJz += arr_shift(edgeJz, 1, 0, dims.period)
   
   node_curl[:,:,:,0] = edgeJx
   node_curl[:,:,:,1] = edgeJy
   node_curl[:,:,:,2] = edgeJz
   
   node_curl *= 0.5
   
   return node_curl

@njit(cache = True, fastmath = True)
def curl_face2node_njit(face_data, dims):
   # Compute the curl of face data at nodes
   # First computes edge curl then interpolates to node
   dim_vector = (dims.dim_vector[0],dims.dim_vector[1],
                 dims.dim_vector[2],dims.dim_vector[3])
   node_curl = np.empty(dim_vector, dtype = float64)
   
   # Diff of y-face data in z-direction
   ydiff_z = arr_diff_njit(face_data[:,:,:,1], 1, 0, dims.period)/dims.dz
   # Diff of z-face data in y-direction
   zdiff_y = arr_diff_njit(face_data[:,:,:,2], 1, 1, dims.period)/dims.dy

   # Diff of z-face data in x-direction
   zdiff_x = arr_diff_njit(face_data[:,:,:,2], 1, 2, dims.period)/dims.dx
   # Diff of x-face data in z-direction
   xdiff_z = arr_diff_njit(face_data[:,:,:,0], 1, 0, dims.period)/dims.dz

   # Diff of x-face data in y-direction
   xdiff_y = arr_diff_njit(face_data[:,:,:,0], 1, 1, dims.period)/dims.dy
   # Diff of y-face data in x-direction
   ydiff_x = arr_diff_njit(face_data[:,:,:,1], 1, 2, dims.period)/dims.dx
   
   edgeJx = zdiff_y - ydiff_z
   xshift_x = arr_shift_njit(edgeJx, 1, 2, dims.period)

   edgeJy = xdiff_z - zdiff_x
   yshift_y = arr_shift_njit(edgeJy, 1, 1, dims.period)
   
   edgeJz = ydiff_x - xdiff_y
   zshift_z = arr_shift_njit(edgeJz, 1, 0, dims.period)
   
   node_curl[:,:,:,0] = edgeJx + xshift_x
   node_curl[:,:,:,1] = edgeJy + yshift_y
   node_curl[:,:,:,2] = edgeJz + zshift_z
   
   node_curl *= 0.5
   
   return node_curl

@njit(cache = True, fastmath = True)
def curl_face2node_njit_alt(face_data, dims):
   # Compute the curl of face data at nodes
   # First computes edge curl then interpolates to node
   dim_vector = (dims.dim_vector[0],dims.dim_vector[1],
                 dims.dim_vector[2],dims.dim_vector[3])
   node_curl = np.empty(dim_vector, dtype = float64)
   
   shift = np.array([[0,0,0], [0,0,-1], [0,-1,0], [0,-1,-1],
                     [-1,0,0], [-1,0,-1], [-1,-1,0]], dtype = int64)
   
   for kk in range(dims.z_size):
      for jj in range(dims.y_size):
         for ii in range(dims.x_size):
            base_node = np.array((kk,jj,ii), dtype = int64)
            shift_indices = get_index_njit(base_node, shift, dims)
            sub_data = np.empty((shift.shape[0],3), dtype = float64)
            for nn,curr_ind in enumerate(shift_indices):
               sub_data[nn] = face_data[curr_ind[0],curr_ind[1],curr_ind[2]]
            
            node_curl[kk,jj,ii,0] = (sub_data[5,1] - sub_data[1,1])/dims.dz
            node_curl[kk,jj,ii,0] += (sub_data[1,2] - sub_data[3,2])/dims.dy
            node_curl[kk,jj,ii,0] += (sub_data[4,1] - sub_data[0,1])/dims.dz
            node_curl[kk,jj,ii,0] += (sub_data[0,2] - sub_data[2,2])/dims.dy

            node_curl[kk,jj,ii,1] = (sub_data[3,2] - sub_data[2,2])/dims.dx
            node_curl[kk,jj,ii,1] += (sub_data[2,0] - sub_data[6,0])/dims.dz
            node_curl[kk,jj,ii,1] += (sub_data[1,2] - sub_data[0,2])/dims.dx
            node_curl[kk,jj,ii,1] += (sub_data[0,0] - sub_data[4,0])/dims.dz

            node_curl[kk,jj,ii,2] = (sub_data[6,0] - sub_data[4,0])/dims.dy
            node_curl[kk,jj,ii,2] += (sub_data[4,1] - sub_data[5,1])/dims.dx
            node_curl[kk,jj,ii,2] += (sub_data[2,0] - sub_data[0,0])/dims.dy
            node_curl[kk,jj,ii,2] += (sub_data[0,1] - sub_data[1,1])/dims.dx
   
   node_curl *= 0.5
   
   return node_curl

def curl_node2face(node_data, dims):
   # Compute the curl of node data at faces
   face_curl = np.zeros(dims.dim_vector, dtype = np.float64)
   
   xnode_data,ynode_data,znode_data = split_axis(node_data, axis = 3)

   xmean_x = (xnode_data + arr_shift(xnode_data, -1, 2, dims.period))/2
   ymean_y = (ynode_data + arr_shift(ynode_data, -1, 1, dims.period))/2
   zmean_z = (znode_data + arr_shift(znode_data, -1, 0, dims.period))/2
   
   ymean_y_diff_z = arr_diff(ymean_y, -1, 0, dims.period)/dims.dz
   zmean_z_diff_y = arr_diff(zmean_z, -1, 1, dims.period)/dims.dy

   zmean_z_diff_x = arr_diff(zmean_z, -1, 2, dims.period)/dims.dx
   xmean_x_diff_z = arr_diff(xmean_x, -1, 0, dims.period)/dims.dz

   xmean_x_diff_y = arr_diff(xmean_x, -1, 1, dims.period)/dims.dy
   ymean_y_diff_x = arr_diff(ymean_y, -1, 2, dims.period)/dims.dx
   
   face_curl[:,:,:,0] += ymean_y_diff_z
   face_curl[:,:,:,0] -= zmean_z_diff_y
   
   face_curl[:,:,:,1] += zmean_z_diff_x
   face_curl[:,:,:,1] -= xmean_x_diff_z
   
   face_curl[:,:,:,2] += xmean_x_diff_y
   face_curl[:,:,:,2] -= ymean_y_diff_x
   
   return face_curl

@njit(cache = True, fastmath = True)
def curl_node2face_njit(node_data, dims):
   # Compute the curl of node data at faces
   dim_vector = (dims.dim_vector[0],dims.dim_vector[1],
                 dims.dim_vector[2],dims.dim_vector[3])
   face_curl = np.zeros(dim_vector, dtype = float64)
   
   xmean_x = (node_data[:,:,:,0] + arr_shift_njit(node_data[:,:,:,0], -1, 2, dims.period))/2
   ymean_y = (node_data[:,:,:,1] + arr_shift_njit(node_data[:,:,:,1], -1, 1, dims.period))/2
   zmean_z = (node_data[:,:,:,2] + arr_shift_njit(node_data[:,:,:,2], -1, 0, dims.period))/2
   
   ymean_y_diff_z = arr_diff_njit(ymean_y, -1, 0, dims.period)/dims.dz
   zmean_z_diff_y = arr_diff_njit(zmean_z, -1, 1, dims.period)/dims.dy

   zmean_z_diff_x = arr_diff_njit(zmean_z, -1, 2, dims.period)/dims.dx
   xmean_x_diff_z = arr_diff_njit(xmean_x, -1, 0, dims.period)/dims.dz

   xmean_x_diff_y = arr_diff_njit(xmean_x, -1, 1, dims.period)/dims.dy
   ymean_y_diff_x = arr_diff_njit(ymean_y, -1, 2, dims.period)/dims.dx
   
   face_curl[:,:,:,0] += ymean_y_diff_z
   face_curl[:,:,:,0] -= zmean_z_diff_y
   
   face_curl[:,:,:,1] += zmean_z_diff_x
   face_curl[:,:,:,1] -= xmean_x_diff_z
   
   face_curl[:,:,:,2] += xmean_x_diff_y
   face_curl[:,:,:,2] -= ymean_y_diff_x
   
   return face_curl

@njit(cache = True, fastmath = True)
def curl_node2face_njit_alt(node_data, dims):
   # Compute the curl of node data at faces
   dim_vector = (dims.dim_vector[0],dims.dim_vector[1],
                 dims.dim_vector[2],dims.dim_vector[3])
   face_curl = np.empty(dim_vector, dtype = float64)
   
   shift = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1],
                     [1,0,0], [1,0,1], [1,1,0]], dtype = int64)
   
   for kk in range(dims.z_size):
      for jj in range(dims.y_size):
         for ii in range(dims.x_size):
            base_node = np.array((kk,jj,ii), dtype = int64)
            shift_indices = get_index_njit(base_node, shift, dims)
            sub_data = np.empty((shift.shape[0],3), dtype = float64)
            for nn,curr_ind in enumerate(shift_indices):
               sub_data[nn] = node_data[curr_ind[0],curr_ind[1],curr_ind[2]]
            face_curl[kk,jj,ii,0] = (sub_data[0,1] + sub_data[2,1])/dims.dz
            face_curl[kk,jj,ii,0] += (sub_data[2,2] + sub_data[6,2])/dims.dy
            face_curl[kk,jj,ii,0] -= (sub_data[6,1] + sub_data[4,1])/dims.dz
            face_curl[kk,jj,ii,0] -= (sub_data[4,2] + sub_data[0,2])/dims.dy
            
            face_curl[kk,jj,ii,1] = (sub_data[0,2] + sub_data[4,2])/dims.dx
            face_curl[kk,jj,ii,1] += (sub_data[4,0] + sub_data[5,0])/dims.dz
            face_curl[kk,jj,ii,1] -= (sub_data[5,2] + sub_data[1,2])/dims.dx
            face_curl[kk,jj,ii,1] -= (sub_data[1,0] + sub_data[0,0])/dims.dz
            
            face_curl[kk,jj,ii,2] = (sub_data[0,0] + sub_data[1,0])/dims.dy
            face_curl[kk,jj,ii,2] += (sub_data[1,1] + sub_data[3,1])/dims.dx
            face_curl[kk,jj,ii,2] -= (sub_data[3,0] + sub_data[2,0])/dims.dy
            face_curl[kk,jj,ii,2] -= (sub_data[2,1] + sub_data[0,1])/dims.dx
   
   face_curl *= 0.5
   
   return face_curl

@ftools.lru_cache(maxsize = 128)
@njit(cache = True, fastmath = True)
def get_operator_curl_face2node(dims):
   # Returns the 3D curl operator for node2face data in matrix form
   operator = np.zeros((3*dims.Ncells_total,3*dims.Ncells_total), dtype = float64)
   
   shift = np.array([[0,0,0], [0,0,-1], [0,-1,0], [0,-1,-1],
                     [-1,0,0], [-1,0,-1], [-1,-1,0]], dtype = int64)
   
   for kk in range(dims.z_size):
      for jj in range(dims.y_size):
         for ii in range(dims.x_size):
            base_node = np.array((kk,jj,ii), dtype = int64)
            shift_indices = get_index_njit(base_node, shift, dims)
            cell_ids = 3*numba_ravel_multi_index(shift_indices,
                                                 dims.dim_scalar)
            
            row = cell_ids[0]

            operator[row + 0, cell_ids[5] + 1] += 1/dims.dz
            operator[row + 0, cell_ids[1] + 1] -= 1/dims.dz
            operator[row + 0, cell_ids[1] + 2] += 1/dims.dy
            operator[row + 0, cell_ids[3] + 2] -= 1/dims.dy
            operator[row + 0, cell_ids[4] + 1] += 1/dims.dz
            operator[row + 0, cell_ids[0] + 1] -= 1/dims.dz
            operator[row + 0, cell_ids[0] + 2] += 1/dims.dy
            operator[row + 0, cell_ids[2] + 2] -= 1/dims.dy

            operator[row + 1, cell_ids[3] + 2] += 1/dims.dx
            operator[row + 1, cell_ids[2] + 2] -= 1/dims.dx
            operator[row + 1, cell_ids[2] + 0] += 1/dims.dz
            operator[row + 1, cell_ids[6] + 0] -= 1/dims.dz
            operator[row + 1, cell_ids[1] + 2] += 1/dims.dx
            operator[row + 1, cell_ids[0] + 2] -= 1/dims.dx
            operator[row + 1, cell_ids[0] + 0] += 1/dims.dz
            operator[row + 1, cell_ids[4] + 0] -= 1/dims.dz

            operator[row + 2, cell_ids[6] + 0] += 1/dims.dy
            operator[row + 2, cell_ids[4] + 0] -= 1/dims.dy
            operator[row + 2, cell_ids[4] + 1] += 1/dims.dx
            operator[row + 2, cell_ids[5] + 1] -= 1/dims.dx
            operator[row + 2, cell_ids[2] + 0] += 1/dims.dy
            operator[row + 2, cell_ids[0] + 0] -= 1/dims.dy
            operator[row + 2, cell_ids[0] + 1] += 1/dims.dx
            operator[row + 2, cell_ids[1] + 1] -= 1/dims.dx

   operator *= 0.5

   return operator

@ftools.lru_cache(maxsize = 128)
@njit(cache = True, fastmath = True)
def get_operator_curl_node2face(dims):
   # Returns the 3D curl operator for node2face data in matrix form
   operator = np.zeros((3*dims.Ncells_total,3*dims.Ncells_total), dtype = float64)
   shift = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1],
                     [1,0,0], [1,0,1], [1,1,0]], dtype = int64)
   
   for kk in range(dims.z_size):
      for jj in range(dims.y_size):
         for ii in range(dims.x_size):
            base_node = np.array((kk,jj,ii), dtype = int64)
            shift_indices = get_index_njit(base_node, shift, dims)
            cell_ids = 3*numba_ravel_multi_index(shift_indices,
                                                 dims.dim_scalar)

            row = cell_ids[0]

            # Note: Each is divided by dx,dy or dz as the line integral
            # first multiplies by dx, dy or dz
            # (whichever direction we are following)
            # then we divide by the area of the surface
            # which cancels this out and leaves the other dimension
            
            operator[row + 0, cell_ids[0] + 1] += 1/dims.dz
            operator[row + 0, cell_ids[2] + 1] += 1/dims.dz
            operator[row + 0, cell_ids[2] + 2] += 1/dims.dy
            operator[row + 0, cell_ids[6] + 2] += 1/dims.dy
            operator[row + 0, cell_ids[6] + 1] -= 1/dims.dz
            operator[row + 0, cell_ids[4] + 1] -= 1/dims.dz
            operator[row + 0, cell_ids[4] + 2] -= 1/dims.dy
            operator[row + 0, cell_ids[0] + 2] -= 1/dims.dy
            
            operator[row + 1, cell_ids[0] + 2] += 1/dims.dx
            operator[row + 1, cell_ids[4] + 2] += 1/dims.dx
            operator[row + 1, cell_ids[4] + 0] += 1/dims.dz
            operator[row + 1, cell_ids[5] + 0] += 1/dims.dz
            operator[row + 1, cell_ids[5] + 2] -= 1/dims.dx
            operator[row + 1, cell_ids[1] + 2] -= 1/dims.dx
            operator[row + 1, cell_ids[1] + 0] -= 1/dims.dz
            operator[row + 1, cell_ids[0] + 0] -= 1/dims.dz
            
            operator[row + 2, cell_ids[0] + 0] += 1/dims.dy
            operator[row + 2, cell_ids[1] + 0] += 1/dims.dy
            operator[row + 2, cell_ids[1] + 1] += 1/dims.dx
            operator[row + 2, cell_ids[3] + 1] += 1/dims.dx
            operator[row + 2, cell_ids[3] + 0] -= 1/dims.dy
            operator[row + 2, cell_ids[2] + 0] -= 1/dims.dy
            operator[row + 2, cell_ids[2] + 1] -= 1/dims.dx
            operator[row + 2, cell_ids[0] + 1] -= 1/dims.dx

   operator *= 0.5

   return operator

@ftools.lru_cache(maxsize = 128)
@njit(cache = True, fastmath = True)
def get_operator_coo_curl_face2node(dims):
   # Returns the 3D curl operator for node2face data in coo matrix arrays
   data = np.empty(3*8*dims.Ncells_total, dtype = float64)
   cols = np.empty(3*8*dims.Ncells_total, dtype = int64)

   sub_data = np.empty(3*8, dtype = float64)
   sub_rows = np.empty(3*8, dtype = float64)
   
   sub_data[0] = 1/dims.dz
   sub_data[1] = -1/dims.dz
   sub_data[2] = 1/dims.dy
   sub_data[3] = -1/dims.dy
   sub_data[4:8] = sub_data[0:4]

   sub_data[8] = 1/dims.dx
   sub_data[9] = -1/dims.dx
   sub_data[10] = 1/dims.dz
   sub_data[11] = -1/dims.dz
   sub_data[12:16] = sub_data[8:12]

   sub_data[16] = 1/dims.dy
   sub_data[17] = -1/dims.dy
   sub_data[18] = 1/dims.dx
   sub_data[19] = -1/dims.dx
   sub_data[20:24] = sub_data[16:20]

   for ii in range(dims.Ncells_total):
      data[24*ii:24*(ii+1)] = 0.5*sub_data

   sub_rows = np.arange(3*dims.Ncells_total, dtype = int64)
   rows = np.repeat(sub_rows, 8)
   
   shift = np.array([[0,0,0], [0,0,-1], [0,-1,0], [0,-1,-1],
                     [-1,0,0], [-1,0,-1], [-1,-1,0]], dtype = int64)
   
   for kk in range(dims.z_size):
      for jj in range(dims.y_size):
         for ii in range(dims.x_size):
            base_node = np.array((kk,jj,ii), dtype = int64)
            shift_indices = get_index_njit(base_node, shift, dims)
            cell_ids = 3*numba_ravel_multi_index(shift_indices,
                                                 dims.dim_scalar)
            
            row = cell_ids[0]

            cols[row*8 + 0] = cell_ids[5] + 1
            cols[row*8 + 1] = cell_ids[1] + 1
            cols[row*8 + 2] = cell_ids[1] + 2
            cols[row*8 + 3] = cell_ids[3] + 2
            cols[row*8 + 4] = cell_ids[4] + 1
            cols[row*8 + 5] = cell_ids[0] + 1
            cols[row*8 + 6] = cell_ids[0] + 2
            cols[row*8 + 7] = cell_ids[2] + 2
            
            cols[row*8 + 8] = cell_ids[3] + 2
            cols[row*8 + 9] = cell_ids[2] + 2
            cols[row*8 + 10] = cell_ids[2] + 0
            cols[row*8 + 11] = cell_ids[6] + 0
            cols[row*8 + 12] = cell_ids[1] + 2
            cols[row*8 + 13] = cell_ids[0] + 2
            cols[row*8 + 14] = cell_ids[0] + 0
            cols[row*8 + 15] = cell_ids[4] + 0

            cols[row*8 + 16] = cell_ids[6] + 0
            cols[row*8 + 17] = cell_ids[4] + 0
            cols[row*8 + 18] = cell_ids[4] + 1
            cols[row*8 + 19] = cell_ids[5] + 1
            cols[row*8 + 20] = cell_ids[2] + 0
            cols[row*8 + 21] = cell_ids[0] + 0
            cols[row*8 + 22] = cell_ids[0] + 1
            cols[row*8 + 23] = cell_ids[1] + 1

   return data,rows,cols

@ftools.lru_cache(maxsize = 128)
@njit(cache = True, fastmath = True)
def get_operator_coo_curl_node2face(dims):
   # Returns the 3D curl operator for node2face data in matrix form
   data = np.empty(3*8*dims.Ncells_total, dtype = float64)
   cols = np.empty(3*8*dims.Ncells_total, dtype = int64)

   sub_data = np.empty(3*8, dtype = float64)
   sub_rows = np.empty(3*8, dtype = float64)
   
   sub_data[0] = 1/dims.dz
   sub_data[1] = 1/dims.dz
   sub_data[2] = 1/dims.dy
   sub_data[3] = 1/dims.dy
   sub_data[4:8] = -sub_data[0:4]

   sub_data[8] = 1/dims.dx
   sub_data[9] = 1/dims.dx
   sub_data[10] = 1/dims.dz
   sub_data[11] = 1/dims.dz
   sub_data[12:16] = -sub_data[8:12]

   sub_data[16] = 1/dims.dy
   sub_data[17] = 1/dims.dy
   sub_data[18] = 1/dims.dx
   sub_data[19] = 1/dims.dx
   sub_data[20:24] = -sub_data[16:20]

   for ii in range(dims.Ncells_total):
      data[24*ii:24*(ii+1)] = 0.5*sub_data

   sub_rows = np.arange(3*dims.Ncells_total, dtype = int64)
   rows = np.repeat(sub_rows, 8)
   
   shift = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1],
                     [1,0,0], [1,0,1], [1,1,0]], dtype = int64)
   
   for kk in range(dims.z_size):
      for jj in range(dims.y_size):
         for ii in range(dims.x_size):
            base_node = np.array((kk,jj,ii), dtype = int64)
            shift_indices = get_index_njit(base_node, shift, dims)
            cell_ids = 3*numba_ravel_multi_index(shift_indices,
                                                 dims.dim_scalar)

            row = cell_ids[0]

            # Note: Each is divided by dx,dy or dz as the line integral
            # first multiplies by dx, dy or dz
            # (whichever direction we are following)
            # then we divide by the area of the surface
            # which cancels this out and leaves the other dimension
            
            cols[row*8 + 0] = cell_ids[0] + 1
            cols[row*8 + 1] = cell_ids[2] + 1
            cols[row*8 + 2] = cell_ids[2] + 2
            cols[row*8 + 3] = cell_ids[6] + 2
            cols[row*8 + 4] = cell_ids[6] + 1
            cols[row*8 + 5] = cell_ids[4] + 1
            cols[row*8 + 6] = cell_ids[4] + 2
            cols[row*8 + 7] = cell_ids[0] + 2
            
            cols[row*8 + 8] = cell_ids[0] + 2
            cols[row*8 + 9] = cell_ids[4] + 2
            cols[row*8 + 10] = cell_ids[4] + 0
            cols[row*8 + 11] = cell_ids[5] + 0
            cols[row*8 + 12] = cell_ids[5] + 2
            cols[row*8 + 13] = cell_ids[1] + 2
            cols[row*8 + 14] = cell_ids[1] + 0
            cols[row*8 + 15] = cell_ids[0] + 0
            
            cols[row*8 + 16] = cell_ids[0] + 0
            cols[row*8 + 17] = cell_ids[1] + 0
            cols[row*8 + 18] = cell_ids[1] + 1
            cols[row*8 + 19] = cell_ids[3] + 1
            cols[row*8 + 20] = cell_ids[3] + 0
            cols[row*8 + 21] = cell_ids[2] + 0
            cols[row*8 + 22] = cell_ids[2] + 1
            cols[row*8 + 23] = cell_ids[0] + 1
            
   return data,rows,cols
