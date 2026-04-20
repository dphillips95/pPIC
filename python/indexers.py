# Module of general tools, e.g. array rolling/sliding

import math
import functools as ftools
import numpy as np
from numba import njit,int64

def get_index(initial, relative_step, dims):
   # Find index starting from initial index with relative step away
   # e.g. if index is (4,5,1) and relative_step is (0,0,2), return (4,5,3)
   # if axis is periodic then index is wrapped, else sticks to edge
   # e.g. if for last example. dims were (5,6,3) periodic, return (4,5,0)
   # and if non-periodic, return (4,5,2)
   # relative_step should be list
   new_ind = np.tile(initial, (relative_step.shape[0], 1))
   new_ind += relative_step
   if dims.period[0]:
      new_ind[:,0] = np.mod(new_ind[:,0], dims.z_size)
   else:
      new_ind[:,0] = np.clip(new_ind[:,0], 0, dims.z_size - 1)
   if dims.period[1]:
      new_ind[:,1] = np.mod(new_ind[:,1], dims.y_size)
   else:
      new_ind[:,1] = np.clip(new_ind[:,1], 0, dims.y_size - 1)
   if dims.period[0]:
      new_ind[:,2] = np.mod(new_ind[:,2], dims.x_size)
   else:
      new_ind[:,2] = np.clip(new_ind[:,2], 0, dims.x_size - 1)

   return new_ind

@njit(cache = True, fastmath = True)
def get_index_njit(initial, relative_step, dims):
   # Find index starting from initial index with relative step away
   # e.g. if index is (4,5,1) and relative_step is (0,0,2), return (4,5,3)
   # if axis is periodic then index is wrapped, else sticks to edge
   # e.g. if for last example. dims were (5,6,3) periodic, return (4,5,0)
   # and if non-periodic, return (4,5,2)
   # relative_step should be 2d array, each row is a different step
   new_ind = np.empty(relative_step.shape, dtype = initial.dtype)
   for ii in range(relative_step.shape[0]):
      for jj in range(3):
         new_ind[ii,jj] = initial[jj] + relative_step[ii,jj]
         if dims.period[jj]:
            new_ind[ii,jj] = new_ind[ii,jj]%dims.dim_scalar[jj]
         else:
            new_ind[ii,jj] = min(max(new_ind[ii,jj], 0), dims.dim_scalar[jj] - 1)
   
   return new_ind

@njit(cache = True, fastmath = True)
def numba_clip(arr, min_val, max_val):
   # Numba-compatible version of np.clip
   shape = arr.shape
   clipped_arr = arr.flatten()
   for ii,x in enumerate(clipped_arr):
      clipped_arr[ii] = min(max(x, min_val), max_val)

   clipped_arr.reshape(shape)

   return clipped_arr

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

def shift_indices(arr, shift, periodicity):
   # Shift 1D array of indices
   if periodicity is True:
      return np.roll(arr, shift)
   return shift_indices_njit(arr, shift, periodicity)

@njit(cache = True, fastmath = True)
def shift_indices_njit(arr, shift, periodicity):
   # Shift 1D array of indices
   if periodicity is True:
      return np.roll(arr, shift)

   shape = arr.shape
   sh_arr = np.zeros(shape, dtype = np.int64)
   if shift > 0:
      for ii in range(shift):
         sh_arr[ii] = arr[0]
      for ii in range(shift, shape[0]):
         sh_arr[ii] = arr[ii-shift]
   elif shift < 0:
      for ii in range(shape[0] + shift):
         sh_arr[ii] = arr[ii-shift]
      for ii in range(shape[0] + shift, shape[0]):
         sh_arr[ii] = arr[-1]
   
   return sh_arr

@njit(cache = True, fastmath = True)
def arr_shift_multi(arr, shift, axis, periodicity):
   # Calls arr_shift for each axis
   # Shift must be a tuple of same length as axis
   sh_arr = arr.copy()
   for sh,ax in zip(shift, axis):
      sh_arr = arr_shift(sh_arr, sh, ax, periodicity)

   return sh_arr

def arr_shift(arr, shift, axis, periodicity):
   # Shift array, if axis is periodic then rolls otherwise uses array_indices
   # periodicity is iterable of bools of length arr.ndim that encodes if each axis is periodic
   # i.e. the same as np.roll, but does not wrap if not periodic
   if shift == 0:
      return arr
   if periodicity[axis]:
      sh_arr = np.roll(arr, shift, axis)
   else:
      sh_arr = slide_array(arr, shift, axis)
   
   return sh_arr

@njit(cache = True, fastmath = True)
def arr_shift_njit(arr, shift, axis, periodicity):
   # Shift array, if axis is periodic then rolls otherwise uses array_indices
   # periodicity is iterable of bools of length arr.ndim that encodes if each axis is periodic
   # i.e. the same as np.roll, but does not wrap if not periodic
   if shift == 0:
      return arr
   if periodicity[axis]:
      sh_arr = roll_array(arr, shift, axis)
   else:
      sh_arr = slide_array(arr, shift, axis)
   
   return sh_arr

def arr_diff(arr, shift, axis, periodicity):
   # Shift array then take difference from original
   diff_arr = arr.copy()
   
   diff_arr -= arr_shift(arr, shift, axis, periodicity)

   return diff_arr

@njit(cache = True, fastmath = True)
def arr_diff_njit(arr, shift, axis, periodicity):
   # Shift array then take difference from original
   diff_arr = arr.copy()
   
   diff_arr -= arr_shift_njit(arr, shift, axis, periodicity)

   return diff_arr

@njit(cache = True, fastmath = True)
def slide_array(arr, shift, axis):
   # Shift array without rolling (new elements become equal to known neighbour)
   ndim = arr.ndim
   if shift == 0:
      return arr
   if ndim == 1:
      sh_arr = shift_1D(arr, shift)
   elif ndim == 2:
      sh_arr = shift_2D(arr, shift, axis)
   elif ndim == 3:
      sh_arr = shift_3D(arr, shift, axis)
   elif ndim == 4:
      sh_arr = shift_4D(arr, shift, axis)
      
   return sh_arr

@njit(cache = True, fastmath = True)
def shift_1D(arr, shift):
   # Shift 1D array
   shape = arr.shape
   sh_arr = np.zeros(shape, dtype = np.float64)
   if shift > 0:
      for ii in range(shift):
         sh_arr[ii] = arr[0]
      for ii in range(shift, shape[0]):
         sh_arr[ii] = arr[ii-shift]
   elif shift < 0:
      for ii in range(shape[0] + shift):
         sh_arr[ii] = arr[ii-shift]
      for ii in range(shape[0] + shift, shape[0]):
         sh_arr[ii] = arr[-1]
         
   return sh_arr

@njit(cache = True, fastmath = True)
def shift_2D(arr, shift, axis):
   # Shift 2D array
   shape = arr.shape
   sh_arr = np.zeros(shape, dtype = np.float64)
   if shift > 0:
      if axis == 0:
         for jj in range(shift):
            for ii in range(shape[1]):
               sh_arr[jj,ii] = arr[0,ii]
         for jj in range(shift, shape[0]):
            for ii in range(shape[1]):
               sh_arr[jj,ii] = arr[jj-shift,ii]
      elif axis == 1:
         for jj in range(shape[0]):
            for ii in range(shift):
               sh_arr[jj,ii] = arr[jj,0]
            for ii in range(shift, shape[1]):
               sh_arr[jj,ii] = arr[jj,ii-shift]
   elif shift < 0:
      if axis == 0:
         for jj in range(shape[0] + shift):
            for ii in range(shape[1]):
               sh_arr[jj,ii] = arr[jj-shift,ii]
         for jj in range(shape[0] + shift, shape[0]):
            for ii in range(shape[1]):
               sh_arr[jj,ii] = arr[-1,ii]
      elif axis == 1:
         for jj in range(shape[0]):
            for ii in range(shape[1] + shift):
               sh_arr[jj,ii] = arr[jj,ii-shift]
            for ii in range(shape[1] + shift, shape[1]):
               sh_arr[jj,ii] = arr[jj,-1]
   
   return sh_arr

@njit(cache = True, fastmath = True)
def shift_3D(arr, shift, axis):
   # Shift 3D array
   shape = arr.shape
   sh_arr = np.zeros(shape, dtype = np.float64)
   if shift > 0:
      if axis == 0:
         for kk in range(shift):
            for jj in range(shape[1]):
               for ii in range(shape[2]):
                  sh_arr[kk,jj,ii] = arr[0,jj,ii]
         for kk in range(shift, shape[0]):
            for jj in range(shape[1]):
               for ii in range(shape[2]):
                  sh_arr[kk,jj,ii] = arr[kk-shift,jj,ii]
      elif axis == 1:
         for kk in range(shape[0]):
            for jj in range(shift):
               for ii in range(shape[2]):
                  sh_arr[kk,jj,ii] = arr[kk,0,ii]
            for jj in range(shift, shape[1]):
               for ii in range(shape[2]):
                  sh_arr[kk,jj,ii] = arr[kk,jj-shift,ii]
      elif axis == 2:
         for kk in range(shape[0]):
            for jj in range(shape[1]):
               for ii in range(shift):
                  sh_arr[kk,jj,ii] = arr[kk,jj,0]
               for ii in range(shift, shape[2]):
                  sh_arr[kk,jj,ii] = arr[kk,jj,ii-shift]
   elif shift < 0:
      if axis == 0:
         for kk in range(shape[0] + shift):
            for jj in range(shape[1]):
               for ii in range(shape[2]):
                  sh_arr[kk,jj,ii] = arr[kk-shift,jj,ii]
         for kk in range(shape[0] + shift, shape[0]):
            for jj in range(shape[1]):
               for ii in range(shape[2]):
                  sh_arr[kk,jj,ii] = arr[-1,jj,ii]
      elif axis == 1:
         for kk in range(shape[0]):
            for jj in range(shape[1] + shift):
               for ii in range(shape[2]):
                  sh_arr[kk,jj,ii] = arr[kk,jj-shift,ii]
            for jj in range(shape[1] + shift, shape[1]):
               for ii in range(shape[2]):
                  sh_arr[kk,jj,ii] = arr[kk,-1,ii]
      elif axis == 2:
         for kk in range(shape[0]):
            for jj in range(shape[1]):
               for ii in range(shape[2] + shift):
                  sh_arr[kk,jj,ii] = arr[kk,jj,ii-shift]
               for ii in range(shape[2] + shift, shape[2]):
                  sh_arr[kk,jj,ii] = arr[kk,jj,-1]

   return sh_arr

@njit(cache = True, fastmath = True)
def shift_4D(arr, shift, axis):
   # Shift 4D array
   shape = arr.shape
   sh_arr = np.zeros(shape, dtype = np.float64)
   if shift > 0:
      if axis == 0:
         for kk in range(shift):
            for jj in range(shape[1]):
               for ii in range(shape[2]):
                  for nn in range(shape[3]):
                     sh_arr[kk,jj,ii,nn] = arr[0,jj,ii,nn]
         for kk in range(shift, shape[0]):
            for jj in range(shape[1]):
               for ii in range(shape[2]):
                  for nn in range(shape[3]):
                     sh_arr[kk,jj,ii,nn] = arr[kk-shift,jj,ii,nn]
      elif axis == 1:
         for kk in range(shape[0]):
            for jj in range(shift):
               for ii in range(shape[2]):
                  for nn in range(shape[3]):
                     sh_arr[kk,jj,ii,nn] = arr[kk,0,ii,nn]
            for jj in range(shift, shape[1]):
               for ii in range(shape[2]):
                  for nn in range(shape[3]):
                     sh_arr[kk,jj,ii,nn] = arr[kk,jj-shift,ii,nn]
      elif axis == 2:
         for kk in range(shape[0]):
            for jj in range(shape[1]):
               for ii in range(shift):
                  for nn in range(shape[3]):
                     sh_arr[kk,jj,ii,nn] = arr[kk,jj,0,nn]
               for ii in range(shift, shape[2]):
                  for nn in range(shape[3]):
                     sh_arr[kk,jj,ii,nn] = arr[kk,jj,ii-shift,nn]
      elif axis == 3:
         for kk in range(shape[0]):
            for jj in range(shape[1]):
               for ii in range(shape[2]):
                  for nn in range(shift):
                     sh_arr[kk,jj,ii,nn] = arr[kk,jj,ii,0]
                  for nn in range(shift, shape[3]):
                     sh_arr[kk,jj,ii,nn] = arr[kk,jj,ii,nn-shift]
   elif shift < 0:
      if axis == 0:
         for kk in range(shape[0] + shift):
            for jj in range(shape[1]):
               for ii in range(shape[2]):
                  for nn in range(shape[3]):
                     sh_arr[kk,jj,ii,nn] = arr[kk-shift,jj,ii,nn]
         for kk in range(shape[0] + shift, shape[0]):
            for jj in range(shape[1]):
               for ii in range(shape[2]):
                  for nn in range(shape[3]):
                     sh_arr[kk,jj,ii,nn] = arr[-1,jj,ii,nn]
      elif axis == 1:
         for kk in range(shape[0]):
            for jj in range(shape[1] + shift):
               for ii in range(shape[2]):
                  for nn in range(shape[3]):
                     sh_arr[kk,jj,ii,nn] = arr[kk,jj-shift,ii,nn]
            for jj in range(shape[1] + shift, shape[1]):
               for ii in range(shape[2]):
                  for nn in range(shape[3]):
                     sh_arr[kk,jj,ii,nn] = arr[kk,-1,ii,nn]
      elif axis == 2:
         for kk in range(shape[0]):
            for jj in range(shape[1]):
               for ii in range(shape[2] + shift):
                  for nn in range(shape[3]):
                     sh_arr[kk,jj,ii,nn] = arr[kk,jj,ii-shift,nn]
               for ii in range(shape[2] + shift, shape[2]):
                  for nn in range(shape[3]):
                     sh_arr[kk,jj,ii,nn] = arr[kk,jj,-1,nn]
      elif axis == 3:
         for kk in range(shape[0]):
            for jj in range(shape[1]):
               for ii in range(shape[2]):
                  for nn in range(shape[3] + shift):
                     sh_arr[kk,jj,ii,nn] = arr[kk,jj,ii,nn-shift]
                  for nn in range(shape[3] + shift, shape[3]):
                     sh_arr[kk,jj,ii,nn] = arr[kk,jj,ii,-1]
   
   return sh_arr

@njit(cache = True, fastmath = True)
def roll_array(arr, shift, axis):
   # Roll super-function that selects correct version of roll for given array dimensions
   ndim = arr.ndim
   if ndim == 1:
      sh_arr = np.roll(arr, shift)
   elif ndim == 2:
      sh_arr = roll_2D(arr, shift, axis)
   elif ndim == 3:
      sh_arr = roll_3D(arr, shift, axis)
   elif ndim == 4:
      sh_arr = roll_4D(arr, shift, axis)

   return sh_arr

@njit(cache = True, fastmath = True)
def roll_2D(arr, shift, axis):
   # numba-friendly array axis rolling for 2D data
   shape = arr.shape

   if axis == 0:
      sh_arr = np.roll(arr, arr[0].size*shift)
   elif axis == 1:
      sh_arr = np.zeros(shape, dtype = np.float64)
      for jj in range(shape[0]):
         tmp = np.roll(arr[jj], shift)
         for ii in range(shape[1]):
            sh_arr[jj,ii] = tmp[ii]
   
   return sh_arr

@njit(cache = True, fastmath = True)
def roll_3D(arr, shift, axis):
   # numba-friendly array axis rolling for 3D data
   shape = arr.shape

   if axis == 0:
      sh_arr = np.roll(arr, arr[0].size*shift)
   elif axis == 1:
      sh_arr = np.zeros(shape, dtype = np.float64)
      for kk in range(shape[0]):
         tmp = np.roll(arr[kk], arr[kk,0].size*shift)
         for jj in range(shape[1]):
            for ii in range(shape[2]):
               sh_arr[kk,jj,ii] = tmp[jj,ii]
   elif axis == 2:
      sh_arr = np.zeros(shape, dtype = np.float64)
      for kk in range(shape[0]):
         for jj in range(shape[1]):
            tmp = np.roll(arr[kk,jj], shift)
            for ii in range(shape[2]):
               sh_arr[kk,jj,ii] = tmp[ii]
   
   return sh_arr

@njit(cache = True, fastmath = True)
def roll_4D(arr, shift, axis):
   # numba-friendly array axis rolling for 4D data
   shape = arr.shape
   
   if axis == 0:
      sh_arr = np.roll(arr, arr[0].size*shift)
   elif axis == 1:
      sh_arr = np.zeros(shape, dtype = np.float64)
      for kk in range(shape[0]):
         tmp = np.roll(arr[kk], arr[kk,0].size*shift)
         for jj in range(shape[1]):
            for ii in range(shape[2]):
               for nn in range(shape[3]):
                  sh_arr[kk,jj,ii,nn] = tmp[jj,ii,nn]
   elif axis == 2:
      sh_arr = np.zeros(shape, dtype = np.float64)
      for kk in range(shape[0]):
         for jj in range(shape[1]):
            tmp = np.roll(arr[kk,jj], arr[kk,jj,0].size*shift)
            for ii in range(shape[2]):
               for nn in range(shape[3]):
                  sh_arr[kk,jj,ii,nn] = tmp[ii,nn]
   elif axis == 3:
      sh_arr = np.zeros(shape, dtype = np.float64)
      for kk in range(shape[0]):
         for jj in range(shape[1]):
            for ii in range(shape[2]):
               tmp = np.roll(arr[kk,jj,ii], shift)
               for nn in range(shape[3]):
                  sh_arr[kk,jj,ii,nn] = tmp[nn]      
   
   return sh_arr

@njit(cache = True, fastmath = True)
def numba_unravel_index(index, dims):
   # Equivalent of np.unravel_index, compatible with numba
   ndim = len(dims)
   if ndim == 2:
      xi = index % dims[1]
      yi = index // dims[1]
      return np.array([yi,xi], dtype = int64)
   elif ndim == 3:
      xi = index % dims[2]
      yi = (index // dims[2]) % dims[1]
      zi = index // (dims[1] * dims[2])
      return np.array([zi,yi,xi], dtype = int64)
   elif ndim == 4:
      xi = index % dims[3]
      yi = (index // dims[3]) % dims[2]
      zi = (index // (dims[2] * dims[3])) % dims[1]
      wi = index // (dims[1] * dims[2] * dims[3])
      return np.array([wi,zi,yi,xi], dtype = int64)

@njit(cache = True, fastmath = True)
def numba_ravel_multi_index(indices, dims):
   # Equivalent of np.ravel_multi_index, compatible with numba
   ndim = len(dims)
   out = np.zeros(indices.shape[0], dtype = int64)
   out += indices[:,0]
   for ii in range(1,ndim):
      out *= dims[ii]
      out += indices[:,ii]
   
   return out

def floatToStr(value, decimals = 0, pad = None):
   # Converts float to string with given number of decimals (rounded)
   # decimals must be an integer
   # If pad is a positive integer then zeros are padded to front to match number before decimal
   if not isinstance(decimals, int):
      raise TypeError("Number of decimals must be integer")
   if pad is not None and not isinstance(pad, int):
      raise TypeError("Number of decimals must be integer")
   formatter = "{0:0"
   if pad is not None:
      total = pad + decimals + 1
      formatter += str(total)
   formatter += "." + str(decimals) + "f}"
   return formatter.format(value)

def get_particle_cellid(r, dims):
   # Get the cellid(s) of the cell containing the given particle(s)
   x_ind = np.floor((r[:,0] - dims.x_min)/dims.dx).astype(int)
   y_ind = np.floor((r[:,1] - dims.y_min)/dims.dy).astype(int)
   z_ind = np.floor((r[:,2] - dims.z_min)/dims.dz).astype(int)

   IDs = np.ravel_multi_index((z_ind,y_ind,x_ind), dims.dim_scalar)
   
   return IDs

@njit(cache = True, fastmath = True)
def get_particle_cellid_njit(r, dims):
   # Get the cellid of the cell containing position r
   x_ind = math.floor((r[0] - dims.x_min)/dims.dx)
   y_ind = math.floor((r[1] - dims.y_min)/dims.dy)
   z_ind = math.floor((r[2] - dims.z_min)/dims.dz)

   ID = (z_ind*dims.y_size + y_ind)*dims.x_size + x_ind
   
   return ID

def CIC_weights_node(r, dims, reshape = True, fraction = True):
   # Calculate CIC weights and interpolation nodes for given population
   # CIC weight is fractional volume of particle cloud intersecting
   # with each given node 'volume'
   # here volume means cell-sized box centred on node
   # Assumes grid is node (or face)
   # If reshape is True then reshapes x_w etc. so that x_w*y_w*z_w works
   # If fraction is True then x_w etc. returned as fraction of distance
   # between points, else returns real distance from point
   x_ind0 = np.floor((r[:,0] - dims.x_min)/dims.dx).astype(int)
   y_ind0 = np.floor((r[:,1] - dims.y_min)/dims.dy).astype(int)
   z_ind0 = np.floor((r[:,2] - dims.z_min)/dims.dz).astype(int)
   
   x0 = x_ind0*dims.dx + dims.x_min
   y0 = y_ind0*dims.dy + dims.y_min
   z0 = z_ind0*dims.dz + dims.z_min

   if fraction:
      x_w1 = (r[:,0] - x0)/dims.dx
      y_w1 = (r[:,1] - y0)/dims.dy
      z_w1 = (r[:,2] - z0)/dims.dz
      
      x_w0 = 1 - x_w1
      y_w0 = 1 - y_w1
      z_w0 = 1 - z_w1
   else:
      x_w1 = r[:,0] - x0
      y_w1 = r[:,1] - y0
      z_w1 = r[:,2] - z0
      
      x_w0 = dims.dx - x_w1
      y_w0 = dims.dy - y_w1
      z_w0 = dims.dz - z_w1
   
   x_ind1 = x_ind0 + 1
   y_ind1 = y_ind0 + 1
   z_ind1 = z_ind0 + 1
   
   x_ind = np.array((x_ind0,x_ind1))
   y_ind = np.array((y_ind0,y_ind1))
   z_ind = np.array((z_ind0,z_ind1))

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
   
   # x_ind1 = dims.x_range_l2r[x_ind0]
   # y_ind1 = dims.y_range_l2r[y_ind0]
   # z_ind1 = dims.z_range_l2r[z_ind0]
   
   # x_ind = np.array((x_ind0,x_ind1))
   # y_ind = np.array((y_ind0,y_ind1))
   # z_ind = np.array((z_ind0,z_ind1))
   
   x_w = np.array((x_w0,x_w1))
   y_w = np.array((y_w0,y_w1))
   z_w = np.array((z_w0,z_w1))
   
   if reshape:
      x_w = x_w.reshape(1,1,2,-1)
      y_w = y_w.reshape(1,2,1,-1)
      z_w = z_w.reshape(2,1,1,-1)

   return (x_ind,y_ind,z_ind),(x_w,y_w,z_w)

def CIC_weights_cell(r, dims, reshape = True, fraction = True):
   # Calculate CIC weights and interpolation cells for given population
   # CIC weight is fractional volume of particle cloud intersecting
   # with each given cell volume
   # Assumes grid is cell
   # If reshape is True then reshapes x_w etc. so that x_w*y_w*z_w works
   # If fraction is True then x_w etc. returned as fraction of distance
   # between points, else returns real distance from point
   x_ind1 = np.round((r[:,0] - dims.x_min)/dims.dx).astype(int)
   y_ind1 = np.round((r[:,1] - dims.y_min)/dims.dy).astype(int)
   z_ind1 = np.round((r[:,2] - dims.z_min)/dims.dz).astype(int)

   x0 = (x_ind1 - 1 + 0.5)*dims.dx + dims.x_min
   y0 = (y_ind1 - 1 + 0.5)*dims.dy + dims.y_min
   z0 = (z_ind1 - 1 + 0.5)*dims.dz + dims.z_min
   if fraction:
      x_w1 = (r[:,0] - x0)/dims.dx
      y_w1 = (r[:,1] - y0)/dims.dy
      z_w1 = (r[:,2] - z0)/dims.dz
      
      x_w0 = 1 - x_w1
      y_w0 = 1 - y_w1
      z_w0 = 1 - z_w1
   else:
      x_w1 = r[:,0] - x0
      y_w1 = r[:,1] - y0
      z_w1 = r[:,2] - z0
      
      x_w0 = dims.dx - x_w1
      y_w0 = dims.dy - y_w1
      z_w0 = dims.dz - z_w1

   x_ind0 = x_ind1 - 1
   y_ind0 = y_ind1 - 1
   z_ind0 = z_ind1 - 1

   x_ind = np.array((x_ind0,x_ind1))
   y_ind = np.array((y_ind0,y_ind1))
   z_ind = np.array((z_ind0,z_ind1))

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
   
   x_w = np.array((x_w0,x_w1))
   y_w = np.array((y_w0,y_w1))
   z_w = np.array((z_w0,z_w1))
   
   if reshape:
      x_w = x_w.reshape(1,1,2,-1)
      y_w = y_w.reshape(1,2,1,-1)
      z_w = z_w.reshape(2,1,1,-1)
   
   return (x_ind,y_ind,z_ind),(x_w,y_w,z_w)

@njit(cache = True, fastmath = True)
def CIC_weights_node_1D_njit(r, dims, fraction = True):
   # Calculate CIC weights and interpolation nodes for given population
   # CIC weight is fractional volume of particle cloud intersecting
   # with each given node 'volume'
   # here volume means cell-sized box centred on node
   # Assumes grid is node (or face)
   # If reshape is True then reshapes x_w etc. so that x_w*y_w*z_w works
   # If fraction is True then x_w etc. returned as fraction of distance
   # between points, else returns real distance from point
   x_ind0 = math.floor((r[0] - dims.x_min)/dims.dx)
   
   x0 = x_ind0*dims.dx + dims.x_min
   
   if fraction:
      x_w1 = (r[0] - x0)/dims.dx
      
      x_w0 = 1 - x_w1
   else:
      x_w1 = r[0] - x0
      
      x_w0 = dims.dx - x_w1
   
   x_ind1 = x_ind0 + 1
   
   x_ind = np.array((x_ind0,x_ind1))
   
   if dims.period[2]:
      x_ind = np.mod(x_ind, dims.x_size)
   else:
      x_ind = numba_clip(x_ind, 0, dims.x_size - 1)
   
   x_w = np.array((x_w0,x_w1))
   
   return x_ind,x_w

@njit(cache = True, fastmath = True)
def CIC_weights_cell_1D_njit(r, dims, fraction = True):
   # Calculate CIC weights and interpolation cells for given population
   # CIC weight is fractional volume of particle cloud intersecting
   # with each given cell volume
   # Assumes grid is cell
   # If fraction is True then x_w etc. returned as fraction of distance
   # between points, else returns real distance from point
   x_ind1 = round((r[0] - dims.x_min)/dims.dx)
   
   x0 = (x_ind1 - 1 + 0.5)*dims.dx + dims.x_min
   if fraction:
      x_w1 = (r[0] - x0)/dims.dx
      
      x_w0 = 1 - x_w1
   else:
      x_w1 = r[0] - x0
      
      x_w0 = dims.dx - x_w1
   
   x_ind0 = x_ind1 - 1
   
   x_ind = np.array((x_ind0,x_ind1))
   
   if dims.period[2]:
      x_ind = np.mod(x_ind, dims.x_size)
   else:
      x_ind = numba_clip(x_ind, 0, dims.x_size - 1)
   
   x_w = np.array((x_w0,x_w1))
   
   return x_ind,x_w

@njit(cache = True, fastmath = True)
def CIC_weights_node_3D_njit(r, dims, fraction = True):
   # Calculate CIC weights and interpolation nodes for given population
   # CIC weight is fractional volume of particle cloud intersecting
   # with each given node 'volume'
   # here volume means cell-sized box centred on node
   # Assumes grid is node (or face)
   # If reshape is True then reshapes x_w etc. so that x_w*y_w*z_w works
   # If fraction is True then x_w etc. returned as fraction of distance
   # between points, else returns real distance from point
   x_ind0 = math.floor((r[0] - dims.x_min)/dims.dx)
   y_ind0 = math.floor((r[1] - dims.y_min)/dims.dy)
   z_ind0 = math.floor((r[2] - dims.z_min)/dims.dz)
   
   x0 = x_ind0*dims.dx + dims.x_min
   y0 = y_ind0*dims.dy + dims.y_min
   z0 = z_ind0*dims.dz + dims.z_min

   if fraction:
      x_w1 = (r[0] - x0)/dims.dx
      y_w1 = (r[1] - y0)/dims.dy
      z_w1 = (r[2] - z0)/dims.dz
      
      x_w0 = 1 - x_w1
      y_w0 = 1 - y_w1
      z_w0 = 1 - z_w1
   else:
      x_w1 = r[0] - x0
      y_w1 = r[1] - y0
      z_w1 = r[2] - z0
      
      x_w0 = dims.dx - x_w1
      y_w0 = dims.dy - y_w1
      z_w0 = dims.dz - z_w1
   
   x_ind1 = x_ind0 + 1
   y_ind1 = y_ind0 + 1
   z_ind1 = z_ind0 + 1
   
   x_ind = np.array((x_ind0,x_ind1))
   y_ind = np.array((y_ind0,y_ind1))
   z_ind = np.array((z_ind0,z_ind1))

   if dims.period[2]:
      x_ind = np.mod(x_ind, dims.x_size)
   else:
      x_ind = numba_clip(x_ind, 0, dims.x_size - 1)
   
   if dims.period[1]:
      y_ind = np.mod(y_ind, dims.y_size)
   else:
      y_ind = numba_clip(y_ind, 0, dims.y_size - 1)
   
   if dims.period[0]:
      z_ind = np.mod(z_ind, dims.z_size)
   else:
      z_ind = numba_clip(z_ind, 0, dims.z_size - 1)
   
   x_w = np.array((x_w0,x_w1))
   y_w = np.array((y_w0,y_w1))
   z_w = np.array((z_w0,z_w1))

   x_w = x_w.reshape(1,1,2)
   y_w = y_w.reshape(1,2,1)
   z_w = z_w.reshape(2,1,1)
   
   weights = x_w*y_w*z_w
   
   ind = np.stack((x_ind,y_ind,z_ind))
   
   return ind,weights

@njit(cache = True, fastmath = True)
def CIC_weights_cell_3D_njit(r, dims, fraction = True):
   # Calculate CIC weights and interpolation cells for given population
   # CIC weight is fractional volume of particle cloud intersecting
   # with each given cell volume
   # Assumes grid is cell
   # If reshape is True then reshapes x_w etc. so that x_w*y_w*z_w works
   # If fraction is True then x_w etc. returned as fraction of distance
   # between points, else returns real distance from point
   x_ind1 = round((r[0] - dims.x_min)/dims.dx)
   y_ind1 = round((r[1] - dims.y_min)/dims.dy)
   z_ind1 = round((r[2] - dims.z_min)/dims.dz)
   
   x0 = (x_ind1 - 1 + 0.5)*dims.dx + dims.x_min
   y0 = (y_ind1 - 1 + 0.5)*dims.dy + dims.y_min
   z0 = (z_ind1 - 1 + 0.5)*dims.dz + dims.z_min
   if fraction:
      x_w1 = (r[0] - x0)/dims.dx
      y_w1 = (r[1] - y0)/dims.dy
      z_w1 = (r[2] - z0)/dims.dz
      
      x_w0 = 1 - x_w1
      y_w0 = 1 - y_w1
      z_w0 = 1 - z_w1
   else:
      x_w1 = r[0] - x0
      y_w1 = r[1] - y0
      z_w1 = r[2] - z0
      
      x_w0 = dims.dx - x_w1
      y_w0 = dims.dy - y_w1
      z_w0 = dims.dz - z_w1

   x_ind0 = x_ind1 - 1
   y_ind0 = y_ind1 - 1
   z_ind0 = z_ind1 - 1

   x_ind = np.array((x_ind0,x_ind1))
   y_ind = np.array((y_ind0,y_ind1))
   z_ind = np.array((z_ind0,z_ind1))

   if dims.period[2]:
      x_ind = np.mod(x_ind, dims.x_size)
   else:
      x_ind = numba_clip(x_ind, 0, dims.x_size - 1)
   
   if dims.period[1]:
      y_ind = np.mod(y_ind, dims.y_size)
   else:
      y_ind = numba_clip(y_ind, 0, dims.y_size - 1)
   
   if dims.period[0]:
      z_ind = np.mod(z_ind, dims.z_size)
   else:
      z_ind = numba_clip(z_ind, 0, dims.z_size - 1)
   
   x_w = np.array((x_w0,x_w1))
   y_w = np.array((y_w0,y_w1))
   z_w = np.array((z_w0,z_w1))

   x_w = x_w.reshape(1,1,2)
   y_w = y_w.reshape(1,2,1)
   z_w = z_w.reshape(2,1,1)
   
   weights = x_w*y_w*z_w
   
   ind = np.stack((x_ind,y_ind,z_ind))
   
   return ind,weights
