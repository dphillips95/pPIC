# Module of general tools, e.g. array rolling/sliding

import numpy as np
from numba import njit,int64

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
