import math
import numpy as np
from numba import njit
from numba import int64,float64

from tools import arr_shift,shift_indices,split_axis

def face2cell(face_data, period):
   # Interpolates data from cell faces to centre
   # Data is assumed to be vector data
   xface_data,yface_data,zface_data = split_axis(face_data, axis = 3)
   
   cell_data = face_data.copy()
   
   cell_data[:,:,:,0] += arr_shift(xface_data, -1, 2, period)
   cell_data[:,:,:,1] += arr_shift(yface_data, -1, 1, period)
   cell_data[:,:,:,2] += arr_shift(zface_data, -1, 0, period)
   
   cell_data *= 0.5
   
   return cell_data

@njit(cache = True, fastmath = True)
def face2cell_njit(face_data, period):
   # Interpolates data from cell faces to centre
   # Data is assumed to be vector data
   cell_data = face_data.copy()

   shape = face_data.shape
   
   shift_x = arr_shift(face_data[:,:,:,0], -1, 2, period)
   shift_y = arr_shift(face_data[:,:,:,1], -1, 1, period)
   shift_z = arr_shift(face_data[:,:,:,2], -1, 0, period)
   for kk in range(shape[0]):
      for jj in range(shape[1]):
         for ii in range(shape[2]):
            cell_data[kk,jj,ii,0] += shift_x[kk,jj,ii]
            cell_data[kk,jj,ii,1] += shift_y[kk,jj,ii]
            cell_data[kk,jj,ii,2] += shift_z[kk,jj,ii]
   
   cell_data *= 0.5
   
   return cell_data

def cell2node(cell_data, period):
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

@njit(cache = True, fastmath = True)
def cell2node_njit(cell_data, period):
   # Interpolates data from cell centres to nodes
   node_data = cell_data.copy()
   
   shift_0 = arr_shift(cell_data, 1, 0, period)
   shift_1 = arr_shift(cell_data, 1, 1, period)
   shift_2 = arr_shift(cell_data, 1, 2, period)
   shift_01 = arr_shift(shift_0, 1, 1, period)
   shift_02 = arr_shift(shift_2, 1, 0, period)
   shift_12 = arr_shift(shift_1, 1, 2, period)
   shift_012 = arr_shift(shift_01, 1, 2, period)
   
   shape = cell_data.shape

   if cell_data.ndim == 3:
      for kk in range(shape[0]):
         for jj in range(shape[1]):
            for ii in range(shape[2]):
               node_data[kk,jj,ii] += shift_0[kk,jj,ii]
               node_data[kk,jj,ii] += shift_1[kk,jj,ii]
               node_data[kk,jj,ii] += shift_2[kk,jj,ii]
               node_data[kk,jj,ii] += shift_01[kk,jj,ii]
               node_data[kk,jj,ii] += shift_02[kk,jj,ii]
               node_data[kk,jj,ii] += shift_12[kk,jj,ii]
               node_data[kk,jj,ii] += shift_012[kk,jj,ii]
   elif cell_data.ndim == 4:
      for kk in range(shape[0]):
         for jj in range(shape[1]):
            for ii in range(shape[2]):
               for nn in range(3):
                  node_data[kk,jj,ii,nn] += shift_0[kk,jj,ii,nn]
                  node_data[kk,jj,ii,nn] += shift_1[kk,jj,ii,nn]
                  node_data[kk,jj,ii,nn] += shift_2[kk,jj,ii,nn]
                  node_data[kk,jj,ii,nn] += shift_01[kk,jj,ii,nn]
                  node_data[kk,jj,ii,nn] += shift_02[kk,jj,ii,nn]
                  node_data[kk,jj,ii,nn] += shift_12[kk,jj,ii,nn]
                  node_data[kk,jj,ii,nn] += shift_012[kk,jj,ii,nn]
   
   node_data *= 0.125

   return node_data

def face2node(face_data, period):
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

@njit(cache = True, fastmath = True)
def face2node_njit(face_data, period):
   # Interpolate face data to node
   # Data is assumed to be vector data, face data only stores one component each
   node_data = face_data.copy()

   shape = face_data.shape
   
   shift_0 = arr_shift(face_data[:,:,:,0], 1, 0, period)
   shift_1 = arr_shift(face_data[:,:,:,0], 1, 1, period)
   shift_01 = arr_shift(shift_0, 1, 1, period)

   for kk in range(shape[0]):
      for jj in range(shape[1]):
         for ii in range(shape[2]):   
            node_data[kk,jj,ii,0] += shift_0[kk,jj,ii]
            node_data[kk,jj,ii,0] += shift_1[kk,jj,ii]
            node_data[kk,jj,ii,0] += shift_01[kk,jj,ii]
   
   shift_0 = arr_shift(face_data[:,:,:,1], 1, 0, period)
   shift_2 = arr_shift(face_data[:,:,:,1], 1, 2, period)
   shift_02 = arr_shift(shift_0, 1, 2, period)

   for kk in range(shape[0]):
      for jj in range(shape[1]):
         for ii in range(shape[2]):   
            node_data[kk,jj,ii,1] += shift_0[kk,jj,ii]
            node_data[kk,jj,ii,1] += shift_2[kk,jj,ii]
            node_data[kk,jj,ii,1] += shift_02[kk,jj,ii]

   shift_1 = arr_shift(face_data[:,:,:,2], 1, 1, period)
   shift_2 = arr_shift(face_data[:,:,:,2], 1, 2, period)
   shift_12 = arr_shift(shift_1, 1, 2, period)

   for kk in range(shape[0]):
      for jj in range(shape[1]):
         for ii in range(shape[2]):   
            node_data[kk,jj,ii,2] += shift_1[kk,jj,ii]
            node_data[kk,jj,ii,2] += shift_2[kk,jj,ii]
            node_data[kk,jj,ii,2] += shift_12[kk,jj,ii]
   
   node_data *= 0.25
   
   return node_data
   
def node2cell(node_data, period):
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

@njit(cache = True, fastmath = True)
def node2cell_njit(node_data, period):
   # Interpolate node data to cell centres
   cell_data = node_data.copy()

   shape = node_data.shape
   
   shift_0 = arr_shift(node_data, -1, 0, period)
   shift_1 = arr_shift(node_data, -1, 1, period)
   shift_2 = arr_shift(node_data, -1, 2, period)
   shift_01 = arr_shift(shift_0, -1, 1, period)
   shift_02 = arr_shift(shift_2, -1, 0, period)
   shift_12 = arr_shift(shift_1, -1, 2, period)
   shift_012 = arr_shift(shift_01, -1, 2, period)

   if node_data.ndim == 3:
      for kk in range(shape[0]):
         for jj in range(shape[1]):
            for ii in range(shape[2]):
               cell_data[kk,jj,ii] += shift_0[kk,jj,ii]
               cell_data[kk,jj,ii] += shift_1[kk,jj,ii]
               cell_data[kk,jj,ii] += shift_2[kk,jj,ii]
               cell_data[kk,jj,ii] += shift_01[kk,jj,ii]
               cell_data[kk,jj,ii] += shift_02[kk,jj,ii]
               cell_data[kk,jj,ii] += shift_12[kk,jj,ii]
               cell_data[kk,jj,ii] += shift_012[kk,jj,ii]
   elif node_data.ndim == 4:
      for kk in range(shape[0]):
         for jj in range(shape[1]):
            for ii in range(shape[2]):
               for nn in range(3):
                  cell_data[kk,jj,ii,nn] += shift_0[kk,jj,ii,nn]
                  cell_data[kk,jj,ii,nn] += shift_1[kk,jj,ii,nn]
                  cell_data[kk,jj,ii,nn] += shift_2[kk,jj,ii,nn]
                  cell_data[kk,jj,ii,nn] += shift_01[kk,jj,ii,nn]
                  cell_data[kk,jj,ii,nn] += shift_02[kk,jj,ii,nn]
                  cell_data[kk,jj,ii,nn] += shift_12[kk,jj,ii,nn]
                  cell_data[kk,jj,ii,nn] += shift_012[kk,jj,ii,nn]
   
   cell_data *= 0.125
   
   return cell_data
   
def node2face(node_data, period):
   # Interpolate node data to face
   # Data is assumed to be vector data, face data only stores one component each
   face_data = node_data.copy()
   
   xnode_data,ynode_data,znode_data = split_axis(node_data, axis = 3)
   
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

@njit(cache = True, fastmath = True)
def node2face_njit(node_data, period):
   # Interpolate node data to face
   # Data is assumed to be vector data, face data only stores one component each
   face_data = node_data.copy()

   shape = node_data.shape
   
   shift_0 = arr_shift(node_data[:,:,:,0], -1, 0, period)
   shift_1 = arr_shift(node_data[:,:,:,0], -1, 1, period)
   shift_01 = arr_shift(shift_0, -1, 1, period)

   for kk in range(shape[0]):
      for jj in range(shape[1]):
         for ii in range(shape[2]):
            face_data[kk,jj,ii,0] += shift_0[kk,jj,ii]
            face_data[kk,jj,ii,0] += shift_1[kk,jj,ii]
            face_data[kk,jj,ii,0] += shift_01[kk,jj,ii]

   shift_0 = arr_shift(node_data[:,:,:,1], -1, 0, period)
   shift_2 = arr_shift(node_data[:,:,:,1], -1, 2, period)
   shift_02 = arr_shift(shift_0, -1, 2, period)
   
   for kk in range(shape[0]):
      for jj in range(shape[1]):
         for ii in range(shape[2]):
            face_data[kk,jj,ii,1] += shift_0[kk,jj,ii]
            face_data[kk,jj,ii,1] += shift_2[kk,jj,ii]
            face_data[kk,jj,ii,1] += shift_02[kk,jj,ii]

   shift_1 = arr_shift(node_data[:,:,:,2], -1, 1, period)
   shift_2 = arr_shift(node_data[:,:,:,2], -1, 2, period)
   shift_12 = arr_shift(shift_1, -1, 2, period)
   
   for kk in range(shape[0]):
      for jj in range(shape[1]):
         for ii in range(shape[2]):
            face_data[kk,jj,ii,2] += shift_1[kk,jj,ii]
            face_data[kk,jj,ii,2] += shift_2[kk,jj,ii]
            face_data[kk,jj,ii,2] += shift_12[kk,jj,ii]
   
   face_data *= 0.25
   
   return face_data
   
def cell2face(cell_data, period):
   # Interpolates data from cell centres to faces
   # Data is assumed to be vector data, face data only stores one component each
   face_data = cell_data.copy()
   
   for ii in range(3):
      vector_comp = 2 - ii
      face_data[:,:,:,vector_comp] += arr_shift(cell_data[:,:,:,vector_comp], 1, ii, period)

   face_data *= 0.5
   
   return face_data

@njit(cache = True, fastmath = True)
def cell2face_njit(cell_data, period):
   # Interpolates data from cell centres to faces
   # Data is assumed to be vector data, face data only stores one component each
   face_data = cell_data.copy()

   shape = cell_data.shape
   
   for nn in range(3):
      shift_axis = 2 - nn
      tmp = arr_shift(cell_data[:,:,:,nn], 1, shift_axis, period)
      for kk in range(shape[0]):
         for jj in range(shape[1]):
            for ii in range(shape[2]):
               face_data[kk,jj,ii,nn] += tmp[kk,jj,ii]

   face_data *= 0.5
   
   return face_data
   
def face2r(face_data, r, period, dims):
   # Interpolate face data to arbitrary position(s) r
   # Data is assumed to be vector data, face data only stores one component each
   r = np.array(r).reshape(-1,3)
   r_data = np.zeros(r.shape)
   
   xface_data,yface_data,zface_data = split_axis(face_data, axis = 3)
   
   x_locs_l = np.floor((r[:,0] - dims.x_min)/dims.dx).astype(int)
   y_locs_l = np.floor((r[:,1] - dims.y_min)/dims.dy).astype(int)
   z_locs_l = np.floor((r[:,2] - dims.z_min)/dims.dz).astype(int)

   x0 = x_locs_l*dims.dx + dims.x_min
   y0 = y_locs_l*dims.dy + dims.y_min
   z0 = z_locs_l*dims.dz + dims.z_min
   x_w1 = (r[:,0] - x0)/dims.dx
   y_w1 = (r[:,1] - y0)/dims.dy
   z_w1 = (r[:,2] - z0)/dims.dz
   x_w0 = 1 - x_w1
   y_w0 = 1 - y_w1
   z_w0 = 1 - z_w1

   x_locs_r = shift_indices(dims.x_range, -1, period[2])[x_locs_l]
   y_locs_r = shift_indices(dims.y_range, -1, period[1])[y_locs_l]
   z_locs_r = shift_indices(dims.z_range, -1, period[0])[z_locs_l]
   
   r_data[:,0] += x_w0*xface_data[z_locs_l,y_locs_l,x_locs_l]
   r_data[:,0] += x_w1*xface_data[z_locs_l,y_locs_l,x_locs_r]

   r_data[:,1] += y_w0*yface_data[z_locs_l,y_locs_l,x_locs_l]
   r_data[:,1] += y_w1*yface_data[z_locs_l,y_locs_r,x_locs_l]

   r_data[:,2] += z_w0*zface_data[z_locs_l,y_locs_l,x_locs_l]
   r_data[:,2] += z_w1*zface_data[z_locs_r,y_locs_l,x_locs_l]

   if r.shape[0] == 1:
      r_data = r_data.reshape(3)
   
   return r_data

@njit(cache = True, fastmath = True)
def face2r_njit(face_data, r, period, dims):
   # Interpolate face data to arbitrary position(s) r
   # Data is assumed to be vector data, face data only stores one component each
   r = r.reshape(-1,3)
   
   p_count = r.shape[0]

   r_data = np.zeros((p_count,3))
   
   x_shift = shift_indices(dims.x_range, -1, period[2])
   y_shift = shift_indices(dims.y_range, -1, period[1])
   z_shift = shift_indices(dims.z_range, -1, period[0])
   
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
      
      x_locs_r = x_shift[x_locs_l]
      y_locs_r = y_shift[y_locs_l]
      z_locs_r = z_shift[z_locs_l]
      
      r_data[ii,0] += x_w0*face_data[z_locs_l,y_locs_l,x_locs_l,0]
      r_data[ii,0] += x_w1*face_data[z_locs_l,y_locs_l,x_locs_r,0]
      
      r_data[ii,1] += y_w0*face_data[z_locs_l,y_locs_l,x_locs_l,1]
      r_data[ii,1] += y_w1*face_data[z_locs_l,y_locs_r,x_locs_l,1]
      
      r_data[ii,2] += z_w0*face_data[z_locs_l,y_locs_l,x_locs_l,2]
      r_data[ii,2] += z_w1*face_data[z_locs_r,y_locs_l,x_locs_l,2]
   
   return r_data

def node2r(node_data, r, period, dims):
   # Interpolate node data to arbitrary position(s) r
   r = np.array(r).reshape(-1,3)
   vec = (node_data.ndim == 4)
   
   if vec:
      r_data = np.zeros(r.shape)
   else:
      r_data = np.zeros(r.shape[0])
   
   x_locs_l = np.floor((r[:,0] - dims.x_min)/dims.dx).astype(int)
   y_locs_l = np.floor((r[:,1] - dims.y_min)/dims.dy).astype(int)
   z_locs_l = np.floor((r[:,2] - dims.z_min)/dims.dz).astype(int)
   
   x0 = x_locs_l*dims.dx + dims.x_min
   y0 = y_locs_l*dims.dy + dims.y_min
   z0 = z_locs_l*dims.dz + dims.z_min
   x_w1 = (r[:,0] - x0)/dims.dx
   y_w1 = (r[:,1] - y0)/dims.dy
   z_w1 = (r[:,2] - z0)/dims.dz
   x_w0 = 1 - x_w1
   y_w0 = 1 - y_w1
   z_w0 = 1 - z_w1
   
   weights = np.array([[x_w0,y_w0,z_w0],[x_w1,y_w0,z_w0],
                          [x_w0,y_w1,z_w0],[x_w1,y_w1,z_w0],
                          [x_w0,y_w0,z_w1],[x_w1,y_w0,z_w1],
                          [x_w0,y_w1,z_w1],[x_w1,y_w1,z_w1]])
   weights = 1/np.linalg.norm(weights, axis = 1)
   weights /= np.sum(weights, axis = 0)
   
   if vec:
      weights = weights.reshape(8,-1,1)
   else:
      weights = weights.reshape(8,-1)
   
   # r_data += weights[0]*node_data[array_indices[0][z_locs_l+1],
   #                                array_indices[1][y_locs_l+1],
   #                                array_indices[2][x_locs_l+1],...]
   # r_data += weights[1]*node_data[array_indices[0][z_locs_l+2],
   #                                array_indices[1][y_locs_l+1],
   #                                array_indices[2][x_locs_l+1],...]
   # r_data += weights[2]*node_data[array_indices[0][z_locs_l+1],
   #                                array_indices[1][y_locs_l+2],
   #                                array_indices[2][x_locs_l+1],...]
   # r_data += weights[3]*node_data[array_indices[0][z_locs_l+2],
   #                                array_indices[1][y_locs_l+2],
   #                                array_indices[2][x_locs_l+1],...]
   # r_data += weights[4]*node_data[array_indices[0][z_locs_l+1],
   #                                array_indices[1][y_locs_l+1],
   #                                array_indices[2][x_locs_l+2],...]
   # r_data += weights[5]*node_data[array_indices[0][z_locs_l+2],
   #                                array_indices[1][y_locs_l+1],
   #                                array_indices[2][x_locs_l+2],...]
   # r_data += weights[6]*node_data[array_indices[0][z_locs_l+1],
   #                                array_indices[1][y_locs_l+2],
   #                                array_indices[2][x_locs_l+2],...]
   # r_data += weights[7]*node_data[array_indices[0][z_locs_l+2],
   #                                array_indices[1][y_locs_l+2],
   #                                array_indices[2][x_locs_l+2],...]

   x_locs_r = shift_indices(dims.x_range, -1, period[2])[x_locs_l]
   y_locs_r = shift_indices(dims.y_range, -1, period[1])[y_locs_l]
   z_locs_r = shift_indices(dims.z_range, -1, period[0])[z_locs_l]
   
   r_data += weights[0]*node_data[z_locs_l, y_locs_l, x_locs_l,...]
   r_data += weights[1]*node_data[z_locs_r, y_locs_l, x_locs_l,...]
   r_data += weights[2]*node_data[z_locs_l, y_locs_r, x_locs_l,...]
   r_data += weights[3]*node_data[z_locs_r, y_locs_r, x_locs_l,...]
   r_data += weights[4]*node_data[z_locs_l, y_locs_l, x_locs_r,...]
   r_data += weights[5]*node_data[z_locs_r, y_locs_l, x_locs_r,...]
   r_data += weights[6]*node_data[z_locs_l, y_locs_r, x_locs_r,...]
   r_data += weights[7]*node_data[z_locs_r, y_locs_r, x_locs_r,...]

   if r.shape[0] == 1:
      r_data = r_data.reshape(3)
   
   return r_data

@njit(cache = True, fastmath = True)
def node2r_njit(node_data, r, period, dims):
   # Interpolate node data to arbitrary position(s) r
   r = r.reshape(-1,3)
   vec = (node_data.ndim == 4)
   
   p_count = r.shape[0]

   if vec:
      r_data = np.zeros((p_count,3))
   else:
      r_data = np.zeros(p_count)

   x_shift = shift_indices(dims.x_range, -1, period[2])
   y_shift = shift_indices(dims.y_range, -1, period[1])
   z_shift = shift_indices(dims.z_range, -1, period[0])
      
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
      
      weights = np.array([[x_w0,y_w0,z_w0],[x_w1,y_w0,z_w0],
                          [x_w0,y_w1,z_w0],[x_w1,y_w1,z_w0],
                          [x_w0,y_w0,z_w1],[x_w1,y_w0,z_w1],
                          [x_w0,y_w1,z_w1],[x_w1,y_w1,z_w1]])
      weights = 1/np.sqrt((weights**2).sum(axis = 1))
      weights /= np.sum(weights, axis = 0)
      
      x_locs_r = x_shift[x_locs_l]
      y_locs_r = y_shift[y_locs_l]
      z_locs_r = z_shift[z_locs_l]

      if node_data.ndim == 3:
         r_data[ii] += weights[0]*node_data[z_locs_l, y_locs_l, x_locs_l]
         r_data[ii] += weights[1]*node_data[z_locs_r, y_locs_l, x_locs_l]
         r_data[ii] += weights[2]*node_data[z_locs_l, y_locs_r, x_locs_l]
         r_data[ii] += weights[3]*node_data[z_locs_r, y_locs_r, x_locs_l]
         r_data[ii] += weights[4]*node_data[z_locs_l, y_locs_l, x_locs_r]
         r_data[ii] += weights[5]*node_data[z_locs_r, y_locs_l, x_locs_r]
         r_data[ii] += weights[6]*node_data[z_locs_l, y_locs_r, x_locs_r]
         r_data[ii] += weights[7]*node_data[z_locs_r, y_locs_r, x_locs_r]
      elif node_data.ndim == 4:
         for nn in range(3):
            r_data[ii,nn] += weights[0]*node_data[z_locs_l, y_locs_l, x_locs_l,nn]
            r_data[ii,nn] += weights[1]*node_data[z_locs_r, y_locs_l, x_locs_l,nn]
            r_data[ii,nn] += weights[2]*node_data[z_locs_l, y_locs_r, x_locs_l,nn]
            r_data[ii,nn] += weights[3]*node_data[z_locs_r, y_locs_r, x_locs_l,nn]
            r_data[ii,nn] += weights[4]*node_data[z_locs_l, y_locs_l, x_locs_r,nn]
            r_data[ii,nn] += weights[5]*node_data[z_locs_r, y_locs_l, x_locs_r,nn]
            r_data[ii,nn] += weights[6]*node_data[z_locs_l, y_locs_r, x_locs_r,nn]
            r_data[ii,nn] += weights[7]*node_data[z_locs_r, y_locs_r, x_locs_r,nn]
   
   return r_data

def cell2r(cell_data, r, period, dims):
   # Interpolate cell data to arbitrary position(s) r
   r = np.array(r).reshape(-1,3)
   vec = (cell_data.ndim == 4)
   
   if vec:
      r_data = np.zeros(r.shape)
   else:
      r_data = np.zeros(r.shape[0])

   x_locs_r = np.round((r[:,0] - dims.x_min)/dims.dx).astype(int)
   y_locs_r = np.round((r[:,1] - dims.y_min)/dims.dy).astype(int)
   z_locs_r = np.round((r[:,2] - dims.z_min)/dims.dz).astype(int)
   
   if period[2]:
      x_locs_r = np.mod(x_locs_r, dims.x_size)
   if period[1]:
      y_locs_r = np.mod(y_locs_r, dims.y_size)
   if period[0]:
      z_locs_r = np.mod(z_locs_r, dims.z_size)
   
   x0 = (x_locs_r - 1 + 0.5)*dims.dx + dims.x_min
   y0 = (y_locs_r - 1 + 0.5)*dims.dy + dims.y_min
   z0 = (z_locs_r - 1 + 0.5)*dims.dz + dims.z_min
   x_w1 = (r[:,0] - x0)/dims.dx
   y_w1 = (r[:,1] - y0)/dims.dy
   z_w1 = (r[:,2] - z0)/dims.dz
   
   if period[2]:
      x_w1 = np.mod(x_w1, 1)
   if period[1]:
      y_w1 = np.mod(y_w1, 1)
   if period[0]:
      z_w1 = np.mod(z_w1, 1)
   
   x_w0 = 1 - x_w1
   y_w0 = 1 - y_w1
   z_w0 = 1 - z_w1
   
   weights = np.array([[x_w0,y_w0,z_w0],[x_w1,y_w0,z_w0],
                       [x_w0,y_w1,z_w0],[x_w1,y_w1,z_w0],
                       [x_w0,y_w0,z_w1],[x_w1,y_w0,z_w1],
                       [x_w0,y_w1,z_w1],[x_w1,y_w1,z_w1]])
   weights = 1/np.linalg.norm(weights, axis = 1)
   weights /= np.sum(weights, axis = 0)
   
   if vec:
      weights = weights.reshape(8,-1,1)
   else:
      weights = weights.reshape(8,-1)
   
   # r_data += weights[0]*cell_data[array_indices[0][z_locs_r],
   #                                array_indices[1][y_locs_r],
   #                                array_indices[2][x_locs_r],...]
   # r_data += weights[1]*cell_data[array_indices[0][z_locs_r+1],
   #                                array_indices[1][y_locs_r],
   #                                array_indices[2][x_locs_r],...]
   # r_data += weights[2]*cell_data[array_indices[0][z_locs_r],
   #                                array_indices[1][y_locs_r+1],
   #                                array_indices[2][x_locs_r],...]
   # r_data += weights[3]*cell_data[array_indices[0][z_locs_r+1],
   #                                array_indices[1][y_locs_r+1],
   #                                array_indices[2][x_locs_r],...]
   # r_data += weights[4]*cell_data[array_indices[0][z_locs_r],
   #                                array_indices[1][y_locs_r],
   #                                array_indices[2][x_locs_r+1],...]
   # r_data += weights[5]*cell_data[array_indices[0][z_locs_r+1],
   #                                array_indices[1][y_locs_r],
   #                                array_indices[2][x_locs_r+1],...]
   # r_data += weights[6]*cell_data[array_indices[0][z_locs_r],
   #                                array_indices[1][y_locs_r+1],
   #                                array_indices[2][x_locs_r+1],...]
   # r_data += weights[7]*cell_data[array_indices[0][z_locs_r+1],
   #                                array_indices[1][y_locs_r+1],
   #                                array_indices[2][x_locs_r+1],...]
   
   x_locs_l = shift_indices(dims.x_range, 1, period[2])[x_locs_r]
   y_locs_l = shift_indices(dims.y_range, 1, period[1])[y_locs_r]
   z_locs_l = shift_indices(dims.z_range, 1, period[0])[z_locs_r]
   
   r_data += weights[0]*cell_data[z_locs_l, y_locs_l, x_locs_l,...]
   r_data += weights[1]*cell_data[z_locs_r, y_locs_l, x_locs_l,...]
   r_data += weights[2]*cell_data[z_locs_l, y_locs_r, x_locs_l,...]
   r_data += weights[3]*cell_data[z_locs_r, y_locs_r, x_locs_l,...]
   r_data += weights[4]*cell_data[z_locs_l, y_locs_l, x_locs_r,...]
   r_data += weights[5]*cell_data[z_locs_r, y_locs_l, x_locs_r,...]
   r_data += weights[6]*cell_data[z_locs_l, y_locs_r, x_locs_r,...]
   r_data += weights[7]*cell_data[z_locs_r, y_locs_r, x_locs_r,...]

   if r.shape[0] == 1:
      r_data = r_data.reshape(3)
   
   return r_data

@njit(cache = True, fastmath = True)
def cell2r_njit(cell_data, r, period, dims):
   # Interpolate cell data to arbitrary position(s) r
   r = r.reshape(-1,3)
   vec = (cell_data.ndim == 4)

   p_count = r.shape[0]

   if vec:
      r_data = np.zeros((p_count,3))
   else:
      r_data = np.zeros(p_count)
   
   x_shift = shift_indices(dims.x_range, 1, period[2])
   y_shift = shift_indices(dims.y_range, 1, period[1])
   z_shift = shift_indices(dims.z_range, 1, period[0])
   
   for ii in range(p_count):
      x_locs_r = round((r[ii,0] - dims.x_min)/dims.dx)
      y_locs_r = round((r[ii,1] - dims.y_min)/dims.dy)
      z_locs_r = round((r[ii,2] - dims.z_min)/dims.dz)
      
      if period[2]:
         x_locs_r = x_locs_r % dims.x_size
      if period[1]:
         y_locs_r = y_locs_r % dims.y_size
      if period[0]:
         z_locs_r = z_locs_r % dims.z_size
      
      x0 = (x_locs_r - 1 + 0.5)*dims.dx + dims.x_min
      y0 = (y_locs_r - 1 + 0.5)*dims.dy + dims.y_min
      z0 = (z_locs_r - 1 + 0.5)*dims.dz + dims.z_min
      x_w1 = (r[ii,0] - x0)/dims.dx
      y_w1 = (r[ii,1] - y0)/dims.dy
      z_w1 = (r[ii,2] - z0)/dims.dz
      
      if period[2]:
         x_w1 = x_w1 % 1
      if period[1]:
         y_w1 = y_w1 % 1
      if period[0]:
         z_w1 = z_w1 % 1
      
      x_w0 = 1 - x_w1
      y_w0 = 1 - y_w1
      z_w0 = 1 - z_w1

      weights = np.array([[x_w0,y_w0,z_w0],[x_w1,y_w0,z_w0],
                          [x_w0,y_w1,z_w0],[x_w1,y_w1,z_w0],
                          [x_w0,y_w0,z_w1],[x_w1,y_w0,z_w1],
                          [x_w0,y_w1,z_w1],[x_w1,y_w1,z_w1]])
      weights = 1/np.sqrt((weights**2).sum(axis = 1))
      weights /= np.sum(weights, axis = 0)
      
      x_locs_l = x_shift[x_locs_r]
      y_locs_l = y_shift[y_locs_r]
      z_locs_l = z_shift[z_locs_r]

      if vec:
         for nn in range(3):
            r_data[ii,nn] += weights[0]*cell_data[z_locs_l, y_locs_l, x_locs_l,nn]
            r_data[ii,nn] += weights[1]*cell_data[z_locs_r, y_locs_l, x_locs_l,nn]
            r_data[ii,nn] += weights[2]*cell_data[z_locs_l, y_locs_r, x_locs_l,nn]
            r_data[ii,nn] += weights[3]*cell_data[z_locs_r, y_locs_r, x_locs_l,nn]
            r_data[ii,nn] += weights[4]*cell_data[z_locs_l, y_locs_l, x_locs_r,nn]
            r_data[ii,nn] += weights[5]*cell_data[z_locs_r, y_locs_l, x_locs_r,nn]
            r_data[ii,nn] += weights[6]*cell_data[z_locs_l, y_locs_r, x_locs_r,nn]
            r_data[ii,nn] += weights[7]*cell_data[z_locs_r, y_locs_r, x_locs_r,nn]
      else:
         r_data[ii] += weights[0]*cell_data[z_locs_l, y_locs_l, x_locs_l]
         r_data[ii] += weights[1]*cell_data[z_locs_r, y_locs_l, x_locs_l]
         r_data[ii] += weights[2]*cell_data[z_locs_l, y_locs_r, x_locs_l]
         r_data[ii] += weights[3]*cell_data[z_locs_r, y_locs_r, x_locs_l]
         r_data[ii] += weights[4]*cell_data[z_locs_l, y_locs_l, x_locs_r]
         r_data[ii] += weights[5]*cell_data[z_locs_r, y_locs_l, x_locs_r]
         r_data[ii] += weights[6]*cell_data[z_locs_l, y_locs_r, x_locs_r]
         r_data[ii] += weights[7]*cell_data[z_locs_r, y_locs_r, x_locs_r]
   
   return r_data
