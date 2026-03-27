# Module of interpolators
import math
import numpy as np
from numba import njit
from numba import int64,float64

from indexers import arr_shift,split_axis,arr_shift_njit,CIC_weights_node,CIC_weights_cell

def face2cell(face_data, dims):
   # Interpolates data from cell faces to centre
   # Data is assumed to be vector data
   xface_data,yface_data,zface_data = split_axis(face_data, axis = 3)
   
   cell_data = face_data.copy()
   
   cell_data[:,:,:,0] += arr_shift(xface_data, -1, 2, dims.period)
   cell_data[:,:,:,1] += arr_shift(yface_data, -1, 1, dims.period)
   cell_data[:,:,:,2] += arr_shift(zface_data, -1, 0, dims.period)
   
   cell_data *= 0.5
   
   return cell_data

@njit(cache = True, fastmath = True)
def face2cell_njit(face_data, dims):
   # Interpolates data from cell faces to centre
   # Data is assumed to be vector data
   cell_data = face_data.copy()

   shape = face_data.shape
   
   shift_x = arr_shift_njit(face_data[:,:,:,0], -1, 2, dims.period)
   shift_y = arr_shift_njit(face_data[:,:,:,1], -1, 1, dims.period)
   shift_z = arr_shift_njit(face_data[:,:,:,2], -1, 0, dims.period)
   for kk in range(shape[0]):
      for jj in range(shape[1]):
         for ii in range(shape[2]):
            cell_data[kk,jj,ii,0] += shift_x[kk,jj,ii]
            cell_data[kk,jj,ii,1] += shift_y[kk,jj,ii]
            cell_data[kk,jj,ii,2] += shift_z[kk,jj,ii]
   
   cell_data *= 0.5
   
   return cell_data

def cell2node(cell_data, dims):
   # Interpolates data from cell centres to nodes
   node_data = cell_data.copy()
   
   shift_x = arr_shift(cell_data, 1, 0, dims.period)
   shift_y = arr_shift(cell_data, 1, 1, dims.period)
   shift_z = arr_shift(cell_data, 1, 2, dims.period)
   shift_xy = arr_shift(shift_x, 1, 1, dims.period)
   shift_xz = arr_shift(shift_z, 1, 0, dims.period)
   shift_yz = arr_shift(shift_y, 1, 2, dims.period)
   shift_xyz = arr_shift(shift_xy, 1, 2, dims.period)
   
   node_data += shift_x
   node_data += shift_y
   node_data += shift_z
   node_data += shift_xy
   node_data += shift_xz
   node_data += shift_yz
   node_data += shift_xyz
   
   node_data *= 0.125

   return node_data

@njit(cache = True, fastmath = True)
def cell2node_njit(cell_data, dims):
   # Interpolates data from cell centres to nodes
   node_data = cell_data.copy()
   
   shift_x = arr_shift_njit(cell_data, 1, 0, dims.period)
   shift_y = arr_shift_njit(cell_data, 1, 1, dims.period)
   shift_z = arr_shift_njit(cell_data, 1, 2, dims.period)
   shift_xy = arr_shift_njit(shift_x, 1, 1, dims.period)
   shift_xz = arr_shift_njit(shift_z, 1, 0, dims.period)
   shift_yz = arr_shift_njit(shift_y, 1, 2, dims.period)
   shift_xyz = arr_shift_njit(shift_xy, 1, 2, dims.period)
   
   shape = cell_data.shape

   if cell_data.ndim == 3:
      for kk in range(shape[0]):
         for jj in range(shape[1]):
            for ii in range(shape[2]):
               node_data[kk,jj,ii] += shift_x[kk,jj,ii]
               node_data[kk,jj,ii] += shift_y[kk,jj,ii]
               node_data[kk,jj,ii] += shift_z[kk,jj,ii]
               node_data[kk,jj,ii] += shift_xy[kk,jj,ii]
               node_data[kk,jj,ii] += shift_xz[kk,jj,ii]
               node_data[kk,jj,ii] += shift_yz[kk,jj,ii]
               node_data[kk,jj,ii] += shift_xyz[kk,jj,ii]
   elif cell_data.ndim == 4:
      for kk in range(shape[0]):
         for jj in range(shape[1]):
            for ii in range(shape[2]):
               for nn in range(3):
                  node_data[kk,jj,ii,nn] += shift_x[kk,jj,ii,nn]
                  node_data[kk,jj,ii,nn] += shift_y[kk,jj,ii,nn]
                  node_data[kk,jj,ii,nn] += shift_z[kk,jj,ii,nn]
                  node_data[kk,jj,ii,nn] += shift_xy[kk,jj,ii,nn]
                  node_data[kk,jj,ii,nn] += shift_xz[kk,jj,ii,nn]
                  node_data[kk,jj,ii,nn] += shift_yz[kk,jj,ii,nn]
                  node_data[kk,jj,ii,nn] += shift_xyz[kk,jj,ii,nn]
   
   node_data *= 0.125

   return node_data

def face2node(face_data, dims):
   # Interpolate face data to node
   # Data is assumed to be vector data, face data only stores one component each
   node_data = face_data.copy()
   
   xface_data,yface_data,zface_data = split_axis(face_data, axis = 3)
   
   shift_x = arr_shift(xface_data, 1, 0, dims.period)
   shift_y = arr_shift(xface_data, 1, 1, dims.period)
   shift_xy = arr_shift(shift_x, 1, 1, dims.period)

   node_data[:,:,:,0] += shift_x
   node_data[:,:,:,0] += shift_y
   node_data[:,:,:,0] += shift_xy

   shift_x = arr_shift(yface_data, 1, 0, dims.period)
   shift_z = arr_shift(yface_data, 1, 2, dims.period)
   shift_xz = arr_shift(shift_x, 1, 2, dims.period)

   node_data[:,:,:,1] += shift_x
   node_data[:,:,:,1] += shift_z
   node_data[:,:,:,1] += shift_xz

   shift_y = arr_shift(zface_data, 1, 1, dims.period)
   shift_z = arr_shift(zface_data, 1, 2, dims.period)
   shift_yz = arr_shift(shift_y, 1, 2, dims.period)

   node_data[:,:,:,2] += shift_y
   node_data[:,:,:,2] += shift_z
   node_data[:,:,:,2] += shift_yz
   
   node_data *= 0.25
   
   return node_data

@njit(cache = True, fastmath = True)
def face2node_njit(face_data, dims):
   # Interpolate face data to node
   # Data is assumed to be vector data, face data only stores one component each
   node_data = face_data.copy()

   shape = face_data.shape
   
   shift_x = arr_shift_njit(face_data[:,:,:,0], 1, 0, dims.period)
   shift_y = arr_shift_njit(face_data[:,:,:,0], 1, 1, dims.period)
   shift_xy = arr_shift_njit(shift_x, 1, 1, dims.period)

   for kk in range(shape[0]):
      for jj in range(shape[1]):
         for ii in range(shape[2]):   
            node_data[kk,jj,ii,0] += shift_x[kk,jj,ii]
            node_data[kk,jj,ii,0] += shift_y[kk,jj,ii]
            node_data[kk,jj,ii,0] += shift_xy[kk,jj,ii]
   
   shift_x = arr_shift_njit(face_data[:,:,:,1], 1, 0, dims.period)
   shift_z = arr_shift_njit(face_data[:,:,:,1], 1, 2, dims.period)
   shift_xz = arr_shift_njit(shift_x, 1, 2, dims.period)

   for kk in range(shape[0]):
      for jj in range(shape[1]):
         for ii in range(shape[2]):   
            node_data[kk,jj,ii,1] += shift_x[kk,jj,ii]
            node_data[kk,jj,ii,1] += shift_z[kk,jj,ii]
            node_data[kk,jj,ii,1] += shift_xz[kk,jj,ii]

   shift_y = arr_shift_njit(face_data[:,:,:,2], 1, 1, dims.period)
   shift_z = arr_shift_njit(face_data[:,:,:,2], 1, 2, dims.period)
   shift_yz = arr_shift_njit(shift_y, 1, 2, dims.period)

   for kk in range(shape[0]):
      for jj in range(shape[1]):
         for ii in range(shape[2]):   
            node_data[kk,jj,ii,2] += shift_y[kk,jj,ii]
            node_data[kk,jj,ii,2] += shift_z[kk,jj,ii]
            node_data[kk,jj,ii,2] += shift_yz[kk,jj,ii]
   
   node_data *= 0.25
   
   return node_data
   
def node2cell(node_data, dims):
   # Interpolate node data to cell centres
   cell_data = node_data.copy()
   
   shift_x = arr_shift(node_data, -1, 0, dims.period)
   shift_y = arr_shift(node_data, -1, 1, dims.period)
   shift_z = arr_shift(node_data, -1, 2, dims.period)
   shift_xy = arr_shift(shift_x, -1, 1, dims.period)
   shift_xz = arr_shift(shift_z, -1, 0, dims.period)
   shift_yz = arr_shift(shift_y, -1, 2, dims.period)
   shift_xyz = arr_shift(shift_xy, -1, 2, dims.period)

   cell_data += shift_x
   cell_data += shift_y
   cell_data += shift_z
   cell_data += shift_xy
   cell_data += shift_xz
   cell_data += shift_yz
   cell_data += shift_xyz
   
   cell_data *= 0.125
   
   return cell_data

@njit(cache = True, fastmath = True)
def node2cell_njit(node_data, dims):
   # Interpolate node data to cell centres
   cell_data = node_data.copy()

   shape = node_data.shape
   
   shift_x = arr_shift_njit(node_data, -1, 0, dims.period)
   shift_y = arr_shift_njit(node_data, -1, 1, dims.period)
   shift_z = arr_shift_njit(node_data, -1, 2, dims.period)
   shift_xy = arr_shift_njit(shift_x, -1, 1, dims.period)
   shift_xz = arr_shift_njit(shift_z, -1, 0, dims.period)
   shift_yz = arr_shift_njit(shift_y, -1, 2, dims.period)
   shift_xyz = arr_shift_njit(shift_xy, -1, 2, dims.period)

   if node_data.ndim == 3:
      for kk in range(shape[0]):
         for jj in range(shape[1]):
            for ii in range(shape[2]):
               cell_data[kk,jj,ii] += shift_x[kk,jj,ii]
               cell_data[kk,jj,ii] += shift_y[kk,jj,ii]
               cell_data[kk,jj,ii] += shift_z[kk,jj,ii]
               cell_data[kk,jj,ii] += shift_xy[kk,jj,ii]
               cell_data[kk,jj,ii] += shift_xz[kk,jj,ii]
               cell_data[kk,jj,ii] += shift_yz[kk,jj,ii]
               cell_data[kk,jj,ii] += shift_xyz[kk,jj,ii]
   elif node_data.ndim == 4:
      for kk in range(shape[0]):
         for jj in range(shape[1]):
            for ii in range(shape[2]):
               for nn in range(3):
                  cell_data[kk,jj,ii,nn] += shift_x[kk,jj,ii,nn]
                  cell_data[kk,jj,ii,nn] += shift_y[kk,jj,ii,nn]
                  cell_data[kk,jj,ii,nn] += shift_z[kk,jj,ii,nn]
                  cell_data[kk,jj,ii,nn] += shift_xy[kk,jj,ii,nn]
                  cell_data[kk,jj,ii,nn] += shift_xz[kk,jj,ii,nn]
                  cell_data[kk,jj,ii,nn] += shift_yz[kk,jj,ii,nn]
                  cell_data[kk,jj,ii,nn] += shift_xyz[kk,jj,ii,nn]
   
   cell_data *= 0.125
   
   return cell_data
   
def node2face(node_data, dims):
   # Interpolate node data to face
   # Data is assumed to be vector data, face data only stores one component each
   face_data = node_data.copy()
   
   xnode_data,ynode_data,znode_data = split_axis(node_data, axis = 3)
   
   shift_x = arr_shift(xnode_data, -1, 0, dims.period)
   shift_y = arr_shift(xnode_data, -1, 1, dims.period)
   shift_xy = arr_shift(shift_x, -1, 1, dims.period)
   
   face_data[:,:,:,0] += shift_x
   face_data[:,:,:,0] += shift_y
   face_data[:,:,:,0] += shift_xy

   shift_x = arr_shift(ynode_data, -1, 0, dims.period)
   shift_z = arr_shift(ynode_data, -1, 2, dims.period)
   shift_xz = arr_shift(shift_x, -1, 2, dims.period)
   
   face_data[:,:,:,1] += shift_x
   face_data[:,:,:,1] += shift_z
   face_data[:,:,:,1] += shift_xz

   shift_y = arr_shift(znode_data, -1, 1, dims.period)
   shift_z = arr_shift(znode_data, -1, 2, dims.period)
   shift_yz = arr_shift(shift_y, -1, 2, dims.period)
   
   face_data[:,:,:,2] += shift_y
   face_data[:,:,:,2] += shift_z
   face_data[:,:,:,2] += shift_yz
   
   face_data *= 0.25
   
   return face_data

@njit(cache = True, fastmath = True)
def node2face_njit(node_data, dims):
   # Interpolate node data to face
   # Data is assumed to be vector data, face data only stores one component each
   face_data = node_data.copy()

   shape = node_data.shape
   
   shift_x = arr_shift_njit(node_data[:,:,:,0], -1, 0, dims.period)
   shift_y = arr_shift_njit(node_data[:,:,:,0], -1, 1, dims.period)
   shift_xy = arr_shift_njit(shift_x, -1, 1, dims.period)

   for kk in range(shape[0]):
      for jj in range(shape[1]):
         for ii in range(shape[2]):
            face_data[kk,jj,ii,0] += shift_x[kk,jj,ii]
            face_data[kk,jj,ii,0] += shift_y[kk,jj,ii]
            face_data[kk,jj,ii,0] += shift_xy[kk,jj,ii]

   shift_x = arr_shift_njit(node_data[:,:,:,1], -1, 0, dims.period)
   shift_z = arr_shift_njit(node_data[:,:,:,1], -1, 2, dims.period)
   shift_xz = arr_shift_njit(shift_x, -1, 2, dims.period)
   
   for kk in range(shape[0]):
      for jj in range(shape[1]):
         for ii in range(shape[2]):
            face_data[kk,jj,ii,1] += shift_x[kk,jj,ii]
            face_data[kk,jj,ii,1] += shift_z[kk,jj,ii]
            face_data[kk,jj,ii,1] += shift_xz[kk,jj,ii]

   shift_y = arr_shift_njit(node_data[:,:,:,2], -1, 1, dims.period)
   shift_z = arr_shift_njit(node_data[:,:,:,2], -1, 2, dims.period)
   shift_yz = arr_shift_njit(shift_y, -1, 2, dims.period)
   
   for kk in range(shape[0]):
      for jj in range(shape[1]):
         for ii in range(shape[2]):
            face_data[kk,jj,ii,2] += shift_y[kk,jj,ii]
            face_data[kk,jj,ii,2] += shift_z[kk,jj,ii]
            face_data[kk,jj,ii,2] += shift_yz[kk,jj,ii]
   
   face_data *= 0.25
   
   return face_data
   
def cell2face(cell_data, dims):
   # Interpolates data from cell centres to faces
   # Data is assumed to be vector data, face data only stores one component each
   face_data = cell_data.copy()
   
   for ii in range(3):
      vector_comp = 2 - ii
      face_data[:,:,:,vector_comp] += arr_shift(cell_data[:,:,:,vector_comp], 1, ii, dims.period)

   face_data *= 0.5
   
   return face_data

@njit(cache = True, fastmath = True)
def cell2face_njit(cell_data, dims):
   # Interpolates data from cell centres to faces
   # Data is assumed to be vector data, face data only stores one component each
   face_data = cell_data.copy()

   shape = cell_data.shape
   
   for nn in range(3):
      shift_axis = 2 - nn
      tmp = arr_shift_njit(cell_data[:,:,:,nn], 1, shift_axis, dims.period)
      for kk in range(shape[0]):
         for jj in range(shape[1]):
            for ii in range(shape[2]):
               face_data[kk,jj,ii,nn] += tmp[kk,jj,ii]

   face_data *= 0.5
   
   return face_data

def face2r(face_data, r, dims):
   # Interpolate face data to arbitrary position(s) r
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
def face2r_njit(face_data, r, dims):
   # Interpolate face data to arbitrary position(s) r
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

def node2r(node_data, r, dims):
   # Interpolate node data to arbitrary position(s) r
   r = np.array(r).reshape(-1,3)
   vec = (node_data.ndim == 4)
   
   if vec:
      r_data = np.zeros(r.shape)
   else:
      r_data = np.zeros(r.shape[0])
   
   if dims.linear:
      (x_ind,y_ind,z_ind),(x_w,y_w,z_w) = CIC_weights_node(r, dims, True, True)
      weights = x_w*y_w*z_w
   else:
      (x_ind,y_ind,z_ind),(x_w,y_w,z_w) = CIC_weights_node(r, dims, False, False)
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
   
   r_data += weights[0,0,0]*node_data[z_ind[0], y_ind[0], x_ind[0],...]
   r_data += weights[0,0,1]*node_data[z_ind[0], y_ind[0], x_ind[1],...]
   r_data += weights[0,1,0]*node_data[z_ind[0], y_ind[1], x_ind[0],...]
   r_data += weights[0,1,1]*node_data[z_ind[0], y_ind[1], x_ind[1],...]
   r_data += weights[1,0,0]*node_data[z_ind[1], y_ind[0], x_ind[0],...]
   r_data += weights[1,0,1]*node_data[z_ind[1], y_ind[0], x_ind[1],...]
   r_data += weights[1,1,0]*node_data[z_ind[1], y_ind[1], x_ind[0],...]
   r_data += weights[1,1,1]*node_data[z_ind[1], y_ind[1], x_ind[1],...]

   if r.shape[0] == 1:
      r_data = r_data.reshape(3)
   
   return r_data

@njit(cache = True, fastmath = True)
def node2r_njit(node_data, r, dims):
   # Interpolate node data to arbitrary position(s) r
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

def cell2r(cell_data, r, dims):
   # Interpolate cell data to arbitrary position(s) r
   # Interpolation is by default trilinear
   # if linear is False then uses weighted average of euclidean distanc
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
   
   r_data += weights[0,0,0]*cell_data[z_ind[0], y_ind[0], x_ind[0],...]
   r_data += weights[0,0,1]*cell_data[z_ind[0], y_ind[0], x_ind[1],...]
   r_data += weights[0,1,0]*cell_data[z_ind[0], y_ind[1], x_ind[0],...]
   r_data += weights[0,1,1]*cell_data[z_ind[0], y_ind[1], x_ind[1],...]
   r_data += weights[1,0,0]*cell_data[z_ind[1], y_ind[0], x_ind[0],...]
   r_data += weights[1,0,1]*cell_data[z_ind[1], y_ind[0], x_ind[1],...]
   r_data += weights[1,1,0]*cell_data[z_ind[1], y_ind[1], x_ind[0],...]
   r_data += weights[1,1,1]*cell_data[z_ind[1], y_ind[1], x_ind[1],...]

   if r.shape[0] == 1:
      if vec:
         r_data = r_data.reshape(3)
      else:
         r_data = r_data.item()
   
   return r_data

@njit(cache = True, fastmath = True)
def cell2r_njit(cell_data, r, dims):
   # Interpolate cell data to arbitrary position(s) r
   # Interpolation is by default trilinear
   # if linear is False then uses weighted average of euclidean distance
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
