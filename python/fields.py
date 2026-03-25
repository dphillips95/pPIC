import numpy as np

from interpolators import cell2face,face2node,cell2face_njit,face2node_njit
from populations import calcNodeData

class Fields:
   # Class contains:
   # cellJp:    cell total particle current
   # faceJp:    face total particle current
   # faceB:     face magnetic fields
   # nodeB:     node magnetic fields
   # nodeE:     node electric field
   def __init__(self, pops, B_types, B_init, E_types, E_init, dims):
      # Initialise all fields
      print("")
      print("Initialising all fields")
      
      self.faceB = np.zeros(dims.dim_vector)
      self.nodeE = np.zeros(dims.dim_vector)
      
      Bx,By,Bz = B_init
      if dims.oneV is True:
         By = Bz = 0
      
      for B_type in B_types:
         if B_type == "uniform":
            self.faceB[:,:,:,0] += Bx
            self.faceB[:,:,:,1] += By
            self.faceB[:,:,:,2] += Bz
      apply_boundaries_fields(self.faceB, dims)
      
      self.update_fields(pops, dims)

      Ex,Ey,Ez = E_init
      if dims.oneV is True:
         Ey = Ez = 0
      
      for E_type in E_types:
         if E_type == "UeB_cross":
            if "e-" in pops.keys():
               nodeUe = pops["e-"].nodeU
            else:
               nodeUe = np.zeros(dims.dim_vector)
            self.nodeE -= np.cross(nodeUe, self.nodeB)
         elif E_type == "uniform":
            self.nodeE[:,:,:,0] += Ex
            self.nodeE[:,:,:,1] += Ey
            self.nodeE[:,:,:,2] += Ez
      
      apply_boundaries_fields(self.nodeE, dims)

   def update_fields(self, pops, dims):
      # Calculates non-upwinded fields, e.g. cellJp, nodeUe, etc.
      if len(pops) == 0:
         self.cellJp = np.zeros(dims.dim_vector)
      else:
         self.cellJp = np.sum([x.cellJi for x in pops.values()], axis = 0)
      apply_boundaries_fields(self.cellJp, dims)

      self.faceJp = cell2face(self.cellJp, dims)
      apply_boundaries_fields(self.faceJp, dims)

      self.nodeB = face2node(self.faceB, dims)
      apply_boundaries_fields(self.nodeB, dims)

def upwind_fields(fields, xnext, dims):
   # Upwind fields given solution to Ax = b
   # Returns updated fields and mid-time nodeE for Lorentz force
   if dims.oneV:
      newFaceB = fields.faceB.copy()
      newNodeE = fields.nodeE.copy()
      newFaceB[:,:,:,0] = xnext[:dims.Ncells_total].reshape(dims.dim_scalar)
      newNodeE[:,:,:,0] = xnext[dims.Ncells_total:].reshape(dims.dim_scalar)

   apply_boundaries_fields((newFaceB,newNodeE), dims)
   
   midNodeE = dims.theta*newNodeE + (1-dims.theta)*fields.nodeE
   
   fields.nodeE = newNodeE
   fields.faceB = newFaceB
   
   return fields,midNodeE

def apply_boundaries_fields(data, dims):
   # Applies Neumann BCs to given data array or each array in tuple
   # All boundary types (cell,node,face) use the same method
   # Data in outermost layer copied from nearest 'real' neighbour
   # Periodic BCs has no boundary layer, so skip for periodic boundaries

   if all(dims.period):
      # All boundaries periodic, no BCs to apply
      return
   
   if isinstance(data, tuple):
      for ii in range(len(data)):
         apply_boundaries_fields(data[ii], dims)
   
   if not dims.period[2]:
      data[:,:,0,...] = data[:,:,1,...]
      data[:,:,-1,...] = data[:,:,-2,...]
   
   if not dims.period[1]:
      data[:,0,...] = data[:,1,...]
      data[:,-1,...] = data[:,-1,...]
   
   if not dims.period[0]:
      data[0,...] = data[1,...]
      data[-1,...] = data[-2,...]
