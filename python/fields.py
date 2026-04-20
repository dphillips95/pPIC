import numpy as np

from interpolators import cell2face,face2node,cell2face_njit,face2node_njit,div_face2cell,div_node2cell

class Fields:
   # Class contains:
   # cellJp: cell total particle current
   # faceJp: face total particle current
   # faceB:  face magnetic fields
   # nodeB:  node magnetic fields
   # nodeE:  node electric field
   # divB:   cell magnetic field divergence
   # divE:   cell electric field divergence
   def __init__(self, pops, B_types, E_types, config, rng, dims):
      # Initialise all fields
      print("")
      print("Initialising all fields")
      
      self.faceB = np.zeros(dims.dim_vector)
      self.nodeE = np.zeros(dims.dim_vector)
      
      for B_type in B_types:
         if B_type == "uniform":
            Bx = config.getfloat("magnetic_field", "Bx")
            By = config.getfloat("magnetic_field", "By")
            Bz = config.getfloat("magnetic_field", "Bz")

            if dims.oneV is True:
               By = Bz = 0
            self.faceB[:,:,:,0] += Bx
            self.faceB[:,:,:,1] += By
            self.faceB[:,:,:,2] += Bz
         elif B_type == "rand":
            Bx_min = config.getfloat("magnetic_field", "rand_Bx_min")
            Bx_max = config.getfloat("magnetic_field", "rand_Bx_max")
            By_min = config.getfloat("magnetic_field", "rand_By_min")
            By_max = config.getfloat("magnetic_field", "rand_By_max")
            Bz_min = config.getfloat("magnetic_field", "rand_Bz_min")
            Bz_max = config.getfloat("magnetic_field", "rand_Bz_max")
            
            if dims.oneV is True:
               By_min = By_max = Bz_min = Bz_max = 0

            if dims.dy == 1 and dims.dz == 1:
               Bx_min = (Bx_min + Bx_max)/2
               Bx_max = Bx_min
               
            Bx = rng.uniform(-Bx_min, Bx_max, dims.dim_scalar)
            By = rng.uniform(-By_min, By_max, dims.dim_scalar)
            Bz = rng.uniform(-Bz_min, Bz_max, dims.dim_scalar)
            
            self.faceB += np.stack((Bx,By,Bz), axis = -1)
      
      apply_boundaries_fields(self.faceB, dims)
      
      self.update_fields(pops, dims)
      
      for E_type in E_types:
         if E_type == "UeB_cross":
            if "e-" in pops.keys():
               nodeUe = pops["e-"].nodeU
            else:
               nodeUe = np.zeros(dims.dim_vector)
            self.nodeE -= np.cross(nodeUe, self.nodeB)
         elif E_type == "uniform":
            Ex = config.getfloat("electric_field", "Ex")
            Ey = config.getfloat("electric_field", "Ey")
            Ez = config.getfloat("electric_field", "Ez")

            if dims.oneV is True:
               Ey = Ez = 0
            
            self.nodeE[:,:,:,0] += Ex
            self.nodeE[:,:,:,1] += Ey
            self.nodeE[:,:,:,2] += Ez
         elif E_type == "rand":
            Ex_min = config.getfloat("electric_field", "rand_Ex_min")
            Ex_max = config.getfloat("electric_field", "rand_Ex_max")
            Ey_min = config.getfloat("electric_field", "rand_Ey_min")
            Ey_max = config.getfloat("electric_field", "rand_Ey_max")
            Ez_min = config.getfloat("electric_field", "rand_Ez_min")
            Ez_max = config.getfloat("electric_field", "rand_Ez_max")

            if dims.oneV is True:
               Ey_min = Ey_max = Ez_min = Ez_max = 0

            Ex = rng.uniform(-Ex_min, Ex_max, dims.dim_scalar)
            Ey = rng.uniform(-Ey_min, Ey_max, dims.dim_scalar)
            Ez = rng.uniform(-Ez_min, Ez_max, dims.dim_scalar)
            
            self.nodeE += np.stack((Ex,Ey,Ez), axis = -1)
      
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

      self.divB = div_face2cell(self.faceB, dims)
      self.divE = div_node2cell(self.nodeE, dims)

def upwind_fields(fields, xnext, dims):
   # Upwind fields given solution to Ax = b
   # Returns updated fields and mid-time nodeE for Lorentz force
   if dims.oneV:
      newFaceB = fields.faceB.copy()
      newNodeE = fields.nodeE.copy()
      newFaceB[:,:,:,0] = xnext[:dims.Ncells_total].reshape(dims.dim_scalar)
      newNodeE[:,:,:,0] = xnext[dims.Ncells_total:].reshape(dims.dim_scalar)
   else:
      newFaceB = xnext[:3*dims.Ncells_total].reshape(dims.dim_vector)
      newNodeE = xnext[3*dims.Ncells_total:].reshape(dims.dim_vector)
      
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
