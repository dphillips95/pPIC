import os
import numpy as np
import h5py
import scipy.constants as const

from indexers import split_axis

def save_data(path, fields, pops, dims):
   # Save all data to hdf5 files
   
   # List of all field datasets together with data
   field_dataset_list = [
      ("faceBx", fields.faceB[:,:,:,0]),
      ("faceBy", fields.faceB[:,:,:,1]),
      ("faceBz", fields.faceB[:,:,:,2]),
      ("nodeEx", fields.nodeE[:,:,:,0]),
      ("nodeEy", fields.nodeE[:,:,:,1]),
      ("nodeEz", fields.nodeE[:,:,:,2]),
      ("divB", fields.divB),
      ("divE", fields.divE),
   ]

   for pop in pops.values():
      sub_list = [
         (pop.name + "_cellUx", pop.cellU[:,:,:,0]),
         (pop.name + "_cellUy", pop.cellU[:,:,:,1]),
         (pop.name + "_cellUz", pop.cellU[:,:,:,2]),
         (pop.name + "_cellN", pop.cellN[:,:,:]),
         (pop.name + "_cellJx", pop.cellJi[:,:,:,0]),
         (pop.name + "_cellJy", pop.cellJi[:,:,:,1]),
         (pop.name + "_cellJz", pop.cellJi[:,:,:,2]),
         (pop.name + "_cellT", pop.cellT[:,:,:])
      ]
      field_dataset_list += sub_list
   
   filepath = path + "/fields.h5"
   
   append_file = os.path.isfile(filepath)

   logs = calcDiagnostics(fields, pops, dims)
   
   with h5py.File(filepath, "a") as f:
      # Add dimensions
      if append_file:
         dset_x = f["x"]
         dset_y = f["y"]
         dset_z = f["z"]
         dset_t = f["t"]
         dset_tstep = f["timestep"]
         
         dset_t.resize(dset_t.shape[0] + 1, axis = 0)
         dset_tstep.resize(dset_tstep.shape[0] + 1, axis = 0)
         
         dset_t[-1] = dims.time
         dset_tstep[-1] = dims.timestep
      else:
         dset_x = f.create_dataset('x', data = dims.x_locs)
         dset_y = f.create_dataset('y', data = dims.y_locs)
         dset_z = f.create_dataset('z', data = dims.z_locs)
         dset_t = f.create_dataset('t', data = [dims.time], maxshape = (None,))
         dset_tstep = f.create_dataset('timestep', data = [dims.timestep], maxshape = (None,))

         dset_x.make_scale('x')
         dset_y.make_scale('y')
         dset_z.make_scale('z')
         dset_t.make_scale('t')
         
         dset_tstep.dims[0].attach_scale(dset_t)
         dset_tstep.dims[0].label = 't'

      scales = (dset_t,dset_z,dset_y,dset_x)
      
      for name,data in field_dataset_list:
         if append_file:
            append_to_dataset(f, name, data)
         else:
            create_dataset(f, name, data, scales)
   
   filepath = path + "/logs.h5"
   
   with h5py.File(filepath, "a") as f:
      # Add dimensions
      if append_file:
         dset_t = f["t"]
         dset_tstep = f["timestep"]
         
         dset_t.resize(dset_t.shape[0] + 1, axis = 0)
         dset_tstep.resize(dset_tstep.shape[0] + 1, axis = 0)
         
         dset_t[-1] = dims.time
         dset_tstep[-1] = dims.timestep
      else:
         dset_t = f.create_dataset('t', data = [dims.time], maxshape = (None,))
         dset_tstep = f.create_dataset('timestep', data = [dims.timestep], maxshape = (None,))

         dset_t.make_scale('t')
         
         dset_tstep.dims[0].attach_scale(dset_t)
         dset_tstep.dims[0].label = 't'

      scales = (dset_t,)
      
      for log_name,log_value in logs.items():
         if append_file:
            append_to_dataset(f, log_name, log_value)
         else:
            create_1D_dataset(f, log_name, log_value, scales)

def create_dataset(f, name, data, scales):
   # Create new h5 dataset
   shape = data.shape
   dtype = data.dtype
   data[np.newaxis,...]
   dset = f.create_dataset(name, data = data, shape = [1] + list(shape), maxshape = [None] + list(shape), dtype = dtype)

   for ii in range(4):
      dset.dims[ii].attach_scale(scales[ii])

   dset.dims[0].label = 't'
   dset.dims[1].label = 'z'
   dset.dims[2].label = 'y'
   dset.dims[3].label = 'x'

def create_1D_dataset(f, name, data, scales):
   # Create new h5 dataset for time-only data (e.g. total energy etc.)
   data = np.array(data)
   dtype = data.dtype
   dset = f.create_dataset(name, data = data, shape = [1], maxshape = [None], dtype = dtype)
   
   dset.dims[0].attach_scale(scales[0])
   
   # dset.dims[0].label = 't'

def append_to_dataset(f, name, data):
   # Append single dataset in hdf5 file
   dset = f[name]
   shape = dset.shape
   dset.resize(shape[0] + 1, axis = 0)
   dset[-1] = data

def calcDiagnostics(fields, pops, dims):
   # Calculate scalar diagnostics, e.g. total energy
   logs = {}

   suffixes = ("x","y","z")
   
   avgFaceB = fields.faceB.mean(axis = (0,1,2))
   avgNodeE = fields.nodeE.mean(axis = (0,1,2))

   for ii,suff in enumerate(suffixes):
      logs["avgFaceB" + suff] = avgFaceB[ii]
      logs["avgNodeE" + suff] = avgNodeE[ii]
   
   logs["avgFaceB_mag"] = np.linalg.norm(fields.faceB, axis = -1).mean()
   logs["avgNodeE_mag"] = np.linalg.norm(fields.nodeE, axis = -1).mean()
   logs["maxFaceB_mag"] = np.linalg.norm(fields.faceB, axis = -1).max()
   logs["maxNodeE_mag"] = np.linalg.norm(fields.nodeE, axis = -1).max()

   logs["energy_B"] = np.sum(dims.dV*fields.faceB**2/(2*const.mu_0))
   logs["energy_E"] = np.sum(dims.dV*fields.nodeE**2*const.epsilon_0/2)

   logs["total_energy"] = logs["energy_B"] + logs["energy_E"]

   if len(pops) == 0:
      cellRhoQ = np.zeros(dims.dim_scalar)
      cellJ = np.zeros(dims.dim_vector)
   else:
      cellRhoQ,cellJ = zip(*((pop.cellRhoQ,pop.cellJi) for pop in pops.values()))
      cellRhoQ = np.array(cellRhoQ).sum(axis = 0)
      cellJ = np.array(cellJ).sum(axis = 0)
   
   logs["avgCellRhoQ"] = np.sum(cellRhoQ)
   
   avgJ = cellJ.mean(axis = (0,1,2))
   
   for ii,suff in enumerate(suffixes):
      logs["avgCellJ" + suff] = avgJ[ii]
   
   logs["avgCellJ_mag"] = np.linalg.norm(cellJ, axis = -1).mean()
   logs["maxCellJ_mag"] = np.linalg.norm(cellJ, axis = -1).max()
   
   for pop in pops.values():
      logs[pop.name + "_Np"] = pop.Np
      logs[pop.name + "_parts"] = pop.Np*pop.w
      
      avgU = pop.v.mean(axis = 0)

      for ii,suff in enumerate(suffixes):
         logs[pop.name + "_avgCellU" + suff] = avgU[ii]

      logs[pop.name + "_avgCellU_mag"] = np.linalg.norm(pop.v, axis = -1).mean()
      logs[pop.name + "_maxCellU_mag"] = np.linalg.norm(pop.v, axis = -1).max()
      
      avgJ = pop.cellJi.mean(axis = (0,1,2))
      
      for ii,suff in enumerate(suffixes):
         logs[pop.name + "_avgCellJi" + suff] = avgJ[ii]

      logs[pop.name + "_avgCellJi_mag"] = np.linalg.norm(pop.cellJi, axis = -1).mean()
      logs[pop.name + "_maxCellJi_mag"] = np.linalg.norm(pop.cellJi, axis = -1).max()
      
      logs[pop.name + "_KE"] = np.sum(pop.v**2*pop.w*pop.m/2)

      logs["total_energy"] += logs[pop.name + "_KE"]
   
   return logs
