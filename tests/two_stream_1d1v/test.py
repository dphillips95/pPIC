import numpy as np
from timeit import default_timer as timer

class my_timers:
   def __init__(self):
      self.timers = dict()

   def tic(self, _timer_):
      self.timers[_timer_] -= timer()
   
   def toc(self, _timer_):
      self.timers[_timer_] += timer()
   
   def start(self, _timer_):
      self.timers[_timer_] = 0.0

   def reset(self, _timer_ = None):
      if _timer_ == None:
         for key in self.timers.keys():
            self.timers[key] = 0.0
      else:
         self.timers[_timer_] = 0.0

timers = my_timers()

timers.start("nest")
timers.start("flat")
timers.start("flatten")

nest = np.zeros((2,2,2,2,2,2), dtype = np.float64)
flat = np.zeros((2,2,2,2,2,2), dtype = np.float64)
flatten = np.zeros((2,2,2,2,2,2), dtype = np.float64)

for _ in range(100000):
   timers.tic("nest")
   for ii in range(2):
      for jj in range(2):
         for kk in range(2):
            for ll in range(2):
               for mm in range(2):
                  for nn in range(2):
                     nest[ii,jj,kk,ll,mm,nn] += 1

   timers.toc("nest")
   timers.tic("flat")
   flat[0,0,0,0,0,0] += 1
   flat[0,0,0,0,0,1] += 1
   flat[0,0,0,0,1,0] += 1
   flat[0,0,0,0,1,1] += 1
   flat[0,0,0,1,0,0] += 1
   flat[0,0,0,1,0,1] += 1
   flat[0,0,0,1,1,0] += 1
   flat[0,0,0,1,1,1] += 1
   flat[0,0,1,0,0,0] += 1
   flat[0,0,1,0,0,1] += 1
   flat[0,0,1,0,1,0] += 1
   flat[0,0,1,0,1,1] += 1
   flat[0,0,1,1,0,0] += 1
   flat[0,0,1,1,0,1] += 1
   flat[0,0,1,1,1,0] += 1
   flat[0,0,1,1,1,1] += 1
   flat[0,1,0,0,0,0] += 1
   flat[0,1,0,0,0,1] += 1
   flat[0,1,0,0,1,0] += 1
   flat[0,1,0,0,1,1] += 1
   flat[0,1,0,1,0,0] += 1
   flat[0,1,0,1,0,1] += 1
   flat[0,1,0,1,1,0] += 1
   flat[0,1,0,1,1,1] += 1
   flat[0,1,1,0,0,0] += 1
   flat[0,1,1,0,0,1] += 1
   flat[0,1,1,0,1,0] += 1
   flat[0,1,1,0,1,1] += 1
   flat[0,1,1,1,0,0] += 1
   flat[0,1,1,1,0,1] += 1
   flat[0,1,1,1,1,0] += 1
   flat[0,1,1,1,1,1] += 1
   flat[1,0,0,0,0,0] += 1
   flat[1,0,0,0,0,1] += 1
   flat[1,0,0,0,1,0] += 1
   flat[1,0,0,0,1,1] += 1
   flat[1,0,0,1,0,0] += 1
   flat[1,0,0,1,0,1] += 1
   flat[1,0,0,1,1,0] += 1
   flat[1,0,0,1,1,1] += 1
   flat[1,0,1,0,0,0] += 1
   flat[1,0,1,0,0,1] += 1
   flat[1,0,1,0,1,0] += 1
   flat[1,0,1,0,1,1] += 1
   flat[1,0,1,1,0,0] += 1
   flat[1,0,1,1,0,1] += 1
   flat[1,0,1,1,1,0] += 1
   flat[1,0,1,1,1,1] += 1
   flat[1,1,0,0,0,0] += 1
   flat[1,1,0,0,0,1] += 1
   flat[1,1,0,0,1,0] += 1
   flat[1,1,0,0,1,1] += 1
   flat[1,1,0,1,0,0] += 1
   flat[1,1,0,1,0,1] += 1
   flat[1,1,0,1,1,0] += 1
   flat[1,1,0,1,1,1] += 1
   flat[1,1,1,0,0,0] += 1
   flat[1,1,1,0,0,1] += 1
   flat[1,1,1,0,1,0] += 1
   flat[1,1,1,0,1,1] += 1
   flat[1,1,1,1,0,0] += 1
   flat[1,1,1,1,0,1] += 1
   flat[1,1,1,1,1,0] += 1
   flat[1,1,1,1,1,1] += 1
   timers.toc("flat")
   timers.tic("flatten")
   for ii in range(flatten.size):
      # tmp = flatten.flatten()
      flatten.flat[ii] += 1
   # flatten  = tmp.reshape(2,2,2,2,2,2)
   timers.toc("flatten")
   
print("nest:    " + str(timers.timers["nest"]))
print("flat:    " + str(timers.timers["flat"]))
print("flatten: " + str(timers.timers["flatten"]))
