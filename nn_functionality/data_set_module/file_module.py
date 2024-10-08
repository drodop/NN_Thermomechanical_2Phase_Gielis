# .......................................................... #
# Functionality for data reading
# D. Rodopoulos
# .......................................................... #
""" The functions: readFile() and lineToData are based on the following link:
    http://www2.lawrence.edu/fast/GREGGJ/CMSC210/files/files.html
"""    

def readDataFile(fileName, key):
   data = []
   with open(fileName) as f:
       for line in f.readlines():

           if (key == 'ThermalConductivity'):           
               data.append(lineToData_tg(line.split()))
           elif (key == 'IsothermalYoung'):
               data.append(lineToData_iy(line.split()))
           elif (key == 'IsothermalShear'):
               data.append(lineToData_is(line.split()))               
           else:
               print("Problem here (read nodes)")
   return data

def readDataFile_general(fileName, ndim, ncm):
   data = []
   with open(fileName) as f:
       for line in f.readlines():
           data.append(lineToData(line.split(), ndim, ncm))

   return data

def readPixelDataFile(fileName, key):
   data = []
   with open(fileName) as f:
       for line in f.readlines():

           if (key == '4lq'):           
               data.append(lineToData_4lq(line.split()))
           else:
               print("Problem here (read nodes)")

   return data

def lineToData_tg(line):
  # Load line of thermal conductivity file
  return (float(line[0]),float(line[1]),float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]), float(line[7]))    

def lineToData_iy(line):
  # Load line of isothermal Young file
  return (float(line[0]),float(line[1]),float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]), float(line[7]), float(line[8]), float(line[9]), float(line[10]))

def lineToData_is(line):
  # Load line of isothermal Young file
  return (float(line[0]),float(line[1]),float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]), float(line[7]), float(line[8]))

def lineToData(line, ndim, ncm):
  # Load line of general file
  read_tuple = ()

  for idim in range(0, ndim):
     read_tuple += (float(line[idim]),)

  for icm in range(0, ncm):
     read_tuple += (float(line[icm + ndim]),)

  return read_tuple

def lineToData_4lq(line):
  # Load line of thermal conductivity file
  return (int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]), float(line[7]), float(line[8]))    








