#!/usr/bin/env python
from optparse import OptionParser  # use a parser for configuration
import aerodynamics
import numpy as np

class inputClass:

    def __init__(self, inputFile, deltaT):
        self.inputFile = inputFile
        self.deltaT = None
        self.U = None
        self.__readInputFile()
        if self.deltaT != deltaT:
            raise Exception('The ROM has been trained with a time step of '+str(deltaT)+', but a time step of '+str(self.deltaT)+' was found in the inputs')

    def __readInputFile(self):
        with open(self.inputFile, 'r') as file:
            print('Opened structural inputs file ' + self.inputFile + '.')
            headerLine = file.readline()
            headerLine = headerLine.strip('\r\n')
            headerLine = headerLine.split('\t')
            nModes = int((len(headerLine)-3)/3)
            self.U = np.empty((nModes, 0))
            timeOld = None
            while True:
                newColumn = np.empty((nModes, 1))
                line = file.readline()
                if not line:
                    break
                line = line.strip('\r\n')
                line = line.split('\t')
                time = float(line.pop(0))
                if not self.deltaT:
                    if not timeOld:
                        timeOld = time
                    else:
                        self.deltaT = time-timeOld
                dummy = line.pop(0)
                dummy = line.pop(0)
                for iMode in range(nModes):
                    newColumn[iMode] = float(line.pop(0))
                    dummy = line.pop(0)
                    dummy = line.pop(0)
                self.U = np.append(self.U, newColumn, axis=1)
            print('Completed reading')

def main():

    parser=OptionParser()
    parser.add_option("-n", "--normals",               dest="normals",
                      help="Read panel normals from FILE", metavar="FILE", default=None)
    parser.add_option("-m", "--modes",                 dest="modes",
                      help="Read mode shapes from FILE", metavar="FILE", default=None)
    parser.add_option("-d", "--dimension",             dest="dimenstion", type="int",
                      help="Specify the number of databases (i.e., the number of conditions)", default=1)
    parser.add_option("-s", "--structuralHistory",     dest="structuralHistory", action = "append",
                      help="Specify the file where the structural history can be found", default=[])
    parser.add_option("-a", "--aerodynamicHistory",    dest="aerodynamicHistory", action = "append",
                      help="Specify the file where the aerodynamic history can be found", default=[])
    parser.add_option("-i", "--inputs",                dest="inputs",
                      help="Specify the file where the future structural inputs can be found", default=[])
    parser.add_option("-o", "--outputs",                dest="outputs",
                      help="Specify the file where the future structural forces are printed", default=[])

    (options, args) = parser.parse_args()

    # Double check that the number of histories correspond to the requested dimension
    if options.dimension != len(options.structuralHistory) or options.dimension != len(options.aerodynamicHistory):
        raise Exception('Please provide history files in the same number as specified in the -d option')

    # Create the physical model
    model = aerodynamics.physicalModel(options.normals, options.modes)

    # Gather the databases
    databases = []
    for i in range(options.dimension):
        databases.append(aerodynamics.database(options.structuralHistory[i], options.aerodynamicHistory[i]))

    # Build the ROM
    ROM = aerodynamics.ROM(databases, model)

    # Build the future inputs class
    inputs = inputClass(options.inputs, databases[0].deltaT)

    # Run the prediction step
    forces = np.empty((inputs.U.shape()[0], 0), dtype=float)
    for i in range(inputs.U.shape()[1]):
        forces = np.append(forces, ROM.predict(inputs.U[:, i]), axis=1)
        ROM.update()

    # Print the obtained modal forces to file
    with open(options.outputs, 'w') as file:
        for i in range(forces.shape()[1]):
            toPrint = str(forces[:, i])
            file.write(toPrint)

if __name__ == '__main__':
    main()
