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
            raise Exception(
                'The ROM has been trained with a time step of ' + str(deltaT) + ', but a time step of ' + str(
                    self.deltaT) + ' was found in the inputs')

    def __readInputFile(self):
        with open(self.inputFile, 'r') as file:
            print('Opened structural inputs file ' + self.inputFile + '.')
            headerLine = file.readline()
            headerLine = headerLine.strip('\r\n')
            headerLine = headerLine.split('\t')
            nModes = int((len(headerLine) - 3) / 3)
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
                        self.deltaT = time - timeOld
                dummy = line.pop(0)
                dummy = line.pop(0)
                for iMode in range(nModes):
                    newColumn[iMode] = float(line.pop(0))
                    dummy = line.pop(0)
                    dummy = line.pop(0)
                self.U = np.append(self.U, newColumn, axis=1)
            print('Completed reading')


def main():
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="cfgFile",
                      help="Read configuration from FILE", metavar="FILE", default=None)

    (options, args) = parser.parse_args()
    configuration = readConfig(options.cfgFile)

    # Double check that the number of histories correspond to the requested dimension
    if configuration["DIMENSION"] != len(configuration["STRUCT_HISTORY"]) or configuration["DIMENSION"] != len(configuration["AERO_HISTORY"]):
        raise Exception('Please provide history files in the same number as specified in the -d option')

    # Create the physical model
    model = aerodynamics.physicalModel(configuration["NORMALS"], configuration["MODES"])

    # Gather the databases
    databases = []
    for i in range(int(configuration["DIMENSION"])):
        databases.append(aerodynamics.database(configuration["STRUCT_HISTORY"][i], configuration["AERO_HISTORY"][i]))

    # Build the ROM
    ROM = aerodynamics.ROM(databases, model)

    if configuration["IMPOSED_MOTION"]=="YES":
        # Build the future inputs class
        inputs = inputClass(configuration["INPUTS"], databases[0].deltaT)

        # Run the prediction step
        modalForces = np.empty((inputs.U.shape()[0], 0), dtype=float)
        forces = np.empty((0), dtype=float)
        for i in range(inputs.U.shape()[1]):
            modalForces = np.append(modalForces, ROM.predict(inputs.U[:, i]), axis=1)
            ROM.update()
            forces = np.append(forces, ROM.model.getCl(ROM.model.X))

        # Print the obtained modal forces to file
        with open(configuration["OUTPUTS"], 'w') as file:
            for i in range(forces.shape()[1]):
                toPrint = str(forces[i])
                file.write(toPrint)
    else:
        import structure
        solverConfiguration = {"N_MODES": ROM.nmodes, "DELTA_T": ROM.deltaT, "MODAL_DAMP": configuration["MODAL_DAMP"],
                               "PUNCH_FILE": configuration["PUNCH_FILE"], "INITIAL_MODES": ROM.databases[0].Uinit,
                               "INITIAL_VEL": ROM.databases[0].Udotinit, "OUTPUTS": configuration["OUTPUTS"]}
        solver = structure.solver(solverConfiguration)
        solver.writeSolution(solver.timeIter)
        for timeIter in range(int(configuration["TIME_ITER"])):
            solver.applyload(np.array(ROM.predict(solver.q)))
            solver.run()
            solver.updateSolution()
            ROM.update()
            solver.writeSolution(solver.timeIter)


def readConfig(cfgFile):
    input_file = open(cfgFile)
    configuration = {}
    while 1:
        line = input_file.readline()
        if not line:
            break
        # remove line returns
        line = line.strip('\r\n')
        # make sure it has useful data
        if ("=" not in line) or (line[0] == '%'):
            continue
        # split across equal sign
        line = line.split("=", 1)
        this_param = line[0].strip()
        this_value = line[1].strip()

        configuration[this_param] = this_value

        return configuration


if __name__ == '__main__':
    main()