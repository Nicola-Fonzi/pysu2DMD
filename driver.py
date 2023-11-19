#!/usr/bin/env python
import pickle
import aerodynamics
import numpy as np
from sys import stdout
import os


class inputClass:

    def __init__(self, inputFile, deltaT):
        self.inputFile = inputFile
        self.deltaT = None
        self.U = None
        self.Udot = None
        self.__readInputFile()
        if (self.deltaT - deltaT) > 1e-16:
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
            self.Udot = np.empty((nModes, 0))
            self.Uddot = np.empty((nModes, 0))
            timeOld = None
            while True:
                newColumn = np.empty((nModes, 1))
                velColumn = np.empty((nModes, 1))
                accColumn = np.empty((nModes, 1))
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
                    velColumn[iMode] = float(line.pop(0))
                    accColumn[iMode] = float(line.pop(0))
                self.U = np.append(self.U, newColumn, axis=1)
                self.Udot = np.append(self.Udot, velColumn, axis=1)
                self.Uddot = np.append(self.Uddot, accColumn, axis=1)
            print('Completed reading')


def main(cfgFile = None):
    if cfgFile is None:
        with open('ROM.dictionary', 'rb') as pickle_file:
            pickle_object = pickle.load(pickle_file)
            ROM = pickle_object["ROM"]
            configuration = pickle_object["configuration"]
    else:
        configuration = readConfig(cfgFile)

        # Create the physical model
        model = aerodynamics.physicalModel(configuration["NORMALS"], configuration["MODES"])

        # Gather the database
        database = aerodynamics.database(configuration["STRUCT_HISTORY"][i],
                                            configuration["AERO_HISTORY"][i], configuration["THRESHOLDING"])

        # Build the ROM
        ROM = aerodynamics.ROM(database, model, configuration["STABILISATION"])

        # Dump the ROM for future use, without the need to rebuild it
        with open('ROM.dictionary', 'wb') as pickle_file:
            pickle_object = {"ROM": ROM, "configuration": configuration}
            pickle.dump(pickle_object, pickle_file)

    if configuration["IMPOSED_MOTION"] == "YES":
        # Build the future inputs class
        inputs = inputClass(configuration["INPUTS"], ROM.database.deltaT)

        # Run the prediction step
        forces = np.empty((0), dtype=float)
        for i in range(inputs.U.shape[1]):
            stdout.write("\rTime iteration " + str(i + 1) + " of " + str(inputs.U.shape[1]))
            stdout.flush()
            aeroState = ROM.predict(inputs.U[:, i], inputs.Udot[:, i], inputs.Uddot[:, i])
            forces = np.append(forces, ROM.getLift())
            ROM.update()
        print("\nCompleted time integration")

        # Print the obtained lift to file
        with open(configuration["OUTPUTS"], 'w') as file:
            file.write("Lift\n")
            for i in range(len(forces)):
                file.write(str(forces[i])+'\n')

        print("The output is stored in file "+configuration["OUTPUTS"])
    else:
        os.chdir("..")
        import structure
        os.chdir("Tutorials")
        solverConfiguration = {"N_MODES": ROM.nmodes, "DELTA_T": ROM.deltaT, "MODAL_DAMP": configuration["MODAL_DAMP"],
                               "PUNCH_FILE": configuration["PUNCH_FILE"], "INITIAL_MODES": ROM.database.Uinit,
                               "INITIAL_VEL": ROM.database.Udotinit, "OUTPUTS": configuration["OUTPUTS"]}
        solver = structure.solver(solverConfiguration)
        solver.writeHeader()
        solver.writeSolution()
        for timeIter in range(1, int(configuration["TIME_ITER"])+1):
            aeroState = ROM.predict(solver.q, solver.qdot, solver.qddot)
            solver.applyload(np.array(ROM.getModalforces()))
            solver.run()
            solver.updateSolution()
            solver.writeSolution()
            ROM.update()
        print("\nCompleted time integration")

def readConfig(cfgFile):
    input_file = open(cfgFile)
    configuration = {}
    while True:
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

        if this_param == "STRUCT_HISTORY" or this_param == "AERO_HISTORY":
            configuration[this_param] = eval(this_value)
        elif this_param == "MODAL_DAMP":
            configuration[this_param] = float(this_value)
        elif this_param == "THRESHOLDING":
            try:
                configuration[this_param] = int(this_value)
            except ValueError:
                if this_value == "OPTIMAL":
                    configuration[this_param] = 0
                elif this_value == "INTERACTIVE":
                    configuration[this_param] = -1
                else:
                    raise Exception("Thresholding type not recognised")
        elif this_param == "STABILISATION":
            try:
                configuration[this_param] = float(this_value)
            except ValueError:
                if this_value == "FLIP":
                    configuration[this_param] = 1
                elif this_value == "BRUNTON":
                    configuration[this_param] = 2
                else:
                    configuration[this_param] = 0
        else:
            configuration[this_param] = this_value

    return configuration


if __name__ == '__main__':
    main()
