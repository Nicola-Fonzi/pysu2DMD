#!/usr/bin/env python

# Imports
import numpy as np
import os
from scipy import integrate
from sys import stdout

# Class database
class database:
    """
    Class containing all the training data later used for the creation of the ROM.
    It reads the file, store the required info, assemble the matrices, and provide
    the SVD of the matrices.
    """

    def __init__(self, filenameStru, filenameAero, thresholding=0):
        print('\n')
        print('Creating the database for the reduced order model.')
        self.filenameStru = filenameStru          # The file where to read the structural history
        self.filenameAero = filenameAero          # The set of files where to read the aerodynamic history
        self.thresholding = thresholding          # Specifies the way we want to reduce the system dimension
        self.timeIter = np.empty((0), dtype=int)  # The time iterations at which history has been saved
        self.pointID = np.empty((0), dtype=int)   # IDs of the points in the aero history
        self.deltaT = None                        # Time step size
        self.U = None                             # Structural snapshot matrix
        self.Udot = None                          # Structural velocity snapshot matrix
        self.Uddot = None                         # Structural acceleration snapshot matrix
        self.X = None                             # Aero snapshot matrix
        self.Xinit = None                         # Aero state to be used for the initialisation of ROM (last snapshot)
        self.Uinit = None                         # Stru state to be used for the initialisation of ROM (last snapshot)
        self.Udotinit = None                      # Stru velocity
        print('Importing the data from the files.')
        self.__readFileStru()
        self.__readFileAero()
        print('Done')

    def __readFileStru(self):
        with open(self.filenameStru, 'r') as file:
            print('Opened structural history file ' + self.filenameStru + '.')
            headerLine = file.readline()
            headerLine = headerLine.strip('\r\n')
            headerLine = headerLine.split('\t')
            nModes = int((len(headerLine)-3)/3)
            self.U = np.empty((nModes, 0))
            self.Udot = np.empty((nModes, 0))
            self.Uddot = np.empty((nModes, 0))
            timeOld = None
            newColumn = None
            velColumn = None
            while True:
                line = file.readline()
                if not line:
                    break
                newColumn = np.empty((0))
                velColumn = np.empty((0))
                accColumn = np.empty((0))
                line = line.strip('\r\n')
                line = line.split('\t')
                time = float(line.pop(0))
                if not self.deltaT:
                    if not timeOld:
                        timeOld = time
                    else:
                        self.deltaT = time-timeOld
                timeIter = int(line.pop(0))
                self.timeIter = np.append(self.timeIter, [timeIter], axis=0)
                FSIiter = line.pop(0)
                for iMode in range(nModes):
                    newColumn = np.append(newColumn, float(line.pop(0)))
                    velColumn = np.append(velColumn, float(line.pop(0)))
                    accColumn = np.append(accColumn, float(line.pop(0)))
                newColumn = newColumn.reshape((len(newColumn), 1))
                velColumn = velColumn.reshape((len(velColumn), 1))
                accColumn = accColumn.reshape((len(accColumn), 1))
                self.U = np.append(self.U, newColumn, axis=1)
                self.Udot = np.append(self.Udot, velColumn, axis=1)
                self.Uddot = np.append(self.Uddot, accColumn, axis=1)
            self.Uinit = np.copy(self.U[:, 0].reshape((self.U.shape[0], 1)))
            self.Udotinit = np.copy(self.Udot[:, 0].reshape((self.Udot.shape[0], 1)))
            print('Completed reading')

    def __readFileAero(self):
        path, extension = os.path.splitext(self.filenameAero)
        print('Starting the reading of ' + os.path.split(self.filenameAero)[1] + ' files.')
        XtoSet = True
        newColumn = None
        for timeIter in self.timeIter:
            stdout.write("\rOpened time iter " + str(timeIter) + " last time iter is " + str(np.max(self.timeIter)))
            stdout.flush()
            newColumn = np.empty((0))
            fileName = path+'_{:05d}'.format(timeIter)+extension
            with open(fileName, 'r') as file:
                headerLine = file.readline()
                headerLine = headerLine.strip('\r\n')
                headerLine = headerLine.split(',')
                for element in range(len(headerLine)):
                    headerLine[element] = headerLine[element].strip()
                if '"Pressure"' not in headerLine:
                    raise Exception('Pressure must be requested as an output to the fluid solver')
                indexCp = headerLine.index('"Pressure"')
                indexID = headerLine.index('"PointID"')
                while True:
                    line = file.readline()
                    if not line:
                        break
                    line = line.strip('\r\n')
                    line = line.split(',')
                    newColumn = np.append(newColumn, float(line[indexCp]))
                    if XtoSet:
                        self.pointID = np.append(self.pointID, int(line[indexID]))
            if XtoSet:
                self.X = np.empty((len(newColumn), 0))
                XtoSet = False
            newColumn = newColumn.reshape((len(newColumn), 1))
            self.X = np.append(self.X, newColumn, axis=1)
        print("\nCompleted reading")
        self.Xinit = np.copy(self.X[:, 0].reshape((self.X.shape[0], 1)))

    def getSVD(self, brunton):
        Xcenter = self.X[:, 0]
        Xcenter = Xcenter.reshape((len(Xcenter), 1))

        if brunton is True:
            import matplotlib.pyplot as plt
            nModes = self.U.shape[0]
            referenceSnapshots = np.empty((self.X.shape[0], nModes+1))
            referenceSnapshots[:, 0] = self.X[:, 0]
            referenceSnapshotsSlope = np.empty((self.X.shape[0], nModes))
            stateToSubtract = np.zeros((self.X.shape[0], self.X.shape[1]))
            for i in range(nModes):
                plt.plot(np.array([x for x in range(self.U.shape[1])]), self.U[i, :])
            plt.show()
            for i in range(nModes):
                print("Please select steady state for mode {}:".format(i))
                reference = int(input())
                referenceSnapshots[:, i+1] = self.X[:, reference]
                referenceSnapshotsSlope[:, i] = (referenceSnapshots[:, i+1] - referenceSnapshots[:, i])/self.U[i, reference]
                for j in range(self.X.shape[1]):
                    stateToSubtract[:, j] = stateToSubtract[:, j] + referenceSnapshotsSlope[:, i]*self.U[i, j]
            import time
            time.sleep(10)
            U, S, VT = self.__performSVD(self.X[:, :-1] - Xcenter - stateToSubtract[:, :-1],
                                         self.Udot[:, :-1], self.Uddot[:, :-1])
            Up, Sp, VTp = self.__performSVD(self.X[:, 1:] - Xcenter - stateToSubtract[:, 1:])
            return Up, Sp, VTp, Xcenter, U, S, VT, referenceSnapshotsSlope, stateToSubtract[:, 1:]
        else:
            U, S, VT = self.__performSVD(self.X[:, :-1] - Xcenter, self.U[:, :-1], self.Udot[:, :-1],
                                         self.Uddot[:, :-1])
            Up, Sp, VTp = self.__performSVD(self.X[:, 1:] - Xcenter)
            return Up, Sp, VTp, Xcenter, U, S, VT, None, None

    def __performSVD(self, M1, M2=None, M3=None, M4=None):
        if M2 is not None:
            M = np.append(M1, M2, axis=0)
            M = np.append(M, M3, axis=0)
            if M4 is not None:
                M = np.append(M, M4, axis=0)
                U, S, VT = np.linalg.svd(M, full_matrices=False)
            else:
                U, S, VT = np.linalg.svd(M, full_matrices=False)
            beta = M.shape[1]/M.shape[0]
        else:
            U, S, VT = np.linalg.svd(M1, full_matrices=False)
            beta = M1.shape[1]/M1.shape[0]

        if self.thresholding == 0:
            tsh = self.getOptimalThreshold(beta, S)
            cut = (S > tsh).argmin() - 1
        elif self.thresholding > 0:
            cut = self.thresholding
        else:
            from matplotlib import pyplot as plt
            n = len(S) + 1
            tsh = self.getOptimalThreshold(beta, S)
            plt.plot(range(1, n), S)
            plt.plot(range(1, n), np.ones(shape=(n - 1, 1), dtype=float) * tsh)
            plt.show()
            cut = int(input("Please enter the required cutting point."
                            "The horizontal line represent the 'optimal' one:"))

        U = U[:, :cut]
        S = S[:cut]
        VT = VT[:cut, :]

        return U, np.diag(S), VT

    def getOptimalThreshold(self, beta, Sig):
        w = (8 * beta) / (beta + 1 + np.sqrt(beta**2 + 14 * beta +1))
        lambda_star = np.sqrt(2 * (beta + 1) + w)
        mpMedian = self.__medianMarcenkoPastur(beta)
        coef = lambda_star / np.sqrt(mpMedian)
        return coef*np.median(Sig)

    def __medianMarcenkoPastur(self, beta):
        lobnd = (1 - np.sqrt(beta))**2
        hibnd = (1 + np.sqrt(beta))**2
        change = 1
        while change and (hibnd - lobnd > .001):
            change = 0
            x0 = np.linspace(lobnd, hibnd, 5)
            y = np.empty((len(x0)))
            for i in range(len(x0)):
                y[i] = self.__marPas(x0[i], beta)
            if any(y < 0.5):
                lobnd = max(x0[y < 0.5])
                change = 1
            if any(y > 0.5):
                hibnd = min(x0[y > 0.5])
                change = 1
        return (hibnd+lobnd)/2

    def __marPas(self, x0, beta):
        if beta > 1:
            raise Exception('Snapshot matrix must be such that number of columns is less than number of rows')
        top = (1 + np.sqrt(beta))**2
        bot = (1 - np.sqrt(beta))**2
        incMarPas, err = integrate.quad(self.__integrationFun, x0, top, args=(top, bot, beta,))
        return 1-incMarPas

    @staticmethod
    def __integrationFun(x, top, bot, beta):
        Q = (top-x)*(x-bot) > 0
        point = np.sqrt((top-x)*(x-bot))/(beta * x)/(2 * np.pi)
        y = point
        if not Q:
            y = 0
        return y
