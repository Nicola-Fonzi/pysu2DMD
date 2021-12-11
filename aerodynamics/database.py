#!/usr/bin/env python

# Imports
import numpy as np
import os
from scipy import integrate

# Class database
class database:
    """
    Class containing all the training data later used for the creation of the ROM.
    It reads the file, store the required info, assamble the matrices, and provide
    the SVD of the matrices.
    """

    def __init__(self, filenameStru, filenameAero):
        print('Creating the database for the reduced order model.')
        self.filenameStru = filenameStru          # The file where to read the structural history
        self.filenameAero = filenameAero          # The set of files where to read the aerodynamic history
        self.timeIter = np.empty((0), dtype=int)  # The time iterations at which history has been saved
        self.pointID = np.empty((0), dtype=int)   # IDs of the points in the aero history
        self.deltaT = None                        # Time step size
        self.U = None                             # Structural snapshot matrix
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
            timeOld = None
            while True:
                newColumn = np.empty((nModes, 1))
                velColumn = newColumn
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
                timeIter = int(line.pop(0))
                self.timeIter = np.append(self.timeIter, [timeIter], axis=0)
                FSIiter = line.pop(0)
                for iMode in range(nModes):
                    newColumn = np.append(newColumn, float(line.pop(0)))
                    velColumn = np.append(velColumn, float(line.pop(0)))
                    acc = float(line.pop(0))
                self.U = np.append(self.U, newColumn, axis=1)
            self.Uinit = newColumn
            self.Udotinit = velColumn
            print('Completed reading')

    def __readFileAero(self):
        path, extension = os.path.splitext(self.filenameAero)
        print('Starting the reading of ' + os.path.split(self.filenameAero)[1] + ' files.')
        XtoSet = True
        for timeIter in self.timeIter:
            newColumn = np.empty((0))
            fileName = path+'_{:05d}'.format(timeIter)+extension
            with open(fileName, 'r') as file:
                headerLine = file.readline()
                headerLine = headerLine.strip('\r\n')
                headerLine = headerLine.split(',')
                for element in range(len(headerLine)):
                    headerLine[element] = headerLine[element].strip()
                if '"Pressure_Coefficient"' not in headerLine:
                    raise Exception('Pressure_Coefficient must be requested as an output to the fluid solver')
                indexCp = headerLine.index('"Pressure_Coefficient"')
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
        self.Xinit = newColumn
        print('Completed reading')

    def getShiftedStateSVD(self, plot=False):
        Xmean = np.mean(self.X[:, 1:], axis=1)
        Xmean = Xmean.reshape((len(Xmean), 1))

        U, S, VT = self.__performSVD(plot, self.X[:, 1:] - Xmean)

        return U, S, VT, Xmean

    def getModifiedStateSVD(self, plot=False):
        Xmean = np.mean(self.X[:, :-1], axis=1)
        Xmean = Xmean.reshape((len(Xmean), 1))
        Umean = np.mean(self.U[:, -1], axis=1)
        Umean = Umean.reshape((len(Umean), 1))

        U, S, VT = self.__performSVD(plot, self.X[:, 1:] - Xmean, self.U[:, 1:] - Umean)

        return U, S, VT, Xmean, Umean

    def __performSVD(self, plot, M1, M2=None):
        if M2:
            U, S, VT = np.linalg.svd(np.append(M1, M2, axis=0), full_matrices=False)
        else:
            U, S, VT = np.linalg.svd(M1, full_matrices=False)

        tsh = self.getOptimalThreshold(S)
        cut = (np.diag(S) > tsh).argmax() - 1
        U = U[:, :cut]
        S = S[:cut, :cut]
        VT = VT[:cut, :]

        if plot:
            from matplotlib import pyplot as plt
            n = len(np.diag(S)) + 1
            plt.stackplot(range(1, n), np.diag(S), np.ones(shape=(1, n), dtype=float) * tsh)
            plt.show()

        return U, S, VT

    def getOptimalThreshold(self, M):
        beta = M.shape[1]/M.shape[0]
        w = (8 * beta) / (beta + 1 + np.sqrt(beta**2 + 14 * beta +1))
        lambda_star = np.sqrt(2 * (beta + 1) + w)
        mpMedian = self.__medianMarcenkoPastur(beta)
        coef = lambda_star / np.sqrt(mpMedian)
        return coef*np.median(np.diag(M))

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
