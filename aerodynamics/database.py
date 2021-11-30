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
        self.filenameStru = filenameStru
        self.filenameAero = filenameAero
        self.timeIter = np.empty((0),dtype=int)
        self.deltaT = None
        self.U = None
        self.X = None
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
            self.U = np.empty((nModes,0))
            timeOld = None
            while True:
                newColumn = np.empty((nModes,1))
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
                self.timeIter = np.append(self.timeIter,[timeIter],axis=0)
                FSIIter = int(line.pop(0))
                for iMode in range(nModes):
                    newColumn[iMode] = float(line.pop(0))
                    dummy = line.pop(0)
                    dummy = line.pop(0)
                self.U = np.append(self.U,newColumn,axis=1)
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
                    headerLine[element]=headerLine[element].strip()
                if '"Pressure_Coefficient"' not in headerLine:
                    raise Exception('Pressure_Coefficient must be requested as an output to the fluid solver')
                indexCp = headerLine.index('"Pressure_Coefficient"')
                while True:
                    line = file.readline()
                    if not line:
                        break
                    line = line.strip('\r\n')
                    line = line.split(',')
                    newColumn = np.append(newColumn, float(line[indexCp]))
            if XtoSet:
                self.X = np.empty((len(newColumn),0))
                XtoSet = False
            newColumn = newColumn.reshape((len(newColumn),1))
            self.X = np.append(self.X,newColumn,axis=1)
        print('Completed reading')

    def getUmatrix(self):
        return self.U

    def getXmatrix(self):
        return self.X

    def getTimeIter(self):
        return self.timeIter

    def getDeltaT(self):
        return self.deltaT

    def getStatePOD(self):
        Xmean = np.mean(self.X[:,:-1],axis=1)
        Xmean = Xmean.reshape((len(Xmean),1))
        print(Xmean)
        """U, S, V = np.linalg.svd(self.X[:,:-1]-Xmean)"""


def getOptimalThreshold(M):

    beta = M.shape[1]/M.shape[0]
    w = (8 * beta) / (beta + 1 + np.sqrt(beta**2 + 14 * beta +1))
    lambda_star = np.sqrt(2 * (beta + 1) + w)
    mpMedian = __medianMarcenkoPastur(beta)
    coef = lambda_star / np.sqrt(mpMedian)
    return coef*np.median(np.diag(M))

def __medianMarcenkoPastur(beta):
    lobnd = (1 - np.sqrt(beta))**2
    hibnd = (1 + np.sqrt(beta))**2
    change = 1
    while (change and (hibnd - lobnd > .001)):
        change = 0
        x0 = np.linspace(lobnd,hibnd,5)
        y = np.empty((len(x0)))
        for i in range(len(x0)):
            y[i] = __marPas(x0[i],beta)
        if any(y < 0.5):
            lobnd = max(x0[y < 0.5])
            change = 1
        if any(y > 0.5):
            hibnd = min(x0[y > 0.5])
            change = 1
    return (hibnd+lobnd)/2

def __marPas(x0,beta):
    if beta>1:
        raise Exception('Snapshot matrix must be such that number of columns is less than number of rows')
    top = (1 + np.sqrt(beta))**2
    bot = (1 - np.sqrt(beta))**2
    incMarPas, err = integrate.quad(__integrationFun,x0,top,args=(top,bot,beta,))
    return 1-incMarPas

def __integrationFun(x,top,bot,beta):
    Q = (top-x)*(x-bot)>0
    point = np.sqrt((top-x)*(x-bot))/(beta* x)/(2 * np.pi)
    y = point
    if not Q:
        y = 0
    return y
