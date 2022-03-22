#!/usr/bin/env python

#  Imports

import numpy as np
import scipy.linalg as linalg
import math
from sys import stdout

class solver:
    """
    Structural solver main class.
    It contains all the required methods for the coupling with SU2.
    """

    def __init__(self, configuration):
        """
        Constructor of the structural solver class.
        """
        print("\n")
        print("Configuring the structural solver for FSI simulation")
        self.nmodes = configuration["N_MODES"]
        self.deltaT = configuration["DELTA_T"]
        self.outputFile = configuration["OUTPUTS"]
        self.modalDamping = configuration["MODAL_DAMP"]
        if self.modalDamping == 0:
            print("The structural model is undamped")
        else:
            print("Assuming {}% of modal damping".format(self.modalDamping * 100))
        self.punchFile = configuration["PUNCH_FILE"]

        print("Creating the structural model")
        self.__setStructuralMatrices()

        print("Setting the integration parameters")
        self.__setIntegrationParameters()
        self.__setInitialConditions(configuration["INITIAL_MODES"], configuration["INITIAL_VEL"])

    def __setStructuralMatrices(self):
        """
        This method reads the punch file and obtains the modal shapes and modal stiffnesses.
        """

        self.M = np.zeros((self.nmodes, self.nmodes))
        self.K = np.zeros((self.nmodes, self.nmodes))
        self.C = np.zeros((self.nmodes, self.nmodes))

        self.q = np.zeros((self.nmodes, 1))
        self.qdot = np.zeros((self.nmodes, 1))
        self.qddot = np.zeros((self.nmodes, 1))
        self.a = np.zeros((self.nmodes, 1))

        self.q_n = np.zeros((self.nmodes, 1))
        self.qdot_n = np.zeros((self.nmodes, 1))
        self.qddot_n = np.zeros((self.nmodes, 1))
        self.a_n = np.zeros((self.nmodes, 1))

        self.F = np.zeros((self.nmodes, 1))

        with open(self.punchFile, 'r') as punchfile:
            print('Opened punch file ' + self.punchFile + '.')
            while True:
                line = punchfile.readline()
                if not line:
                    break

                pos = line.find('MODE ')
                if pos != -1:
                    line = line.strip('\r\n').split()
                    n = int(line[5])
                    imode = n - 1
                    k_i = float(line[2])
                    self.M[imode][imode] = 1
                    self.K[imode][imode] = k_i
                    w_i = math.sqrt(k_i)
                    self.C[imode][imode] = 2 * self.modalDamping * w_i
                    if n == self.nmodes:
                        break

        self.__setNonDiagonalStructuralMatrices()

        if n < self.nmodes:
            raise Exception('Available {} degrees of freedom instead of {} as requested'.format(n, self.nmodes))
        else:
            print('Using {} degrees of freedom'.format(n))

    def __setNonDiagonalStructuralMatrices(self):
        """
        This method is part of an advanced feature of this solver that allows to set
        nondiagonal matrices for the structural modes.
        """

        K_updated = self.__readNonDiagonalMatrix('NDK')
        M_updated = self.__readNonDiagonalMatrix('NDM')
        C_updated = self.__readNonDiagonalMatrix('NDC')
        if K_updated and M_updated and (not C_updated):
            print('Setting modal damping')
            self.__setNonDiagonalDamping()
        elif (not K_updated) and (not M_updated):
            print('Modal stiffness and mass matrices are diagonal')
        elif (not K_updated) and M_updated:
            raise Exception('Non-Diagonal stiffness matrix is missing')
        elif (not M_updated) and K_updated:
            raise Exception('Non-Diagonal mass matrix is missing')

    def __readNonDiagonalMatrix(self, keyword):
        """
        This method reads from the punch file the definition of nondiagonal structural
        matrices.
        """

        matrixUpdated = False

        with open(self.punchFile, 'r') as punchfile:

            while 1:
                line = punchfile.readline()
                if not line:
                    break

                pos = line.find(keyword)
                if pos != -1:
                    while 1:
                        line = punchfile.readline()
                        line = line.strip('\r\n').split()
                        if line[0] != '-CONT-':
                            i = int(line[0]) - 1
                            j = 0
                            el = line[1:]
                            ne = len(el)
                        elif line[0] == '-CONT-':
                            el = line[1:]
                            ne = len(el)
                        if keyword == 'NDK':
                            self.K[i][j:j + ne] = np.array(el)
                        elif keyword == 'NDM':
                            self.M[i][j:j + ne] = np.array(el)
                        elif keyword == 'NDC':
                            self.C[i][j:j + ne] = np.array(el)
                        j = j + ne
                        if i + 1 == self.nmodes and j == self.nmodes:
                            matrixUpdated = True
                            break

        return matrixUpdated

    def __setNonDiagonalDamping(self):

        D, V = linalg.eig(self.K, self.M)
        D = D.real
        D = np.sqrt(D)
        Mmodal = ((V.transpose()).dot(self.M)).dot(V)
        Mmodal = np.diag(Mmodal)
        C = 2 * self.modalDamping * np.multiply(D, Mmodal)
        C = np.diag(C)
        Vinv = linalg.inv(V)
        C = C.dot(Vinv)
        VinvT = Vinv.transpose()
        self.C = VinvT.dot(C)

    def __setIntegrationParameters(self):
        """
        This method uses the time step size to define the integration parameters.
        """
        self.rhoAlphaGen = 0.5
        self.alpha_m = (2.0 * self.rhoAlphaGen - 1.0) / (self.rhoAlphaGen + 1.0)
        self.alpha_f = self.rhoAlphaGen / (self.rhoAlphaGen + 1.0)
        self.gamma = 0.5 + self.alpha_f - self.alpha_m
        self.beta = 0.25 * (self.gamma + 0.5) ** 2

        self.gammaPrime = self.gamma / (self.deltaT * self.beta)
        self.betaPrime = (1.0 - self.alpha_m) / ((self.deltaT ** 2) * self.beta * (1.0 - self.alpha_f))

    def __setInitialConditions(self, initialModes, initialVel):
        """
        This method uses the list of initial modal amplitudes to set the initial conditions
        """

        print('Setting initial conditions.')
        self.q = initialModes
        self.qdot = initialVel

        RHS = np.zeros((self.nmodes, 1))
        RHS += self.F
        RHS -= self.C.dot(self.qdot)
        RHS -= self.K.dot(self.q)
        self.qddot = linalg.solve(self.M, RHS)
        self.qddot_n = np.copy(self.qddot)
        self.a = np.copy(self.qddot)
        self.a_n = np.copy(self.qddot)

        self.timeIter = 0

    @staticmethod
    def __reset(vector):
        """
        This method sets to zero any vector.
        """

        for ii in range(vector.shape[0]):
            vector[ii] = 0.0

    def run(self):
        """
        This method is the main function for advancing the solution of one time step.
        """
        self.__temporalIteration()
        line = '\r{}'.format(self.timeIter) + '\t'
        for imode in range(min([self.nmodes, 5])):
            line = line + '{:6.4f}'.format(float(self.q[imode])) + '\t' + '{:6.4f}'.format(
                float(self.qdot[imode])) + '\t' + '{:6.4f}'.format(float(self.qddot[imode])) + '\t'
        stdout.write(line)
        stdout.flush()

    def __temporalIteration(self):
        """
        This method integrates in time the solution.
        """

        self.__reset(self.q)
        self.__reset(self.qdot)
        self.__reset(self.qddot)
        self.__reset(self.a)

        eps = 1e-6

        # Prediction step

        self.a += self.alpha_f / (1 - self.alpha_m) * self.qddot_n
        self.a -= self.alpha_m / (1 - self.alpha_m) * self.a_n

        self.q = np.copy(self.q_n)
        self.q += self.deltaT * self.qdot_n
        self.q += (0.5 - self.beta) * self.deltaT * self.deltaT * self.a_n
        self.q += self.deltaT * self.deltaT * self.beta * self.a

        self.qdot = np.copy(self.qdot_n)
        self.qdot += (1 - self.gamma) * self.deltaT * self.a_n
        self.qdot += self.deltaT * self.gamma * self.a

        # Correction step
        res = self.M.dot(self.qddot) + self.C.dot(self.qdot) + self.K.dot(self.q) - self.F

        while linalg.norm(res) >= eps:
            St = self.betaPrime * self.M + self.gammaPrime * self.C + self.K
            Deltaq = -1 * (linalg.solve(St, res))
            self.q += Deltaq
            self.qdot += self.gammaPrime * Deltaq
            self.qddot += self.betaPrime * Deltaq
            res = self.M.dot(self.qddot) + self.C.dot(self.qdot) + self.K.dot(self.q) - self.F

        self.a += (1 - self.alpha_f) / (1 - self.alpha_m) * self.qddot

    def writeScreen(self):
        """
        This method only writes an header that will be used during the time integration
        """
        line = 'Time iter\t'
        for imode in range(min([self.nmodes, 5])):
            line = line + 'q' + str(imode + 1) + '\t' + 'qdot' + str(imode + 1) + '\t' + 'qddot' + str(
                imode + 1) + '\t'
        print(line)

    def writeSolution(self):
        """
        This method is the main function for output. It writes the file StructHistoryModal.dat
        """

        # Modal History
        histFile = open(self.outputFile, "a")
        line = str(self.timeIter) + '\t'
        for imode in range(self.nmodes):
            line = line + str(float(self.q[imode])) + '\t' + str(float(self.qdot[imode])) + '\t' + str(
                float(self.qddot[imode])) + '\t'
        line = line + '\n'
        histFile.write(line)
        histFile.close()

    def updateSolution(self):
        """
        This method updates the solution.
        """

        self.q_n = np.copy(self.q)
        self.qdot_n = np.copy(self.qdot)
        self.qddot_n = np.copy(self.qddot)
        self.a_n = np.copy(self.a)
        self.__reset(self.q)
        self.__reset(self.qdot)
        self.__reset(self.qddot)
        self.__reset(self.a)

        self.timeIter += 1

    def applyload(self, forces):
        """
        This method can be accessed from outside to set the modal forces.
        """
        if len(forces) != self.nmodes:
            raise Exception('Number of modal forces not equal to number of modes')

        self.F = forces
