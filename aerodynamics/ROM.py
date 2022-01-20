#!/usr/bin/env python

# Imports
import numpy as np

# Class ROM
class ROM:
    """
    Class containing the creation of the ROM. This, in turns, uses the database
    obtained at different operating conditions and the physical model to obtain a reduced
    order model for the aerodynamics.
    In can be evolved in time to obtain a solution.
    It also checks that all the databases have the same deltaT.
    """

    def __init__(self, databases, model, stabilisation=0):
        self.databases = databases                    # Set of aero databases that will be used to build the ROM
        self.model = model                            # Physical aero model
        self.deltaT = databases[0].deltaT             # The time step of the ROM
        self.nmodes = databases[0].Uinit.shape[0]     # Number of modal coordinates
        self.A = None                                 # A matrix, evolving the state
        self.B = None                                 # B matrix, including the structural inputs
        self.X = None                                 # Aerodynamic state
        self.Xnew = None                              # Next time level aerodynamic state
        self.Xmean = None                             # Mean of the physiscal state
        self.Up = None                                # Projection operator
        for i in range(1, len(self.databases)):
            if databases[i].deltaT != self.deltaT:
                raise Exception('Different time step for the different databases')
        self.__createABmatrices()
        self.__setInitialCondition()
        self.__stabilise(stabilisation)
        if not self.model.normals == self.databases[0].pointID.tolist():
            raise Exception('Cps are not defined on the same cells as for the definition of the normals')

    def __createABmatrices(self):
        if len(self.databases) > 1:
            raise Exception('Multiple databases not yet supported')

        for i in range(len(self.databases)):
            U, S, VT = self.databases[i].getModifiedStateSVD()
            Up, Sp, VTp, Xmean = self.databases[i].getShiftedStateSVD()
            n = self.databases[i].X.shape[0]
            U_1 = U[:n, :]
            U_2 = U[n:, :]
            self.A = np.linalg.multi_dot([Up.conj().T, self.databases[i].X[:, 1:] - Xmean, VT.conj().T,
                                          np.linalg.inv(S), U_1.conj().T, Up])
            self.B = np.linalg.multi_dot([Up.conj().T, self.databases[i].X[:, 1:] - Xmean, VT.conj().T,
                                       np.linalg.inv(S), U_2.conj().T])
            self.Xmean = np.copy(Xmean)
            self.Up = np.copy(Up)

    def __setInitialCondition(self, customInitial=None):
        # TODO we need to treat the different operating conditions and "mix" the initial conditions
        if customInitial is not None:
            self.X = self.Up.conj().T.dot(customInitial - self.Xmean)
        else:
            self.X = self.Up.conj().T.dot(self.databases[0].Xinit - self.Xmean)
        self.Xnew = np.copy(self.X)

    def __stabilise(self, stabilisation):
        if not stabilisation:
            return
        resultEig = np.linalg.eig(self.A)
        for i in range(len(resultEig[0])):
            if (np.real(resultEig[0][i])**2 + np.imag(resultEig[0][i])**2) > 1:
                oldReal = np.real(resultEig[0][i])
                oldImag = np.imag(resultEig[0][i])
                # This is the flip method
                if stabilisation == 1:
                    if not oldReal:
                        pass
                    else:
                        phase = np.abs(oldImag / oldReal)
                        newReal = np.sqrt((2 - oldReal**2 - oldImag**2)/(1 + phase**2))
                        newImag = phase*newReal
                        newReal = np.sign(oldReal)*newReal
                        newImag = np.sign(oldImag)*newImag
                        resultEig[0][i] = newReal + newImag*1j
                # This is the fixed value method
                if np.mod(stabilisation, 1):  # Check if float
                    if not oldReal:
                        pass
                    else:
                        phase = np.abs(oldImag / oldReal)
                        newReal = np.sqrt((1 - stabilisation)/(1 + phase**2))
                        newImag = phase*newReal
                        newReal = np.sign(oldReal)*newReal
                        newImag = np.sign(oldImag)*newImag
                        resultEig[0][i] = newReal + newImag*1j
                # This is Brunton method
                if stabilisation == 2:
                    pass  #TODO implement this
        self.A = np.real(np.linalg.multi_dot([resultEig[1], np.diag(resultEig[0]), np.linalg.inv(resultEig[1])]))



    def predict(self, inputs_q, inputs_qdot, inputs_qddot):
        # TODO we need to treat the different operating conditions
        inputs_q = inputs_q.reshape((len(inputs_q), 1))
        inputs_qdot = inputs_qdot.reshape((len(inputs_qdot), 1))
        inputs_qddot = inputs_qddot.reshape((len(inputs_qddot), 1))
        inputs = np.append(inputs_q, inputs_qdot, axis=0)
        inputs = np.append(inputs, inputs_qddot, axis=0)
        self.Xnew = self.A.dot(self.X) + self.B.dot(inputs)
        return self.Up.dot(self.Xnew) + self.Xmean

    def getModalforces(self):
        forces = self.model.getModalForces(self.Up.dot(self.Xnew) + self.Xmean)
        return forces

    def getLift(self):
        lift = self.model.getCl(self.Up.dot(self.Xnew) + self.Xmean)
        return lift

    def update(self):
        self.X = self.Xnew
