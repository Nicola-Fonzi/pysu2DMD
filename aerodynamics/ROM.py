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

    def __init__(self, databases, model):
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
        if not self.model.normals == self.databases[0].pointID.tolist():
            raise Exception('Cps are not defined on the same cells as for the definition of the normals')

    def __createABmatrices(self):
        if len(self.databases) > 1:
            raise Exception('Multiple databases not yet supported')

        for i in range(len(self.databases)):
            U, S, VT = self.databases[i].getModifiedStateSVD()
            Up, Sp, VTp, Xmean = self.databases[i].getShiftedStateSVD()
            n = self.databases[i].X.shape[0]
            q = self.databases[i].U.shape[0]
            U_1 = U[:n, :]
            U_2 = U[n:n+q, :]
            U_3 = U[n+q:, :]
            self.A = np.linalg.multi_dot([Up.conj().T, self.databases[i].X[:, 2:] - Xmean, VT.conj().T,
                                          np.linalg.inv(S), U_1.conj().T, Up])
            B_1 = np.linalg.multi_dot([Up.conj().T, self.databases[i].X[:, 2:] - Xmean, VT.conj().T,
                                          np.linalg.inv(S), U_2.conj().T])
            B_2 = np.linalg.multi_dot([Up.conj().T, self.databases[i].X[:, 2:] - Xmean, VT.conj().T,
                                          np.linalg.inv(S), U_3.conj().T])
            self.B = np.append(B_1, B_2, axis=1)
            self.Xmean = Xmean
            self.Up = Up

    def __setInitialCondition(self, customInitial=None):
        # TODO we need to treat the different operating conditions and "mix" the initial conditions
        if customInitial is not None:
            self.X = self.Up.conj().T.dot(customInitial - self.Xmean)
        else:
            self.X = self.Up.conj().T.dot(self.databases[0].Xinit - self.Xmean)
        self.Xnew = self.X

    def predict(self, inputs_q, inputs_qdot):
        # TODO we need to treat the different operating conditions
        inputs_q = inputs_q.reshape((len(inputs_q), 1))
        inputs_qdot = inputs_qdot.reshape((len(inputs_qdot), 1))
        inputs = np.append(inputs_q, inputs_qdot, axis=0)
        self.Xnew = self.A.dot(self.X) + self.B.dot(inputs)
        return self.Xnew

    def getModalforces(self):
        forces = self.model.getModalForces(self.Up.dot(self.Xnew) + self.Xmean)
        return forces

    def getLift(self):
        lift = self.model.getCl(self.Up.dot(self.Xnew) + self.Xmean)
        return lift

    def update(self):
        self.X = self.Xnew
