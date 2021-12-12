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
        self.Umean = None                             # Mean of the inputs
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
            U, S, VT, Xmean, Umean = self.databases[i].getModifiedStateSVD()
            Up, Sp, VTp, Xpmean = self.databases[i].getShiftedStateSVD()
            n = self.databases[i].X.shape[0]
            q = self.databases[i].U.shape[0]
            U_1 = U[:n, :]
            U_2 = U[n:, :]
            self.A = np.linalg.multi_dot([Up.conj().T, Up, Sp, VTp, VT.conj().T, np.linalg.inv(S), U_1.conj().T, Up])
            self.B = np.linalg.multi_dot([Up.conj().T, Up, Sp, VTp, VT.conj().T, np.linalg.inv(S), U_2.conj().T])
            self.Xmean = Xmean
            self.Umean = Umean
            self.Up = Up

    def __setInitialCondition(self):
        # TODO we need to treat the different operating conditions and "mix" the initial conditions
        self.X = self.Up.conj().T.dot(self.databases[0].Xinit)
        self.Xnew = self.X

    def predict(self, inputs):
        # TODO we need to treat the different operating conditions
        inputs = inputs - self.Umean
        self.Xnew = self.A.dot(self.X) + self.B.dot(inputs)
        forces = self.model.getModalForces(self.Up.dot(self.Xnew)+self.Xmean)

        return forces

    def update(self):
        self.X = self.Xnew
