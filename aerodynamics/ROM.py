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
        self.databases = databases
        self.model = model
        self.deltaT = databases[0].deltaT
        self.A = None
        self.B = None
        self.X = None
        self.Xnew = None
        for i in range(1,len(self.databases)):
            if databases[i].deltaT != self.deltaT:
                raise Exception('Different time step for the different databases')
        self.__createABmatrices()
        self.__setInitialCondition()
        if not self.model.normals == self.databases[0].pointID.tolist():
            raise Exception('Cps are not defined on the same cells as for the definition of the normals')


    def __createABmatrices(self):
        if len(self.databases):
            raise Exception('Multiple databases not yet supported')

        for i in range(len(self.databases)):
            U, S, VT, Xmean, Umean = self.databases[i].getModifiedStatePOD()
            Up, Sp, VTp, Xpmean = self.databases[i].getShiftedStatePOD()
            n = self.databases[i].X.shape()[0]
            q = self.databases[i].U.shape()[0]
            U_1 = U[:n, :]
            U_2 = U[n+q:, :]
            self.A = Up.transpose() * VT.transpose() * np.linalg.inv(S) * U_1.transpose() * Up
            self.B = Up.transpose() * VT.transpose() * np.linalg.inv(S) * U_2.transpose()

    def __setInitialCondition(self):
        # TODO we need to treat the different operating conditions and "mix" the initial conditions
        self.X = self.databases[0].Xinit
        self.Xnew = self.X

    def predict(self, inputs):
        # TODO we need to treat the different operating conditions
        self.Xnew = self.A*self.X + self.B*inputs
        forces = self.model.getModalForces(self.Xnew)

        return forces

    def update(self):
        self.X = self.Xnew
