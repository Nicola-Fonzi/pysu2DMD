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
        self.X = None                                 # Aerodynamic state in reduced coordinates
        self.Xnew = None                              # Next time level aerodynamic state in reduced coordinates
        self.Z = None                                 # Aerodynamic state in physical coordinates
        self.Xcenter = None                           # Reference condition of the physiscal state
        self.referenceSnapshotSlope = None            # Used for the steady part of the forces in case of Brunton stabilisation
        self.Up = None                                # Projection operator
        self.stabilisation = stabilisation            # Treatment of unstable aerodynamic systems
        for i in range(1, len(self.databases)):
            if databases[i].deltaT != self.deltaT:
                raise Exception('Different time step for the different databases')
        self.__createABmatrices()
        self.__setInitialCondition()
        self.__stabilise()
        if not self.model.normals == self.databases[0].pointID.tolist():
            raise Exception('Cps are not defined on the same cells as for the definition of the normals')

    def __createABmatrices(self):
        if len(self.databases) > 1:
            raise Exception('Multiple databases not yet supported')

        for i in range(len(self.databases)):
            Up, Sp, VTp, Xcenter, U, S, VT, referenceSnapshotSlope, stateToSubtract \
                = self.databases[i].getSVD(self.stabilisation == 2) # In the True case we do not use the modal amplitude as input

            n = self.databases[i].X.shape[0]
            U_1 = U[:n, :]
            U_2 = U[n:, :]

            if stateToSubtract is not None:
                self.A = np.linalg.multi_dot([Up.conj().T, self.databases[i].X[:, 1:] - Xcenter - stateToSubtract,
                                              VT.conj().T, np.linalg.inv(S), U_1.conj().T, Up])
                self.B = np.linalg.multi_dot([Up.conj().T, self.databases[i].X[:, 1:] - Xcenter - stateToSubtract,
                                              VT.conj().T, np.linalg.inv(S), U_2.conj().T])
            else:
                self.A = np.linalg.multi_dot([Up.conj().T, self.databases[i].X[:, 1:] - Xcenter,
                                              VT.conj().T, np.linalg.inv(S), U_1.conj().T, Up])
                self.B = np.linalg.multi_dot([Up.conj().T, self.databases[i].X[:, 1:] - Xcenter,
                                              VT.conj().T, np.linalg.inv(S), U_2.conj().T])

            self.Xcenter = np.copy(Xcenter)
            self.referenceSnapshotSlope = referenceSnapshotSlope
            self.Up = np.copy(Up)

    def __setInitialCondition(self):
        if self.stabilisation != 2:
            self.Z = self.databases[0].Xinit - self.Xcenter
        else:
            self.Z = self.databases[0].Xinit - self.Xcenter - self.referenceSnapshotSlope.dot(self.databases[0].Uinit)

        self.Z = self.Z.reshape((len(self.Z),1))
        self.X = self.Up.conj().T.dot(self.Z)
        self.Xnew = np.copy(self.X)

    def __stabilise(self):
        if not self.stabilisation or self.stabilisation == 2:
            return
        resultEig = np.linalg.eig(self.A)
        for i in range(len(resultEig[0])):
            if (np.real(resultEig[0][i])**2 + np.imag(resultEig[0][i])**2) > 1:
                oldReal = np.real(resultEig[0][i])
                oldImag = np.imag(resultEig[0][i])
                # This is the flip method
                if self.stabilisation == 1:
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
                if np.mod(self.stabilisation, 1):  # Check if float
                    if not oldReal:
                        pass
                    else:
                        phase = np.abs(oldImag / oldReal)
                        newReal = np.sqrt((1 - self.stabilisation)/(1 + phase**2))
                        newImag = phase*newReal
                        newReal = np.sign(oldReal)*newReal
                        newImag = np.sign(oldImag)*newImag
                        resultEig[0][i] = newReal + newImag*1j
        self.A = np.real(np.linalg.multi_dot([resultEig[1], np.diag(resultEig[0]),
                          np.linalg.inv(resultEig[1])]))

    def predict(self, inputs_q, inputs_qdot, inputs_qddot):
        inputs_q = inputs_q.reshape((len(inputs_q), 1))
        inputs_qdot = inputs_qdot.reshape((len(inputs_qdot), 1))
        inputs_qddot = inputs_qddot.reshape((len(inputs_qddot), 1))
        if self.stabilisation != 2:
            inputs = np.append(inputs_q, inputs_qdot, axis=0)
            inputs = np.append(inputs, inputs_qddot, axis=0)
            self.Xnew = self.A.dot(self.X) + self.B.dot(inputs)
            self.Z = self.Up.dot(self.Xnew) + self.Xcenter
        else:
            inputs = np.append(inputs_qdot, inputs_qddot, axis=0)
            self.Xnew = self.A.dot(self.X) + self.B.dot(inputs)
            self.Z = self.Up.dot(self.Xnew) + self.Xcenter + self.referenceSnapshotSlope.dot(inputs_q)

        self.Z = self.Z.reshape((len(self.Z), 1))
        return self.Z

    def getModalforces(self):
        forces = self.model.getModalForces(self.Z)
        return forces

    def getLift(self):
        lift = self.model.getLift(self.Z)
        return lift

    def update(self):
        self.X = self.Xnew
