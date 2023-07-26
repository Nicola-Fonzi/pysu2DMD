#!/usr/bin/env python

# Imports
import glob
import os


class nodeNormal:
    def __init__(self, ID, nx, ny, nz):
        self.ID = ID
        self.nx = nx
        self.ny = ny
        self.nz = nz

    def __lt__(self, other):
        return self.ID < other.ID

    def __eq__(self, other):
        try:
            int(other)
            return self.ID == other
        except TypeError:
            return self.ID == other.ID

    def __mul__(self, other):
        return self.nx * other, self.ny * other, self.nz * other


class nodeForce(nodeNormal):
    def __mul__(self, other):
        return self.nx * other.ux + self.ny * other.uy + self.nz * other.uz


class nodeShape:
    def __init__(self, ID, ux, uy, uz):
        self.ID = ID
        self.ux = ux
        self.uy = uy
        self.uz = uz

    def __lt__(self, other):
        return self.ID < other.ID

    def __eq__(self, other):
        return self.ID == other.ID

    def __sub__(self, other):
        return self.ux - other.ux, self.uy - other.uy, self.uz - other.uz


class physicalModel:
    """
    Class containing the relation between the state of the aerodynamic ROM and
    the actual modal forces. This class uses the normals to each panel and the
    modal shapes to project the output of the ROM to the structural model. The
    normals are dimensional normals, meaning that their norm is equal to the
    panel area
    """

    def __init__(self, filenameNormals, filenameModes):
        print('Creating the physical model.')
        self.filenameNormals = filenameNormals
        self.filenameModes = filenameModes
        self.normals = []
        self.shapes = []
        self.undeformedShape = []
        print("Importing the data from the files.")
        self.__readFileNormals()
        self.__readFileModes()
        self.__sortPoints()
        print('Done')

    def __readFileNormals(self):
        print('Obtaining the normals to the different panels.')
        with open(self.filenameNormals, 'r') as file:
            while True:
                line = file.readline()
                if not line:
                    break
                line = line.strip("\r\n")
                line = line.split(",")
                for i in range(len(line)):
                    line[i] = line[i].strip('"[')
                    line[i] = line[i].strip(']"')
                self.normals.append(nodeNormal(int(line[0]), -float(line[1]), -float(line[2]), -float(line[3])))
        print('Completed reading')

    def __readFileModes(self):
        print('Obtaining the mode shapes')

        files, undeformedFile = self.__getFiles()

        print('Starting with the undeformed condition')
        with open(undeformedFile, 'r') as file:
            headerLine = file.readline()
            headerLine = headerLine.strip('\r\n')
            headerLine = headerLine.split(',')
            for element in range(len(headerLine)):
                headerLine[element] = headerLine[element].strip()
            if '"x"' not in headerLine or '"y"' not in headerLine or '"z"' not in headerLine:
                raise Exception('Grid positions must be requested as an output to the fluid solver')
            indexX = headerLine.index('"x"')
            indexID = headerLine.index('"PointID"')
            while True:
                line = file.readline()
                if not line:
                    break
                line = line.strip('\r\n')
                line = line.split(',')
                self.undeformedShape.append(nodeShape(int(line[indexID]), float(line[indexX]), float(line[indexX + 1]),
                                                      float(line[indexX + 2])))

        print('Obtain now the deformation due to modes')
        for mode in range(len(files)):
            print('Opened file {} of {}'.format(mode+1, len(files)))
            newColumn = []
            with open(files[mode], 'r') as file:
                headerLine = file.readline()
                headerLine = headerLine.strip('\r\n')
                headerLine = headerLine.split(',')
                for element in range(len(headerLine)):
                    headerLine[element] = headerLine[element].strip()
                if '"x"' not in headerLine or '"y"' not in headerLine or '"z"' not in headerLine:
                    raise Exception('Grid positions must be requested as an output to the fluid solver')
                indexX = headerLine.index('"x"')
                indexID = headerLine.index('"PointID"')
                while True:
                    line = file.readline()
                    if not line:
                        break
                    line = line.strip('\r\n')
                    line = line.split(',')
                    newColumn.append(nodeShape(int(line[indexID]), float(line[indexX]), float(line[indexX + 1]),
                                               float(line[indexX + 2])))
            for i in range(len(newColumn)):
                newColumn[i].ux, newColumn[i].uy, newColumn[i].uz = newColumn[i] - self.undeformedShape[i]
            self.shapes.append(newColumn)
        print('Completed reading')

    def __getFiles(self):
        path, extension = os.path.splitext(self.filenameModes)
        endPaths = glob.glob(path + '*')
        paths = []
        for i in range(len(endPaths)):
            if ".vtu" not in endPaths[i]:
                paths.append(endPaths[i])
        undeformedPath = os.path.join(os.path.split(self.filenameModes)[0], 'Undeformed') + extension
        return paths, undeformedPath

    def __sortPoints(self):
        self.normals.sort()
        for i in range(len(self.shapes)):
            self.shapes[i].sort()
        if not self.normals == self.shapes[0]:
            raise Exception('The cells for normals do not coincide with the cells in the shapes')

    def getModalForces(self, Pressure):
        force = []
        for i in range(len(self.shapes)):
            force.append(0.0)
        for i in range(len(self.shapes)):
            for j in range(len(self.normals)):
                nodalForcex, nodalForcey, nodalForcez = self.normals[j] * Pressure[j]
                force[i] += nodeForce(1, nodalForcex, nodalForcey, nodalForcez) * self.shapes[i][j]
        return force

    def getLift(self, Pressure):
        force = 0.0
        for j in range(len(self.normals)):
            force += self.normals[j].nz * Pressure[j]
        return force
