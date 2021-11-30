#!/usr/bin/env python

# Imports


class physicalModel:
    """
    Class containing the relation between the state of the aerodynamic ROM and
    the actual modal forces. This class uses the normals to each panel and the
    modal shapes to project the output of the ROM to the structural model. The
    normals are dimensional normals, meaning that their norm is equal to the
    panel area
    """

    def __init__(self,filenameNormals,filenameModes):
        print('Creating the physical model.')
        self.filenameNormals = filenameNormals
        self.filenameModes = filenameModes
        self.Normals = None
        self.Shapes = None
        print('Importing the data from the files.')
        self.__readFileNormals()
        self.__readFileModes()
        print('Done')

    def __readFileNormals():
        
