__author__ = 'Haohan Wang'

import sys
import numpy as np
import pysnptools

class FileReader():
    def __init__(self, fileType, fileName):
        self.fileType = fileType
        self.fileName = fileName

    def readFiles(self):
        if self.fileType == 'plink':
            from pysnptools.snpreader import Bed
            snpreader = Bed(self.fileName+'.bed')
            snpdata = snpreader.read()
            X = snpdata.val
            Xname = snpdata.sid

            from pysnptools.snpreader import Pheno
            phenoreader = Pheno(self.fileName+".fam")
            phenodata = phenoreader.read()
            y = phenodata.val[:,-1]
            return X, y, Xname

        if self.fileType == 'csv':
            X = np.loadtxt(self.fileName+'.geno.csv', delimiter=',')
            y = np.loadtxt(self.fileName+'.pheno.csv', delimiter=',')
            try:
                Xname = np.loadtxt(self.fileName+'.marker.csv', delimiter=',')
            except:
                Xname = ['geno ' + str(i+1) for i in range(X.shape[1])]

            return X, y, Xname