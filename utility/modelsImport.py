__author__ = 'Haohan Wang'

from models.PrecisionLasso import PrecisionLasso as PL

models = [PL()]
names = ['Precision Lasso']
implementation = [2]

modelDict = {'pl':(models[0], implementation[0]),}
