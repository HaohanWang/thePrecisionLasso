__author__ = 'Haohan Wang'

# Main file for usage of Precision Lasso
# Cite information:
# Wang H, Lengerich BJ, Aragam B, Xing EP. Precision Lasso: Accounting for Correlations and Linear Dependencies in High-Dimensional Genomic Data. Bioinformatics. 2017
#

def printOutHead(): out.write("\t".join(["RANK", "SNP_ID", "BETA_ABS"]) + "\n")


def outputResult(rank, id, beta):
    out.write("\t".join([str(x) for x in [rank, id, beta]]) + "\n")

def KFold(X,y,k=5):
    foldsize = int(X.shape[0]/k)
    for idx in range(k):
        testlst = range(idx*foldsize,idx*foldsize+foldsize)
        Xtrain = np.delete(X,testlst,0)
        ytrain = np.delete(y,testlst,0)
        Xtest = X[testlst]
        ytest = y[testlst]
        yield Xtrain, ytrain, Xtest, ytest


from optparse import OptionParser, OptionGroup

usage = """usage: %prog [options] -t fileType(plink/csv) -n fileName
This program provides the basic usage to precision lasso, e.g:
python runPL.py -t csv -n data/toy
	    """
parser = OptionParser(usage=usage)

dataGroup = OptionGroup(parser, "Data Options")
modelGroup = OptionGroup(parser, "Model Options")
advancedGroup = OptionGroup(parser, "Advanced Parameter Options")

## data options
dataGroup.add_option("-t", dest='fileType', default='plink', help="choices of input file type")
dataGroup.add_option("-n", dest='fileName', help="name of the input file")

## model options
modelGroup.add_option("--model", dest="model", default="pl",
                      help="choices of the model used, if None given, the Precision Lasso will be run. ")
modelGroup.add_option("--lambda", dest="lmbd", default=None,
                      help="the weight of the penalizer, either lambda or snum must be given.")
modelGroup.add_option("--snum", dest="snum", default=None,
                      help="the number of targeted variables the model selects, either lambda or snum must be given.")

## advanced options
advancedGroup.add_option("--gamma", dest="gamma", default=None,
                         help="gamma parameter of the Precision Lasso, if none given, the Precision Lasso will calculate it automatically")
advancedGroup.add_option("--lr", dest="lr", default=1,
                         help="learning rate of some of the models")
modelGroup.add_option('-m', action='store_true', dest='missing', default=False, help='Run without missing genotype imputation')
modelGroup.add_option('-b', action='store_true', dest='logisticFlag', default=False, help='Enable the logistic regression version of Precision Lasso')
parser.add_option_group(dataGroup)
parser.add_option_group(modelGroup)
parser.add_option_group(advancedGroup)

(options, args) = parser.parse_args()

import sys
import os
import numpy as np
from scipy import linalg
from utility.dataLoader import FileReader
from utility.modelsImport import modelDict

fileType = 0
IN = None

if len(args) != 0:
    parser.print_help()
    sys.exit()

outFile = options.fileName + '.output'

reader = FileReader(fileName=options.fileName, fileType=options.fileType, imputation=(not options.missing))
X, Y, Xname = reader.readFiles()


model, implementation = modelDict[options.model]
if implementation == 1:
    model.setLearningRate(float(options.lr))
model.setLogisticFlag(options.logisticFlag)

if options.model == 'pl':
    if options.gamma is not None:
        model.setGamma(float(options.gamma))
    else:
        model.calculateGamma(X)

if options.snum is None and options.lmbd is None:
    min_mse = np.inf
    min_lam = 0
    for i in range(11):
        lam = 10**(i-5)
        model.setLambda(lam)
        model.setLearningRate(options.lr)
        mse = 0
        for Xtrain, ytrain, Xtest, ytest in KFold(X, Y, 5):
            model.fit(Xtrain, ytrain)
            pred = model.predict(Xtest)
            mse += np.linalg.norm(pred - ytest)
        if mse < min_mse:
            min_mse = mse
            min_lam = lam
    model.setLambda(min_lam)
    model.fit(X, Y)
    beta = model.getBeta()

elif options.lmbd is None:
    snum = int(options.snum)
    betaM = None
    min_lambda = 1e-30
    max_lambda = 1e30
    patience = 50

    iteration = 0

    while min_lambda < max_lambda and iteration < patience:
        iteration += 1
        lmbd = np.exp((np.log(min_lambda) + np.log(max_lambda)) / 2.0)
        print ("Iter:{}\tlambda:{}".format(iteration, lmbd))
        model.setLambda(lmbd)
        if implementation == 1:
            model.setLearningRate(options.lr)  # learning rate must be set again every time we run it.
        model.fit(X, Y)
        beta = model.getBeta()

        c = len(np.where(np.abs(beta) > 0)[0])  # we choose regularizers based on the number of non-zeros it reports
        print ("# Chosen:{}".format(c))
        if c < snum:  # Regularizer too strong
            max_lambda = lmbd
        elif c > snum:  # Regularizer too weak
            min_lambda = lmbd
            betaM = beta
        else:
            betaM = beta
            break
    beta = betaM
else:
    model.setLambda(float(options.lmbd))
    model.fit(X, Y)
    beta = model.getBeta()

ind = np.where(beta != 0)[0]
bs = beta[ind].tolist()
xname = []
for i in ind:
    xname.append(i)

beta_name = zip(beta, Xname)
bn = sorted(beta_name)
bn.reverse()

out = open(outFile, 'w')
printOutHead()

for i in range(len(bn)):
    outputResult(i+1, bn[i][1], bn[i][0])

out.close()
