__author__ = 'Haohan Wang'

# Main file for usage of Precision Lasso
# Cite information:
# Wang H, Lengerich BJ, Aragam B, Xing EP. Precision Lasso: Accounting for Correlations and Linear Dependencies in High-Dimensional Genomic Data. Bioinformatics. 2017
#

def printOutHead(): out.write("\t".join(["RANK", "SNP_ID", "BETA_ABS"]) + "\n")


def outputResult(rank, id, beta):
    out.write("\t".join([str(x) for x in [rank, id, beta]]) + "\n")


from optparse import OptionParser, OptionGroup

usage = """usage: %prog [options] --[tfile | bfile | gcsv] genotypeFile --[pfile | pcsv ] phenotypeFile  outfileBase
This program provides the basic usage to precision lasso
      python runPL.py -v --bfile plinkFile --phenofile plinkFormattedPhenotypeFile resultFile
	    """
parser = OptionParser(usage=usage)

dataGroup = OptionGroup(parser, "Data Options")
modelGroup = OptionGroup(parser, "Model Options")
advancedGroup = OptionGroup(parser, "Advanced Parameter Options")

## data options

dataGroup.add_option("--pfile", dest="pfile",
                 help="The base for a PLINK ped file")
dataGroup.add_option("--tfile", dest="tfile",
                      help="The base for a PLINK tped file")
dataGroup.add_option("--bfile", dest="bfile",
                      help="The base for a PLINK binary bed file")
dataGroup.add_option("--phenofile", dest="phenoFile",
                      help="The plink phenotype file")
dataGroup.add_option("--gcsv", dest="gCSVFile",
                      help="genomic variables in CSV data format, each row is one sample, each column is one variable")
dataGroup.add_option("--gname", dest="gNameFile",
                     help="name of the corresponding genomic variables, if none given, index will be used.")
dataGroup.add_option("--pcsv", dest="pCSVFile",
                      help="phenotypes in CSV data format, each row is one sample, each column is one phenotype")
dataGroup.add_option("--pname", dest="pNameFile",
                     help="name of the corresponding phenotypes, if none given, index will be used.")

## model options
modelGroup.add_option("--model", dest="model", default="pl",
                      help="choices of the model used, if None given, the Precision Lasso will be run. ")
modelGroup.add_option("--lambda", dest="lmbd",default=1,
                 help="the weight of the penalizer, either lambda or snum must be given.")
modelGroup.add_option("--snum", dest="snum",default=None,
                 help="the number of targeted variables the model selects, either lambda or snum must be given.")

## advanced options
advancedGroup.add_option("--gamma", dest="gamma",default=None,
                 help="gamma parameter of the Precision Lasso, if none given, the Precision Lasso will calculate it automatically")
advancedGroup.add_option("--lr", dest="lr",default=1,
                 help="learning rate of some of the models")


parser.add_option_group(dataGroup)
parser.add_option_group(modelGroup)
parser.add_option_group(advancedGroup)

(options, args) = parser.parse_args()

import sys
import os
import numpy as np
from scipy import linalg
from utility.dataLoader import plink, CSVReader
from utility.modelsImport import modelDict

fileType = 0
IN = None
X = None
Y = None

if len(args) != 1:
    parser.print_help()
    sys.exit()

outFile = args[0]

if not options.tfile and not options.bfile and not options.gCSVFile:
    # if not options.pfile and not options.tfile and not options.bfile:
    parser.error(
        "You must provide at least one PLINK input file base (--tfile or --bfile) or an CSV formatted file (--gcsv).")

# READING PLINK input
if options.verbose: sys.stderr.write("Reading SNP input...\n")
if options.bfile:
    IN = plink(options.bfile, type='b', phenoFile=options.phenoFile, normGenotype=options.normalizeGenotype)
    fileType = 1
elif options.tfile:
    IN = plink(options.tfile, type='t', phenoFile=options.phenoFile, normGenotype=options.normalizeGenotype)
    fileType = 1
elif options.gCSVFile:
    IN = CSVReader(gCSVFile=options.gCSVFile, gName=options.gNameFile, pCSVFile=options.pCSVFile, pName=options.pNameFile)
    fileType = 2
else:
    parser.error("You must provide at least one genotype file")

if options.pfile:
    IN = plink(options.pfile,type='p', phenoFile=options.phenoFile,normGenotype=options.normalizeGenotype)
    fileType = 1
elif options.pCSVFile:
    IN = CSVReader(gCSVFile=options.gCSVFile, gName=options.gNameFile, pCSVFile=options.pCSVFile, pName=options.pNameFile)
    fileType = 2
else:
    parser.error("You must provide at least one phenotype file")

if fileType == 1:
    X = []
    Xname = []
    for snp, id in IN:
        X.append(snp)
        Xname.append(id)
    Y = IN.getPhenos(options.phenoFile)
elif fileType == 2:
    X, Xname = IN.getGenotype()
    Y = IN.getPhenotype()
else:
    sys.stderr.write("Input files missing")
    Xname = None

model, implementation = modelDict[options.model]
if implementation == 1:
    model.setLearningRate(options.lr)
model.setLambda(options.lmbd)

if options.model == 'pl':
    if options.gamma is not None:
        model.setGamma(options.gamma)
    else:
        model.calculateGamma(X)

if options.snum is None:
    model.fit(X, Y)
    beta = model.getBeta()
else:
    betaM = None
    min_lambda = 1e-15
    max_lambda = 1e15
    patience = 50

    iteration = 0

    while min_lambda < max_lambda and iteration < patience:
        iteration += 1
        lmbd = np.exp((np.log(min_lambda)+np.log(max_lambda)) / 2.0)
        # print "Iter:{}\tlambda:{}".format(iteration, lmbd)
        model.setLambda(lmbd)
        if implementation == 1:
            model.setLearningRate(options.lr) # learning rate must be set again every time we run it.
        model.fit(X, Y)
        beta = model.getBeta()

        c = len(np.where(np.abs(beta)>0)[0]) # we choose regularizers based on the number of non-zeros it reports
        # print "# Chosen:{}".format(c)
        if c < options.snum:   # Regularizer too strong
            max_lambda = lmbd
        elif c > options.snum: # Regularizer too weak
            min_lambda = lmbd
            betaM = beta
        else:
            betaM = beta
            break
    beta = betaM

ind = np.where(beta!=0)[0]
bs = beta[ind].tolist()
xname = []
for i in ind:
    xname.append(i)

beta_name = zip(beta, Xname)
bn = sorted(beta_name)
bn.reverse()

out = open(outFile,'w')
printOutHead()

for i in range(len(bn)):
    outputResult(i, bn[i][1], bn[i][0])

out.close()