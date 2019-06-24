![Precision](pl.PNG "Precision Lasso")

# Precision Lasso

Implementation of the Precision Lasso in this paper:

   Haohan Wang, Benjamin J Lengerich, Bryon Aragam, Eric P Xing, Precision Lasso: accounting for correlations and linear dependencies in high-dimensional genomic data, Bioinformatics, Volume 35, Issue 7, 01 April 2019, Pages 1181–1187, https://doi.org/10.1093/bioinformatics/bty750

## Introduction

The Precision Lasso is a Lasso variant that is showed to work better compared to other Lasso variants in terms of variable selection when there are correlated and linearly dependent variables existing.

**Replication:** This repository serves for the purpose to guide others to use our tool, if you are interested in the scripts to replicate our results, please contact us and we will share the repository for replication. Contact information is at the bottom of this page. 

## File Structure:

* models/ main method for the Precision Lasso
* utility/ other helper files
* runPL.py main entry point of using the Precision Lasso to work with your own data

## An Example Command:

```
python runPL.py -t csv -n data/toy
```
#### Data Support
* Precision Lasso currently supports CSV and binary PLINK files. 
* Extensions to other data format can be easily implemented through `FileReader` in `utility/dataLoadear`. Feel free to contact us for the support of other data format. 

## Installation (Not Required)
You will need to have numpy and scipy installed on your current system.
You can install precision lasso using pip by doing the following

```
   pip install git+https://github.com/HaohanWang/thePrecisionLasso
```

You can also clone the repository and do a manual install.
```
   git clone https://github.com/HaohanWang/thePrecisionLasso
   python setup.py install
```

## Python Users
Proficient python users can directly call the Precision Lasso with python code, see the example [here](https://github.com/HaohanWang/thePrecisionLasso/blob/master/BasicExample.ipynb)

## Contact
[Haohan Wang](http://www.cs.cmu.edu/~haohanw/)
&middot;
[@HaohanWang](https://twitter.com/HaohanWang)
