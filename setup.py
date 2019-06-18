
from distutils.core import setup

setup(
      name='precisionLasso',
      version='0.99',
      author = "Haohan Wang",
      author_email='haohanw@cs.cmu.edu',
      url = "https://github.com/HaohanWang/thePrecisionLasso",
      description = 'Precision Lasso, Accounting for Correlations and Linear Dependencies in High-Dimensional Genomic Data',
      packages=['models', 'utility'],
      scripts=['runPL.py'],
    )
