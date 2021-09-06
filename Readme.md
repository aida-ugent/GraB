# Quantifying and Reducing Imbalance in Networks

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the source code, installation and use instructions for the method presented in the paper: 
*Quantifying and Reducing Imbalance in Networks*.

We provide Python implementations of the complete GraB model. The repository is maintained by Yoosof Mashayekhi (yoosof.mashayekhi(at)ugent.be).

## Installation

Install directly from GitHub with:


$ pip install git+https://github.com/aida-ugent/GraB.git



**Note:** GraB code has been extensively tested and is stable under Python 3.6.9, thus this is the recommendend environment.

data folder:
    data.csv: graph links
    roles.csv: roles with start id, end id and node type in each row. It is used for the block prior
    types.csv: roles with start id, end id and node type in each row. It is not used for the block prior [optional]

for emd solver: follow the steps in https://wihoho.github.io/2013/08/18/EMD-Python.html
