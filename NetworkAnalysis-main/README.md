# OmicsAnalysis_tf2
![Coverage](./pytests/Coverage/coverage.svg)

## FORKED FROM ORIGINAL REPO
**This repo will be transferred from tensorflow 1.13.1 to 2.4.**
### installation instructions
It is strongly advised to use conda for this package, as it relies on older versions of some dependencies.
 
Once conda is properly installed create a conda environment:<br/>

 ````
conda create -n <OmicsAnalysis> python=3.8
 ````

Activate the conda environment using:<br/>
 ````
conda activate <OmicsAnalysis>
 ````


Clone this repository and enter it:<br/>

 ````
git clone https://github.ugent.be/mlarmuse/OmicsAnalysis.git
cd OmicsAnalysis/
 ````

 
Once inside run:

 ````
pip install .
 ````

 
Due to some inconsistency with the packages you have to run:

 ````
pip uninstall gensim
pip install gensim==3.8.1 --no-cache-dir
 ````
 
