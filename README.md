PD.py peridynamics example code
===============================

_PD.Py_ is a 2D explicit-time integration code that serves as an example code
of how one might write a peridynamics code in parallel with PyTrilinos.

To clone the repo:

````
git clone 
````

### External dependencies ###
[PyTrilinos](http://trilinos.sandia.gov/packages/pytrilinos/)

## Python packages available via `pip` ##
NumPy
SciPy
matplotlib
progressbar

These can be installed with the `requirements.txt` file

````
pip install -r requirements.txt
````

To run the code:

````
mpiexec -np 4 python PD.py
````

where `4` can be replaced with any arbitrary number of processsors.

The results can be viewed in parallel with [Paraview](http://www.paraview.org/)

