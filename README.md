_PD.py_ peridynamics example code
===============================

_PD.Py_ is a 2D explicit-time integration code that serves as an example code
of how one might write a peridynamics code in parallel with PyTrilinos. The
code is heavily commented to provide as much insight as possible.

To clone the repo:

````
git clone https://github.com/johntfoster/PDpy.git
````

### External dependencies ###
[PyTrilinos](http://trilinos.sandia.gov/packages/pytrilinos/)

#### Python packages available via `pip` ####
NumPy, SciPy, matplotlib, progressbar, mpi4py (required by PyTrilinos)

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

