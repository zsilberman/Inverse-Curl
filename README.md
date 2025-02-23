# Inverse-Curl
Codes to calculate the inverse curl of a vector quantity, as described in Z.J. Silberman, et al., Numerical generation of vector potentials from specified magnetic fields, J. Comp. Phys. 379 (2019) 421-437, https://doi.org/10.1016/j.jcp.2018.12.006

cell_by_cell.C: builds the inverse curl by walking through the grid, creating the inverse curl as it goes

global_linear_algebra.c: sets up the curl operator as a matrix to solve for the inverse curl on the entire grid "all at once" using the methods of linear algegra

kernel_generation.C: used to create a convolution kernel for putting the inverse curl field into the Coulomb gauge after generation by the cell-by-cell method

## External Packages
Here are the external software packages required to run our codes.

To run the Cell-by-Cell Method, you will need:

FFTW: http://www.fftw.org/

To run the Global Linear Algebra Method, you will need:

MUMPS: http://mumps.enseeiht.fr/index.php

To run the kernel generator, you will need:

eigen: http://eigen.tuxfamily.org/index.php?title=Main_Page

## Other Files
sizes.txt: Because of how the codes handle the curl operation, the various arrays in the codes have different sizes. This file describes what those sizes are and how ghost zones are handled in the code.

## Sample Data
The data/ folder contains some sample data that can be used to test the codes. 

The rotor/ subfolder contains input data for the rotor test as described in our paper linked above (and references therein). The dimensions of these data match the current parameters set in cell_by_cell.C

