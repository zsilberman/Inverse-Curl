#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <fftw3.h>
#include <time.h>

//----------------------------------------------------------------------
// B to A Solver Cell-by-Cell Method
// Copyright (C) 2018 Zachary J. Silberman and Joshua A. Faber
//----------------------------------------------------------------------
//This program is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.
//
//This program is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.
//
//You should have received a copy of the GNU General Public License
//along with this program.  If not, see <http://www.gnu.org/licenses/>.
//----------------------------------------------------------------------
// This program takes a given magnetic field B and constructs the
// corresponding magnetic vector potential A. The B field is defined
// on the faces of grid cells, for ease of calculating the divergence
// of B at cell centers, as well as other derivatives. The A field is 
// therefore defined on the edges of the cells so that it is possible 
// to show that B is the curl of A. After making sure the B field is 
// divergenceless, the program builds an A field for each cell in turn 
// in such a way as to ensure that the curl of A is B. The code 
// requires three arguments: the names of the three files that contain 
// Bx, By, and Bz, in that order. See the read_B function for the 
// expected file format. If reading in a file containing the 
// convolution kernel for the gauge transformation, a fourth argument 
// is required: the name of the file containing the kernel. This file 
// must contain a single column: the value of the kernel at each 
// location in the grid, in row-major order, with the additional first 
// line that gives the size of the kernel.
//----------------------------------------------------------------------

using namespace std;

int Nx = 404;
int Ny = 404;
int Nz = 14;
int i, j, k, l;
int initi = 0;               //where to start calculation along i-direction
int initj = 0;               //where to start calculation along j-direction
int initk = 0;               //where to start calculation along k-direction
int cutxindex = 202;         //fixed x-direction index for output (plotting)
int cutyindex = 202;         //fixed y-direction index for output (plotting)
int cutzindex = 7;           //fixed z-direction index for output (plotting)
int zdiffindex = 0;          //difference in fixed values along z (plotting)
int expox = floor(log2(Nx+1))+1;
int expoy = floor(log2(Ny+1))+1;
int expoz = floor(log2(Nz+1))+1;
int Nx_fft = pow(2,expox);   //smallest power of 2 strictly greater than Nx+1
int Ny_fft = pow(2,expoy);   //smallest power of 2 strictly greater than Ny+1
int Nz_fft = pow(2,expoz);   //smallest power of 2 strictly greater than Nz+1
int Nx_fft2 = 2*Nx_fft;
int Ny_fft2 = 2*Ny_fft;
int Nz_fft2 = 2*Nz_fft;
double t_sim;                //restart time of simulation
double step;                 //numerical size of cell
double stepx, stepy, stepz;  //numerical size of cell along each direction
double dx, dy, dz;           //physical size of cell along each direction
double dV;                   //physical volume of cell
double xinit, yinit, zinit;  //beginning of physical grid along each direction
double accerr = 1.0e-7;      //level of accuracy for constraint equations
double poison = 114.0;       //poison value for initializations
double option5 = 0.00;       //see options for calculation in calcA5faces
double option4 = 1.00;       //see options for calculation in calcA4faces
int option3 = 0;             //see options for calculation in calcA3faces
double comb3 = 0.00;         //option for linear combination of options 5 & 6 in calcA3faces
                             // 0.00: pure option 5 
                             // 1.00: pure option 6
int print_flag = 3;          //if and when to print the B=curl(A) check
                             // 1: print B=curl(A) check before gauge transformation
                             // 2: print B=curl(A) check after gauge transformation
                             // 3: do not print B=curl(A) check
int kernel_flag = 2;         //option for kernel in gauge transformation
                             // 1: 1/r kernel 
                             // 2: kernel from file
int ngauge = 1;              //number of times to run the gauge transformation (usually 1)
int test_flag = 1;           //option for the test being run, used to set up the grid
                             // 1: Rotor Test
                             // 2: TOV stars Test
int dim = 2;                 //dimension of data
                             // 2: Code copies data along the z-axis after reading
                             // 3: Do nothing special after reading
int n_ghost = 3;             //number of ghost cells at the edge of the grid
int ext_order_flag = 3;      //order of extrapolation for ghost cells of A
                             // 0: copy tangential, zero normal
                             // 1: linear tangential, copy normal
                             // 2: quadratic tangential, linear normal
                             // 3: cubic tangential, quadratic normal
double self_convolution = 3.1;
                             //the contribution from a grid cell to itself in the 
                             //convolution various arguments suggest the value 
                             //should be approximately 3, possibly things like 
                             //11/2-2sqrt(2) = 2.67 and 2pi/3+1 = 3.1

//Functions for setup and input/output

void grid_setup (int test_flag);
void read_B (double* Bx,double* By,double* Bz,
	     double* Bx0,double* By0,double* Bz0,
	     char *argv[]);
void out_B (double* Bx,double* By,double* Bz,int flag);
void out_A (double* Ax,double* Ay,double* Az,int flag);
void out_A_IGRMHD (double* Ax,double* Ay,double* Az);
void out_diffs (double* Bx,double* By,double* Bz,
		double* Bx0,double* By0,double* Bz0,
		double* Ax,double* Ay,double* Az);

//Functions that check to make sure the program is working
//and manipulate the various quantities

bool divB (double* Bx,double* By,double* Bz,int flag);
double divB_cell (double Bxm,double Bxp,double Bym,double Byp,double Bzm,double Bzp);
void divA (double* Ax,double* Ay,double* Az,double* div_A,int flag);
void laplacian (int Nx,int Ny,int Nz,double* array);
void curl (int Nx,int Ny,int Nz,double* Ax,double* Ay,double* Az,double* Bx,double* By,double* Bz);
double BcurlA_face (double B,double Amm,double Apm,double Amp,double App,
		    double step12,double step34);
double curlA_cell (double Axmm,double Axpm,double Axmp,double Axpp,
		   double Aymm,double Aypm,double Aymp,double Aypp,
		   double Azmm,double Azpm,double Azmp,double Azpp,
		   double Bxm,double Bxp,double Bym, double Byp,double Bzm,double Bzp);
void curlA_test (double* Bx,double* By,double* Bz,
		 double* Ax,double* Ay,double* Az);
void powspec (double* Ax,double* Ay,double* Az,int flag);
void derivs (double* Ax,double* Ay,double* Az,int flag);
void Coulomb (double* Ax,double* Ay,double* Az,char *argv[]);
void extrap_tests (double* Ax_ext,double* Ay_ext,double* Az_ext,int flag);
void extrap_ghost (double* Ax_ext,double* Ay_ext,double* Az_ext);

//Functions to make sure div(B) = 0

void clean6faces (double B1m,double B1p,double& B1mnew,double& B1pnew,
		  double B2m,double B2p,double& B2mnew,double& B2pnew,
		  double B3m,double B3p,double& B3mnew,double& B3pnew);
void clean5faces (double B1m,double B1p,double& B1pnew,
		  double B2m,double B2p,double& B2mnew,double& B2pnew,
		  double B3m,double B3p,double& B3mnew,double& B3pnew);
void clean4faces (double B1m,double B1p,double& B1pnew,
		  double B2m,double B2p,double& B2pnew,
		  double B3m,double B3p,double& B3mnew,double& B3pnew);
void clean3faces (double B1m,double B1p,double& B1pnew,
		  double B2m,double B2p,double& B2pnew,
		  double B3m,double B3p,double& B3pnew);

//Functions to calculate A

void calcA6faces (double& A1mm,double& A1pm,double& A1mp,double& A1pp,
		  double& A2mm,double& A2pm,double& A2mp,double& A2pp,
		  double& A3mm,double& A3pm,double& A3mp,double& A3pp,
		  double B1m,double B1p,double B2m,double B2p,double B3m,double B3p,
		  double step1,double step2,double step3);
void calcA5faces (double& A1mm,double& A1pm,double& A1mp,double& A1pp,
		  double& A2mm,double& A2pm,double& A2mp,double& A2pp,
		  double& A3mm,double& A3pm,double& A3mp,double& A3pp,
		  double B1m,double B1p,double B2m,double B2p,double B3m,double B3p,
		  double A2mm_prev,double A3mm_prev,
		  double A2pm_prev,double A3mp_prev);
void calcA4faces (double& A1mm,double& A1pm,double& A1mp,double& A1pp,
		  double& A2mm,double& A2pm,double& A2mp,double& A2pp,
		  double& A3mm,double& A3pm,double& A3mp,double& A3pp,
		  double B1m,double B1p,double B2m,double B2p,double B3m,double B3p,
		  double A1mp_prev,double A2pm_prev,double A3mp_prev,
		  double A2mm_prev,double A1mm_prev,double A3pm_prev,double A3mm_prev,
		  double A1pm_prev,double A1pp_prev,double A2mp_prev,double A2pp_prev);
void calcA3faces (double& A1mm,double& A1pm,double& A1mp,double& A1pp,
		  double& A2mm,double& A2pm,double& A2mp,double& A2pp,
		  double& A3mm,double& A3pm,double& A3mp,double& A3pp,
		  double B1m,double B1p,double B2m,double B2p,double B3m,double B3p,
		  double A1mm_prev,double A1pm_prev,double A1mp_prev,
		  double A2mm_prev,double A2pm_prev,double A2mp_prev,
		  double A3mm_prev,double A3pm_prev,double A3mp_prev,
		  double A1pp_prev,double A2pp_prev,double A3pp_prev);

void master (int i,int j,int k,string dir,string sign, 
	     double* Bx,double* By,double* Bz,
	     double* Ax,double* Ay,double* Az);


int main(int argc, char *argv[])
{
  double *Bx0 = new double[(Nx+1)*(Ny)*(Nz)];
  double *By0 = new double[(Nx)*(Ny+1)*(Nz)];
  double *Bz0 = new double[(Nx)*(Ny)*(Nz+1)];

  double *Bx = new double[(Nx+1)*(Ny)*(Nz)];
  double *By = new double[(Nx)*(Ny+1)*(Nz)];
  double *Bz = new double[(Nx)*(Ny)*(Nz+1)];

  double *Ax = new double[(Nx)*(Ny+1)*(Nz+1)];
  double *Ay = new double[(Nx+1)*(Ny)*(Nz+1)];
  double *Az = new double[(Nx+1)*(Ny+1)*(Nz)];

  int bxindex, byindex, bzindex;
  int axindex, ayindex, azindex;
  int bxpindex, bxmindex, bypindex, bymindex, bzpindex, bzmindex;
  int axppindex, axpmindex, axmpindex, axmmindex;
  int ayppindex, aypmindex, aympindex, aymmindex;
  int azppindex, azpmindex, azmpindex, azmmindex;

  double Bxp, Bxm, Byp, Bym, Bzp, Bzm;

  double Axpp, Axpm, Axmp, Axmm;
  double Aypp, Aypm, Aymp, Aymm;
  double Azpp, Azpm, Azmp, Azmm;


  //Declarations for checking the
  //divergences of B and A:

  bool div0;

  double *div_A = new double[(Nx-1)*(Ny-1)*(Nz-1)];

  int divaindex;


  //Declarations for timing the code:

  struct timespec start, end;
  struct timespec gstart, gend;

  double prog_len, gauge_len, net_len;


  //End of Declarations


  //START OF PROGRAM
  clock_gettime(CLOCK_REALTIME, &start);


  //Set up the grid spacing

  grid_setup(test_flag);


  //Read in the magnetic field data.

  read_B(Bx,By,Bz,Bx0,By0,Bz0,argv);


  //Print the original magnetic field

  out_B(Bx0,By0,Bz0,1);


  //Make sure the B field has zero divergence.

  //First, check the divergence.
  div0 = divB(Bx0,By0,Bz0,1);

  //Remove any non-zero divergence
  if(div0 == false)
  {
    cout << "First Divergence Check Failed, Cleaning Field" << endl;

    for(i=0; i<Nx; i++)
    {
      for(j=0; j<Ny; j++)
      {
	for(k=0; k<Nz; k++)
	{
	  bxmindex = k+(Nz)*(j+(Ny)*i);
	  bxpindex = k+(Nz)*(j+(Ny)*(i+1));

	  bymindex = k+(Nz)*(j+(Ny+1)*i);
	  bypindex = k+(Nz)*((j+1)+(Ny+1)*i);

	  bzmindex = k+(Nz+1)*(j+(Ny)*i);
	  bzpindex = (k+1)+(Nz+1)*(j+(Ny)*i);

	  Bxm = Bx[bxmindex];
	  Bxp = Bx[bxpindex];

	  Bym = By[bymindex];
	  Byp = By[bypindex];

	  Bzm = Bz[bzmindex];
	  Bzp = Bz[bzpindex];

	  //Conditions on cells:
	  //i=j=k=0  : nothing fixed
	  //j=k=0    : Bxm fixed
	  //k=i=0    : Bym fixed
	  //i=j=0    : Bzm fixed
	  //k=0      : Bxm, Bym fixed
	  //i=0      : Bym, Bzm fixed
	  //j=0      : Bzm, Bxm fixed
	  //i,j,k!=0 : Bxm, Bym, Bzm fixed
	  //These conditions are used to make sure the
	  //functions are operating properly.

	  if(i == 0 && j == 0 && k == 0)
	    clean6faces(Bxm,Bxp,Bx[bxmindex],Bx[bxpindex],
			Bym,Byp,By[bymindex],By[bypindex],
			Bzm,Bzp,Bz[bzmindex],Bz[bzpindex]);
	  else if(i != 0 && j == 0 && k == 0)
	    clean5faces(Bxm,Bxp,Bx[bxpindex],
			Bym,Byp,By[bymindex],By[bypindex],
			Bzm,Bzp,Bz[bzmindex],Bz[bzpindex]);
	  else if(i == 0 && j != 0 && k == 0)
	    clean5faces(Bym,Byp,By[bypindex],
			Bzm,Bzp,Bz[bzmindex],Bz[bzpindex],
			Bxm,Bxp,Bx[bxmindex],Bx[bxpindex]);
	  else if(i == 0 && j == 0 && k != 0)
	    clean5faces(Bzm,Bzp,Bz[bzpindex],
			Bxm,Bxp,Bx[bxmindex],Bx[bxpindex],
			Bym,Byp,By[bymindex],By[bypindex]);
	  else if(i != 0 && j != 0 && k == 0)
	    clean4faces(Bxm,Bxp,Bx[bxpindex],
			Bym,Byp,By[bypindex],
			Bzm,Bzp,Bz[bzmindex],Bz[bzpindex]);
	  else if(i == 0 && j != 0 && k != 0)
	    clean4faces(Bym,Byp,By[bypindex],
			Bzm,Bzp,Bz[bzpindex],
			Bxm,Bxp,Bx[bxmindex],Bx[bxpindex]);
	  else if(i != 0 && j == 0 && k != 0)
	    clean4faces(Bzm,Bzp,Bz[bzpindex],
			Bxm,Bxp,Bx[bxpindex],
			Bym,Byp,By[bymindex],By[bypindex]);
	  else
	    clean3faces(Bxm,Bxp,Bx[bxpindex],
			Bym,Byp,By[bypindex],
			Bzm,Bzp,Bz[bzpindex]);

	}
      }
    }
  }

  //Check again
  div0 = divB(Bx,By,Bz,2);

  if (div0 == true)
  {
    cout << "divB: Success!" << endl << endl;
  }
  else
  {
    cout << "divB: Failure :(" << endl << endl;
    cout << endl << "Divergence Failure!" << endl;
    //return 0;
  }


  //Print the cleaned magnetic field
  //(might be identical to original)

  out_B(Bx,By,Bz,2);


  //Build the A field.

  //Calculate A for the first cell.
  
  i = initi;
  j = initj;
  k = initk;
    
  master(i,j,k,"","",Bx,By,Bz,Ax,Ay,Az);

  //Calculate A for the 6 coordinate rays from 
  //the first cell to the boundaries.
  
  //-x from first cell
  for (i=initi-1; i>=0; i--)
  {
    j = initj;
    k = initk;

    master(i,j,k,"x","-",Bx,By,Bz,Ax,Ay,Az);
  }

  //+x from first cell
  for (i=initi+1; i<Nx; i++)
  {
    j = initj;
    k = initk;

    master(i,j,k,"x","+",Bx,By,Bz,Ax,Ay,Az);
  }

  //-y from first cell
  for (j=initj-1; j>=0; j--)
  {
    k = initk;
    i = initi;

    master(i,j,k,"y","-",Bx,By,Bz,Ax,Ay,Az);
  }

  //+y from first cell
  for (j=initj+1; j<Ny; j++)
  {
    k = initk;
    i = initi;

    master(i,j,k,"y","+",Bx,By,Bz,Ax,Ay,Az);
  }

  //-z from first cell
  for (k=initk-1; k>=0; k--)
  {
    i = initi;
    j = initj;

    master(i,j,k,"z","-",Bx,By,Bz,Ax,Ay,Az);
  }

  //+z from first cell
  for (k=initk+1; k<Nz; k++)
  {
    i = initi;
    j = initj;

    master(i,j,k,"z","+",Bx,By,Bz,Ax,Ay,Az);
  }
  
  //Calculate A for the 3 coordinate planes that 
  //share the first cell. Each of these planes
  //is divided into 4 regions by the coordinate
  //rays that have already been determined.

  //-x and -y from first cell
  for (i=initi-1; i>=0; i--)
  {
    for (j=initj-1; j>=0; j--)
    {
      k = initk;

      master(i,j,k,"xy","--",Bx,By,Bz,Ax,Ay,Az);
    }
  }

  //+x and -y from first cell
  for (i=initi+1; i<Nx; i++)
  {
    for (j=initj-1; j>=0; j--)
    {
      k = initk;

      master(i,j,k,"xy","+-",Bx,By,Bz,Ax,Ay,Az);
    }
  }

  //-x and +y from first cell
  for (i=initi-1; i>=0; i--)
  {
    for (j=initj+1; j<Ny; j++)
    {
      k = initk;

      master(i,j,k,"xy","-+",Bx,By,Bz,Ax,Ay,Az);
    }
  }

  //+x and +y from first cell
  for (i=initi+1; i<Nx; i++)
  {
    for (j=initj+1; j<Ny; j++)
    {
      k = initk;

      master(i,j,k,"xy","++",Bx,By,Bz,Ax,Ay,Az);
    }
  }


  //-y and -z from first cell
  for (j=initj-1; j>=0; j--)
  {
    for (k=initk-1; k>=0; k--)
    {
      i = initi;

      master(i,j,k,"yz","--",Bx,By,Bz,Ax,Ay,Az);
    }
  }

  //+y and -z from first cell
  for (j=initj+1; j<Ny; j++)
  {
    for (k=initk-1; k>=0; k--)
    {
      i = initi;

      master(i,j,k,"yz","+-",Bx,By,Bz,Ax,Ay,Az);
    }
  }

  //-y and +z from first cell
  for (j=initj-1; j>=0; j--)
  {
    for (k=initk+1; k<Nz; k++)
    {
      i = initi;

      master(i,j,k,"yz","-+",Bx,By,Bz,Ax,Ay,Az);
    }
  }

  //+y and +z from first cell
  for (j=initj+1; j<Ny; j++)
  {
    for (k=initk+1; k<Nz; k++)
    {
      i = initi;

      master(i,j,k,"yz","++",Bx,By,Bz,Ax,Ay,Az);
    }
  }


  //-z and -x from first cell
  for (k=initk-1; k>=0; k--)
  {
    for (i=initi-1; i>=0; i--)
    {
      j = initj;

      master(i,j,k,"zx","--",Bx,By,Bz,Ax,Ay,Az);
    }
  }

  //+z and -x from first cell
  for (k=initk+1; k<Nz; k++)
  {
    for (i=initi-1; i>=0; i--)
    {
      j = initj;

      master(i,j,k,"zx","+-",Bx,By,Bz,Ax,Ay,Az);
    }
  }

  //-z and +x from first cell
  for (k=initk-1; k>=0; k--)
  {
    for (i=initi+1; i<Nx; i++)
    {
      j = initj;

      master(i,j,k,"zx","-+",Bx,By,Bz,Ax,Ay,Az);
    }
  }

  //+z and +x from first cell
  for (k=initk+1; k<Nz; k++)
  {
    for (i=initi+1; i<Nx; i++)
    {
      j = initj;

      master(i,j,k,"zx","++",Bx,By,Bz,Ax,Ay,Az);
    }
  }

  //Calculate A for the remaining cells in the
  //grid, which are divided into 8 regions by
  //the coordinate planes that have already 
  //been determined.

  //-x, -y, and -z from first cell
  for(i=initi-1; i>=0; i--)
  {
    for(j=initj-1; j>=0; j--)
    {
      for(k=initk-1; k>=0; k--)
      {
	master(i,j,k,"xyz","---",Bx,By,Bz,Ax,Ay,Az);
      }
    }
  }

  //-x, +y, and -z from first cell
  for(i=initi-1; i>=0; i--)
  {
    for(j=initj+1; j<Ny; j++)
    {
      for(k=initk-1; k>=0; k--)
      {
	master(i,j,k,"xyz","-+-",Bx,By,Bz,Ax,Ay,Az);
      }
    }
  }

  //-x, -y, and +z from first cell
  for(i=initi-1; i>=0; i--)
  {
    for(j=initj-1; j>=0; j--)
    {
      for(k=initk+1; k<Nz; k++)
      {
	master(i,j,k,"xyz","--+",Bx,By,Bz,Ax,Ay,Az);
      }
    }
  }

  //-x, +y, and +z from first cell
  for(i=initi-1; i>=0; i--)
  {
    for(j=initj+1; j<Ny; j++)
    {
      for(k=initk+1; k<Nz; k++)
      {
	master(i,j,k,"xyz","-++",Bx,By,Bz,Ax,Ay,Az);
      }
    }
  }

  //+x, -y, and -z from first cell
  for(i=initi+1; i<Nx; i++)
  {
    for(j=initj-1; j>=0; j--)
    {
      for(k=initk-1; k>=0; k--)
      {
	master(i,j,k,"xyz","+--",Bx,By,Bz,Ax,Ay,Az);
      }
    }
  }

  //+x, +y, and -z from first cell
  for(i=initi+1; i<Nx; i++)
  {
    for(j=initj+1; j<Ny; j++)
    {
      for(k=initk-1; k>=0; k--)
      {
	master(i,j,k,"xyz","++-",Bx,By,Bz,Ax,Ay,Az);
      }
    }
  }

  //+x, -y, and +z from first cell
  for(i=initi+1; i<Nx; i++)
  {
    for(j=initj-1; j>=0; j--)
    {
      for(k=initk+1; k<Nz; k++)
      {
	master(i,j,k,"xyz","+-+",Bx,By,Bz,Ax,Ay,Az);
      }
    }
  }

  //+x, +y, and +z from first cell
  for(i=initi+1; i<Nx; i++)
  {
    for(j=initj+1; j<Ny; j++)
    {
      for(k=initk+1; k<Nz; k++)
      {
	master(i,j,k,"xyz","+++",Bx,By,Bz,Ax,Ay,Az);
      }
    }
  }


  //Tests:

  //Check that B=curl(A) for every cell.
  //(only done if flag is set)

  if (print_flag == 1)
  {
    curlA_test(Bx,By,Bz,Ax,Ay,Az);
  }
  
  //Calculate and print the divergence of A

  divA(Ax,Ay,Az,div_A, 1);

  //Print the calculated vector potential

  out_A(Ax,Ay,Az,1);

  //Calculate the power spectrum of the A fields.

  powspec(Ax, Ay, Az, 1);

  //Calculate the higher derivatives.

  derivs(Ax, Ay, Az, 1);


  //Now we do the Coulomb piece.

  //START OF GAUGE TRANSFORMATION
  clock_gettime(CLOCK_REALTIME, &gstart);

  Coulomb(Ax,Ay,Az,argv);

  //END OF GAUGE TRANSFORMATION
  clock_gettime(CLOCK_REALTIME, &gend);


  //Calculate, print, and check the divergence of A_c

  divA(Ax,Ay,Az,div_A, 2);
  /*
  for(i=0; i<Nx-1; i++)
  {
    for(j=0; j<Ny-1; j++)
    {
      for(k=0; k<Nz-1; k++)
      {  
	divaindex = k+(Nz-1)*(j+(Ny-1)*i);
	
	if (fabs(div_A[divaindex]) > accerr)
	{
	  cout << "divA_c: Failure :(" << endl << endl;
	  cout << endl << "Divergence Failure!" << endl;
	  //return 0;
	}
      }
    }
  }
  */


  //Print the calculated vector potential
  //in the Coulomb gauge

  out_A(Ax,Ay,Az,2);


  //Print the calculated vector potential
  //(for use as input to IllinoisGRMHD)

  out_A_IGRMHD(Ax,Ay,Az);


  //Tests:

  //Check that B=curl(A) for every cell.
  //(only done if flag is set)

  if (print_flag == 2)
  {
    curlA_test(Bx,By,Bz,Ax,Ay,Az);
  }

  //Print the differences between various fields.

  out_diffs(Bx,By,Bz,Bx0,By0,Bz0,Ax,Ay,Az);

  //Calculate the power spectrum of the A fields (Coulomb).

  powspec(Ax, Ay, Az, 2);

  //Calculate the higher derivatives (Coulomb).

  derivs(Ax, Ay, Az, 2);


  //END OF PROGRAM
  clock_gettime(CLOCK_REALTIME, &end);


  //Print times

  prog_len = end.tv_sec - start.tv_sec;
  gauge_len = gend.tv_sec - gstart.tv_sec;
  net_len = prog_len - gauge_len;

  cout << endl;
  cout << "Times:" << endl;
  cout << "Start of program: " << start.tv_sec << endl;
  cout << "End of program: " << end.tv_sec << endl;
  cout << "Start of gauge transformation: " << gstart.tv_sec << endl;
  cout << "End of gauge transformation: " << gend.tv_sec << endl;
  cout << endl;
  cout << "Length of entire program: " << prog_len << endl;
  cout << "Length of gauge transformation: " << gauge_len << endl;
  cout << "Length of program without guage transformation: " << net_len << endl;
  cout << endl;


  //Unallocate memory

  cout << "Free Memory" << endl << endl;
  
  delete[] Bx0;
  delete[] By0;
  delete[] Bz0;
  delete[] Bx;
  delete[] By;
  delete[] Bz;
  delete[] Ax;
  delete[] Ay;
  delete[] Az;
  delete[] div_A;

  cout << "Done!" << endl << endl;

  return 0;
}


//----------------------------------------------------------------------
// The following functions set up the grid and handle input and
// output of the various fields.
//----------------------------------------------------------------------

void grid_setup (int test_flag)
{
  //This function sets up the physical and numerical grids of the
  //simulation, based on the geometry of the test being performed.
  //Currently supports the rotor test and the magnetized neutron 
  //star configurations, as described in Mosta, et al. 2014

  if (test_flag == 1)
  {
    //Rotor Test
    step = 1.0/(Nx-2*(n_ghost-1));
    stepx = 1.0/(Nx-2*(n_ghost-1));
    stepy = 1.0/(Ny-2*(n_ghost-1));
    stepz = stepx;
    dx = 1.0/(Nx-2*(n_ghost-1));
    dy = 1.0/(Ny-2*(n_ghost-1));
    dz = dx;
    dV = dx*dy*dz;
    xinit = -(n_ghost-1)*dx;
    yinit = -(n_ghost-1)*dy;
    zinit = -7*dz;
  }
  else if (test_flag == 2)
  {
    //Neutron Stars
    step = 4.0/(Nx-2*(n_ghost-1));
    stepx = 4.0/(Nx-2*(n_ghost-1));
    stepy = 4.0/(Nx-2*(n_ghost-1));
    stepz = 4.0/(Nx-2*(n_ghost-1));
    dx = 4.0/(Nx-2*(n_ghost-1));
    dy = 4.0/(Nx-2*(n_ghost-1));
    dz = 4.0/(Nx-2*(n_ghost-1));
    dV = dx*dy*dz;
    xinit = -2.0-(n_ghost-1)*dx;
    yinit = -2.0-(n_ghost-1)*dx;
    zinit = -2.0-(n_ghost-1)*dx;
  }

  return;
}

void read_B (double* Bx,double* By,double* Bz,
	     double* Bx0,double* By0,double* Bz0,
	     char *argv[])
{
  //This function reads in the magnetic fields from files,
  //one for each component of B. Currently it assumes output
  //files from IllinoisGRMHD, which have the following columns:
  //iteration, tl, rl, c, ml, i, j, k, time, x, y, z, B
  //Here tl, rl, c, and ml are related to mesh refinement, so
  //for unigrrid simulations run in serial, as this method is,
  //they will all be zero throughout the grid. It should be
  //straight-forward to adapt this function to any input file
  //format.

  FILE *inx, *iny, *inz;
  double xval, yval, zval, Bval;
  double col1, col2, col3, col4, col5, col9;

  char dummy1[100], dummy2[100], dummy3[100];
  int iter;

  int kcut; //For 2D data

  int bxindex, byindex, bzindex;
  int bx0index, by0index, bz0index;

  //Initialize the magnetic fields
  
  for(i=0; i<Nx+1; i++)
  {
    for(j=0; j<Ny; j++)
    {
      for(k=0; k<Nz; k++)
      {
	bxindex = k+(Nz)*(j+(Ny)*i);
	Bx0[bxindex] = poison;
	Bx[bxindex] = poison;
      }
    }
  }

  for(i=0; i<Nx; i++)
  {
    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz; k++)
      {
	byindex = k+(Nz)*(j+(Ny+1)*i);
	By0[byindex] = poison;
	By[byindex] = poison;
      }
    }
  }

  for(i=0; i<Nx; i++)
  {
    for(j=0; j<Ny; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	bzindex = k+(Nz+1)*(j+(Ny)*i);
	Bz0[bzindex] = poison;
	Bz[bzindex] = poison;
      }
    }
  }
  
  //Read in the magnetic field data.

  inx = fopen(argv[1], "r");
  fscanf(inx, "%99s %99s %i %99s %lf\n", 
	 dummy1, dummy2, &iter, dummy3, &t_sim);
  fscanf(inx, "%*[^\n]\n", NULL);
  fscanf(inx, "%*[^\n]\n", NULL);
  fscanf(inx, "%*[^\n]\n", NULL);
  while(!feof(inx))
  {
    fscanf(inx, "%lf %lf %lf %lf %lf %i %i %i %lf %lf %lf %lf %lf\n", 
	   &col1, &col2, &col3, &col4, &col5, &i, &j, &k, 
	   &col9, &xval, &yval, &zval, &Bval);
    if (j != 0 && k != 0)
    {
      bxindex = (k-1)+(Nz)*((j-1)+(Ny)*i);
      Bx0[bxindex] = Bval;
      Bx[bxindex] = Bval;
    }
    fgetc(inx);
  }
  fclose(inx);

  iny = fopen(argv[2], "r");
  fscanf(iny, "%*[^\n]\n", NULL);
  fscanf(iny, "%*[^\n]\n", NULL);
  fscanf(iny, "%*[^\n]\n", NULL);
  fscanf(iny, "%*[^\n]\n", NULL);
  while(!feof(iny))
  {
    fscanf(iny, "%lf %lf %lf %lf %lf %i %i %i %lf %lf %lf %lf %lf\n", 
	   &col1, &col2, &col3, &col4, &col5, &i, &j, &k, 
	   &col9, &xval, &yval, &zval, &Bval);
    if (i != 0 && k != 0)
    {
      byindex = (k-1)+(Nz)*(j+(Ny+1)*(i-1));
      By0[byindex] = Bval;
      By[byindex] = Bval;
    }
    fgetc(iny);
  }
  fclose(iny);

  inz = fopen(argv[3], "r");
  fscanf(inz, "%*[^\n]\n", NULL);
  fscanf(inz, "%*[^\n]\n", NULL);
  fscanf(inz, "%*[^\n]\n", NULL);
  fscanf(inz, "%*[^\n]\n", NULL);
  while(!feof(inz))
  {
    fscanf(inz, "%lf %lf %lf %lf %lf %i %i %i %lf %lf %lf %lf %lf\n", 
	   &col1, &col2, &col3, &col4, &col5, &i, &j, &k, 
	   &col9, &xval, &yval, &zval, &Bval);
    if (i != 0 && j != 0)
    {
      bzindex = k+(Nz+1)*((j-1)+(Ny)*(i-1));
      Bz0[bzindex] = Bval;
      Bz[bzindex] = Bval;
    }
    fgetc(inz);
  }
  kcut = k;
  fclose(inz);

  if (dim == 2)
  {
    //If this was 2D data, copy that 
    //data along the z-direction.

    for(i=0; i<Nx+1; i++)
    {
      for(j=0; j<Ny; j++)
      {
	for(k=0; k<Nz; k++)
	{
	  if(k != (kcut-1))
	  {
	    bxindex = k+(Nz)*(j+(Ny)*i);
	    bx0index = (kcut-1)+(Nz)*(j+(Ny)*i);
	    Bx0[bxindex] = Bx0[bx0index];
	    Bx[bxindex] = Bx[bx0index];
	  }
	}
      }
    }
    for(i=0; i<Nx; i++)
    {
      for(j=0; j<Ny+1; j++)
      {
	for(k=0; k<Nz; k++)
	{
	  if(k != (kcut-1))
	  {
	    byindex = k+(Nz)*(j+(Ny+1)*i);
	    by0index = (kcut-1)+(Nz)*(j+(Ny+1)*i);
	    By0[byindex] = By0[by0index];
	    By[byindex] = By[by0index];
	  }
	}
      }
    }
    for(i=0; i<Nx; i++)
    {
      for(j=0; j<Ny; j++)
      {
	for(k=0; k<Nz+1; k++)
	{
	  if(k != kcut)
	  {
	    bzindex = k+(Nz+1)*(j+(Ny)*i);
	    bz0index = kcut+(Nz+1)*(j+(Ny)*i);
	    Bz0[bzindex] = Bz0[bz0index];
	    Bz[bzindex] = Bz[bz0index];
	  }
	}
      }
    }
  }

  return;
}

void out_B (double* Bx,double* By,double* Bz,int flag)
{
  //This function outputs the B field for comparison
  //to the input B field, as a check of consistency.
  //It is called both before and after the check for
  //div(B)=0, so the flag is there to distinguish
  //between these two calls.

  int bxindex, byindex, bzindex;

  FILE *outBxi, *outBxj, *outBxk; 
  FILE *outByi, *outByj, *outByk;
  FILE *outBzi, *outBzj, *outBzk;
  FILE *outBx, *outBy, *outBz;
  ostringstream Bxi, Bxj, Bxk;
  ostringstream Byi, Byj, Byk;
  ostringstream Bzi, Bzj, Bzk;
  ostringstream Bx_out, By_out, Bz_out;  

  if (flag == 1)
  {
    Bxi << "Bx0_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    Bxj << "Bx0_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    Bxk << "Bx0_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    Byi << "By0_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    Byj << "By0_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    Byk << "By0_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    Bzi << "Bz0_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    Bzj << "Bz0_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    Bzk << "Bz0_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    Bx_out << "Bx0.dat";
    By_out << "By0.dat";
    Bz_out << "Bz0.dat";
  }
  else if (flag == 2)
  {
    Bxi << "Bx_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    Bxj << "Bx_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    Bxk << "Bx_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    Byi << "By_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    Byj << "By_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    Byk << "By_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    Bzi << "Bz_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    Bzj << "Bz_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    Bzk << "Bz_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    Bx_out << "Bx.dat";
    By_out << "By.dat";
    Bz_out << "Bz.dat";
  }

  //First output the fields for testing and graphing purposes

  outBxi = fopen(Bxi.str().c_str(), "w");
  for(j=0; j<Ny; j++)
  {
    for(k=0; k<Nz; k++)
    {
      i = cutxindex;
      bxindex = k+(Nz)*(j+(Ny)*i);
      fprintf(outBxi, "%i %i %i %lf\n", 
	      i, j, k, Bx[bxindex]);
    }	
  }
  fclose(outBxi);
  
  outBxj = fopen(Bxj.str().c_str(), "w");
  for(i=0; i<Nx+1; i++)
  {
    for(k=0; k<Nz; k++)
    {  
      j = cutyindex;
      bxindex = k+(Nz)*(j+(Ny)*i);
      fprintf(outBxj, "%i %i %i %lf\n", 
	      i, j, k, Bx[bxindex]);
    }	
  }
  fclose(outBxj);
  
  outBxk = fopen(Bxk.str().c_str(), "w");
  for(i=0; i<Nx+1; i++)
  {
    for(j=0; j<Ny; j++)
    {
      k = cutzindex;
      bxindex = k+(Nz)*(j+(Ny)*i);
      fprintf(outBxk, "%i %i %i %lf\n", 
	      i, j, k, Bx[bxindex]);
    }	
  }
  fclose(outBxk);
  
  outByi = fopen(Byi.str().c_str(), "w");
  for(j=0; j<Ny+1; j++)
  {
    for(k=0; k<Nz; k++)
    {  
      i = cutxindex;
      byindex = k+(Nz)*(j+(Ny+1)*i);
      fprintf(outByi, "%i %i %i %lf\n", 
	      i, j, k, By[byindex]);
    }	
  }
  fclose(outByi);
  
  outByj = fopen(Byj.str().c_str(), "w");
  for(i=0; i<Nx; i++)
  {
    for(k=0; k<Nz; k++)
    {  
      j = cutyindex;
      byindex = k+(Nz)*(j+(Ny+1)*i);
      fprintf(outByj, "%i %i %i %lf\n", 
	      i, j, k, By[byindex]);
    }	
  }
  fclose(outByj);
  
  outByk = fopen(Byk.str().c_str(), "w");
  for(i=0; i<Nx; i++)
  {
    for(j=0; j<Ny+1; j++)
    {
      k = cutzindex;
      byindex = k+(Nz)*(j+(Ny+1)*i);
      fprintf(outByk, "%i %i %i %lf\n", 
	      i, j, k, By[byindex]);
    }	
  }
  fclose(outByk);
  
  outBzi = fopen(Bzi.str().c_str(), "w");
  for(j=0; j<Ny; j++)
  {
    for(k=0; k<Nz+1; k++)
    {  
      i = cutxindex;
      bzindex = k+(Nz+1)*(j+(Ny)*i);
      fprintf(outBzi, "%i %i %i %lf\n", 
	      i, j, k, Bz[bzindex]);
    }	
  }
  fclose(outBzi);
  
  outBzj = fopen(Bzj.str().c_str(), "w");
  for(i=0; i<Nx; i++)
  {
    for(k=0; k<Nz+1; k++)
    {  
      j = cutyindex;
      bzindex = k+(Nz+1)*(j+(Ny)*i);
      fprintf(outBzj, "%i %i %i %lf\n", 
	      i, j, k, Bz[bzindex]);
    }	
  }
  fclose(outBzj);
  
  outBzk = fopen(Bzk.str().c_str(), "w");
  for(i=0; i<Nx; i++)
  {
    for(j=0; j<Ny; j++)
    {
      k = cutzindex;
      bzindex = k+(Nz+1)*(j+(Ny)*i);
      fprintf(outBzk, "%i %i %i %lf\n", 
	      i, j, k, Bz[bzindex]);
    }	
  }
  fclose(outBzk);


  //Then output the fields in the "correct" format

  outBx = fopen(Bx_out.str().c_str(), "w");
  for(i=0; i<Nx+1; i++)
  {
    for(j=0; j<Ny; j++)
    {
      for(k=0; k<Nz; k++)
      {  
	bxindex = k+(Nz)*(j+(Ny)*i);
	fprintf(outBx, "%.1lf %.1lf %.1lf %.16e\n", 
	        1.0*i-0.5, 1.0*j, 1.0*k, Bx[bxindex]);
      }
    }
  }
  fclose(outBx);

  outBy = fopen(By_out.str().c_str(), "w");
  for(i=0; i<Nx; i++)
  {
    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz; k++)
      {  
	byindex = k+(Nz)*(j+(Ny+1)*i);
	fprintf(outBy, "%.1lf %.1lf %.1lf %.16e\n",  
	        1.0*i, 1.0*j-0.5, 1.0*k, By[byindex]);
      }
    }
  }
  fclose(outBy);

  outBz = fopen(Bz_out.str().c_str(), "w");
  for(i=0; i<Nx; i++)
  {
    for(j=0; j<Ny; j++)
    {
      for(k=0; k<Nz+1; k++)
      {  
	bzindex = k+(Nz+1)*(j+(Ny)*i);
	fprintf(outBz, "%.1lf %.1lf %.1lf %.16e\n", 
	        1.0*i, 1.0*j, 1.0*k-0.5, Bz[bzindex]);
      }
    }
  }
  fclose(outBz);

  return;
}

void out_A (double* Ax,double* Ay,double* Az,int flag)
{
  //This function outputs the A field for testing and
  //plotting purposes. The files have less header
  //information and fewer columns than the output that
  //will be used as input into numerical relativity
  //codes (see function below for this output).
  //It is called both before and after the gauge
  //transformation, so the flag is there to 
  //distinguish between these two calls.

  int axindex, ayindex, azindex;
  double outx, outy, outz;

  FILE *outAxi, *outAxj, *outAxk;
  FILE *outAyi, *outAyj, *outAyk;
  FILE *outAzi, *outAzj, *outAzk;
  FILE *outAx, *outAy, *outAz;
  ostringstream Axi, Axj, Axk;
  ostringstream Ayi, Ayj, Ayk;
  ostringstream Azi, Azj, Azk;
  ostringstream Ax_out, Ay_out, Az_out;

  if (flag == 1)
  {
    Axi << "Ax_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    Axj << "Ax_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    Axk << "Ax_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    Ayi << "Ay_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    Ayj << "Ay_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    Ayk << "Ay_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    Azi << "Az_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    Azj << "Az_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    Azk << "Az_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    Ax_out << "Ax0.dat";
    Ay_out << "Ay0.dat";
    Az_out << "Az0.dat";
  }
  else if (flag == 2)
  {
    Axi << "Ac_x_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    Axj << "Ac_x_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    Axk << "Ac_x_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    Ayi << "Ac_y_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    Ayj << "Ac_y_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    Ayk << "Ac_y_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    Azi << "Ac_z_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    Azj << "Ac_z_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    Azk << "Ac_z_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    Ax_out << "Ax.dat";
    Ay_out << "Ay.dat";
    Az_out << "Az.dat";
  }

  //First output the fields for testing and graphing purposes

  outAxi = fopen(Axi.str().c_str(), "w");
  for(j=0; j<Ny+1; j++)
  {
    for(k=0; k<Nz+1; k++)
    {  
      i = cutxindex;
      axindex = k+(Nz+1)*(j+(Ny+1)*i);
      fprintf(outAxi, "%i %i %i %lf\n", 
	      i, j, k, Ax[axindex]);
    }	
  }
  fclose(outAxi);
  
  outAxj = fopen(Axj.str().c_str(), "w");
  for(i=0; i<Nx; i++)
  {
    for(k=0; k<Nz+1; k++)
    {  
      j = cutyindex;
      axindex = k+(Nz+1)*(j+(Ny+1)*i);
      fprintf(outAxj, "%i %i %i %lf\n", 
	      i, j, k, Ax[axindex]);
    }	
  }
  fclose(outAxj);
  
  outAxk = fopen(Axk.str().c_str(), "w");
  for(i=0; i<Nx; i++)
  {
    for(j=0; j<Ny+1; j++)
    {
      k = cutzindex;
      axindex = k+(Nz+1)*(j+(Ny+1)*i);
      fprintf(outAxk, "%i %i %i %lf\n", 
	      i, j, k, Ax[axindex]);
    }	
  }
  fclose(outAxk);
  
  outAyi = fopen(Ayi.str().c_str(), "w");
  for(j=0; j<Ny; j++)
  {
    for(k=0; k<Nz+1; k++)
    {  
      i = cutxindex;
      ayindex = k+(Nz+1)*(j+(Ny)*i);
      fprintf(outAyi, "%i %i %i %lf\n", 
	      i, j, k, Ay[ayindex]);
    }	
  }
  fclose(outAyi);
  
  outAyj = fopen(Ayj.str().c_str(), "w");
  for(i=0; i<Nx+1; i++)
  {
    for(k=0; k<Nz+1; k++)
    {  
      j = cutyindex;
      ayindex = k+(Nz+1)*(j+(Ny)*i);
      fprintf(outAyj, "%i %i %i %lf\n", 
	      i, j, k, Ay[ayindex]);
    }	
  }
  fclose(outAyj);
  
  outAyk = fopen(Ayk.str().c_str(), "w");
  for(i=0; i<Nx+1; i++)
  {
    for(j=0; j<Ny; j++)
    {
      k = cutzindex;
      ayindex = k+(Nz+1)*(j+(Ny)*i);

      fprintf(outAyk, "%i %i %i %lf\n", 
	      i, j, k, Ay[ayindex]);
    }	
  }
  fclose(outAyk);
  
  outAzi = fopen(Azi.str().c_str(), "w");
  for(j=0; j<Ny+1; j++)
  {
    for(k=0; k<Nz; k++)
    {  
      i = cutxindex;
      azindex = k+(Nz)*(j+(Ny+1)*i);
      fprintf(outAzi, "%i %i %i %lf\n", 
	      i, j, k, Az[azindex]);
    }	
  }
  fclose(outAzi);

  outAzj = fopen(Azj.str().c_str(), "w");
  for(i=0; i<Nx+1; i++)
  {
    for(k=0; k<Nz; k++)
    {  
      j = cutyindex;
      azindex = k+(Nz)*(j+(Ny+1)*i);
      fprintf(outAzj, "%i %i %i %lf\n", 
	      i, j, k, Az[azindex]);
    }	
  }
  fclose(outAzj);

  outAzk = fopen(Azk.str().c_str(), "w");
  for(i=0; i<Nx+1; i++)
  {
    for(j=0; j<Ny+1; j++)
    {
      k = cutzindex;
      azindex = k+(Nz)*(j+(Ny+1)*i);
      fprintf(outAzk, "%i %i %i %lf\n", 
	      i, j, k, Az[azindex]);
    }	
  }
  fclose(outAzk);


  //Then output the fields in the "correct" format

  outAx = fopen(Ax_out.str().c_str(), "w");
  for(i=0; i<Nx; i++)
  {
    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {  
	outx = 1.0*i;
	outy = 1.0*j-0.5;
	outz = 1.0*k-0.5;
	axindex = k+(Nz+1)*(j+(Ny+1)*i);
	fprintf(outAx, "%.1lf %.1lf %.1lf %.16e\n", 
		outx, outy, outz, Ax[axindex]);
      }
    }
  }
  fclose(outAx);

  outAy = fopen(Ay_out.str().c_str(), "w");
  for(i=0; i<Nx+1; i++)
  {
    for(j=0; j<Ny; j++)
    {
      for(k=0; k<Nz+1; k++)
      {  
	outx = 1.0*i-0.5;
	outy = 1.0*j;
	outz = 1.0*k-0.5;
	ayindex = k+(Nz+1)*(j+(Ny)*i);
	fprintf(outAy, "%.1lf %.1lf %.1lf %.16e\n", 
		outx, outy, outz, Ay[ayindex]);
      }
    }
  }
  fclose(outAy);

  outAz = fopen(Az_out.str().c_str(), "w");
  for(i=0; i<Nx+1; i++)
  {
    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz; k++)
      {  
	outx = 1.0*i-0.5;
	outy = 1.0*j-0.5;
	outz = 1.0*k;
	azindex = k+(Nz)*(j+(Ny+1)*i);
	fprintf(outAz, "%.1lf %.1lf %.1lf %.16e\n", 
		outx, outy, outz, Az[azindex]);
      }
    }
  }
  fclose(outAz);

  return;
}

void out_A_IGRMHD (double* Ax,double* Ay,double* Az)
{
  //This function outputs the A field in a format that
  //will be used as input into IllinoisGRMHD. There
  //is one file for each component of A, with the
  //following columns, in column-major order:
  //i, j, k, x, y, z, B
  //There is also header information; see the output
  //statements below. Here again it should be
  //straight-forward to adapt this function.

  FILE *outAxET, *outAyET, *outAzET;
  ostringstream AxET, AyET, AzET;
  AxET << "Ax_stagger.xyz.dat";
  AyET << "Ay_stagger.xyz.dat";
  AzET << "Az_stagger.xyz.dat";

  int axindex, ayindex, azindex;

  double outx, outy, outz;


  //Before it outputs the data, this function has to
  //manipulate the data so that the fields are all
  //the same size.

  double *Ax_ext = new double[(Nx+1)*(Ny+1)*(Nz+1)];
  double *Ay_ext = new double[(Nx+1)*(Ny+1)*(Nz+1)];
  double *Az_ext = new double[(Nx+1)*(Ny+1)*(Nz+1)];

  int ext_index, index1, index2, index3;

  //First populate the bigger arrays in
  //preparation for the necessary
  //extrapolations.

  for(i=0; i<Nx+1; i++)
  {  
    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	Ax_ext[ext_index] = poison;
	Ay_ext[ext_index] = poison;
	Az_ext[ext_index] = poison;
      }
    }
  }

  for(i=0; i<Nx; i++)
  {  
    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	axindex = k+(Nz+1)*(j+(Ny+1)*i);
	ext_index = k+(Nz+1)*(j+(Ny+1)*(i+1));
	Ax_ext[ext_index] = Ax[axindex];
      }
    }
  }
  
  for(i=0; i<Nx+1; i++)
  {  
    for(j=0; j<Ny; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	ayindex = k+(Nz+1)*(j+(Ny)*i);
	ext_index = k+(Nz+1)*((j+1)+(Ny+1)*i);
	Ay_ext[ext_index] = Ay[ayindex];
      }
    }
  }
  
  for(i=0; i<Nx+1; i++)
  {  
    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz; k++)
      {
	azindex = k+(Nz)*(j+(Ny+1)*i);
	ext_index = (k+1)+(Nz+1)*(j+(Ny+1)*i);
	Az_ext[ext_index] = Az[azindex];
      }
    }
  }
  
  //Now extrapolate the fields to make
  //them all the same size.

  for(j=0; j<Ny+1; j++)
  {
    for(k=0; k<Nz+1; k++)
    {
      i = 0;
      ext_index = k+(Nz+1)*(j+(Ny+1)*i);
      index1 = k+(Nz+1)*(j+(Ny+1)*(i+1));
      index2 = k+(Nz+1)*(j+(Ny+1)*(i+2));
      index3 = k+(Nz+1)*(j+(Ny+1)*(i+3));
      Ax_ext[ext_index] = 3*Ax_ext[index1] 
	                - 3*Ax_ext[index2]
	                + 1*Ax_ext[index3];
      /*
      if (j==52 && k==42)
      {
	cout << endl;
	cout << "Extrapolation of Ax (Coulomb):" << endl;
	cout << "Edge: " << Ax_ext[index1] << " One-In: "<< Ax_ext[index2] << " Two-In: " << Ax_ext[index3] << endl;
	cout << "Calculation: " << Ax_ext[ext_index] << endl;
	cout << endl;
      }
      */
    }
  }

  for(i=0; i<Nx+1; i++)
  {
    for(k=0; k<Nz+1; k++)
    {
      j = 0;
      ext_index = k+(Nz+1)*(j+(Ny+1)*i);
      index1 = k+(Nz+1)*((j+1)+(Ny+1)*i);
      index2 = k+(Nz+1)*((j+2)+(Ny+1)*i);
      index3 = k+(Nz+1)*((j+3)+(Ny+1)*i);
      Ay_ext[ext_index] = 3*Ay_ext[index1] 
	                - 3*Ay_ext[index2]
	                + 1*Ay_ext[index3];
    }
  }

  for(i=0; i<Nx+1; i++)
  {
    for(j=0; j<Ny+1; j++)
    {
      k = 0;
      ext_index = k+(Nz+1)*(j+(Ny+1)*i);
      index1 = (k+1)+(Nz+1)*(j+(Ny+1)*i);
      index2 = (k+2)+(Nz+1)*(j+(Ny+1)*i);
      index3 = (k+3)+(Nz+1)*(j+(Ny+1)*i);
      Az_ext[ext_index] = 3*Az_ext[index1] 
	                - 3*Az_ext[index2]
	                + 1*Az_ext[index3];
    }
  }

  //Perform extrapolation tests

  //extrap_tests(Ax_ext, Ay_ext, Az_ext, 2);


  //Now anticipate the boundary conditions
  //of IllinoisGRMHD by extrapolating in
  //the ghost zones

  extrap_ghost(Ax_ext,Ay_ext,Az_ext);


  //Finally, output the extrapolated fields 
  //for use in IllinoisGRMHD

  outAxET = fopen(AxET.str().c_str(), "w");
  fprintf(outAxET, "# time %lf\n", t_sim);
  fprintf(outAxET, "# Nx: %i   Ny: %i   Nz: %i\n", (Nx+1), (Ny+1), (Nz+1));
  fprintf(outAxET, "# column format: 1:ix   2:iy   3:iz   4:x   5:y   6:z   7:data\n");
  for(k=0; k<Nz+1; k++)
  {
    for(j=0; j<Ny+1; j++)
    {
      for(i=0; i<Nx+1; i++)
      {  
	outx = xinit + i*dx;
	outy = yinit + j*dy;
	outz = zinit + k*dz;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	fprintf(outAxET, "%i %i %i %.16e %.16e %.16e %.16e\n", 
		i, j, k, outx, outy, outz, Ax_ext[ext_index]);
      }
    }
  }
  fclose(outAxET);

  outAyET = fopen(AyET.str().c_str(), "w");
  fprintf(outAyET, "# time %lf\n", t_sim);
  fprintf(outAyET, "# Nx: %i   Ny: %i   Nz: %i\n", (Nx+1), (Ny+1), (Nz+1));
  fprintf(outAyET, "# column format: 1:ix   2:iy   3:iz   4:x   5:y   6:z   7:data\n");
  for(k=0; k<Nz+1; k++)
  {  
    for(j=0; j<Ny+1; j++)
    {
      for(i=0; i<Nx+1; i++)
      {
	outx = xinit + i*dx;
	outy = yinit + j*dy;
	outz = zinit + k*dz;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	fprintf(outAyET, "%i %i %i %.16e %.16e %.16e %.16e\n", 
		i, j, k, outx, outy, outz, Ay_ext[ext_index]);
      }
    }
  }
  fclose(outAyET);

  outAzET = fopen(AzET.str().c_str(), "w");
  fprintf(outAzET, "# time %lf\n", t_sim);
  fprintf(outAzET, "# Nx: %i   Ny: %i   Nz: %i\n", (Nx+1), (Ny+1), (Nz+1));
  fprintf(outAzET, "# column format: 1:ix   2:iy   3:iz   4:x   5:y   6:z   7:data\n");
  for(k=0; k<Nz+1; k++)
  {  
    for(j=0; j<Ny+1; j++)
    {
      for(i=0; i<Nx+1; i++)
      {
	outx = xinit + i*dx;
	outy = yinit + j*dy;
	outz = zinit + k*dz;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	fprintf(outAzET, "%i %i %i %.16e %.16e %.16e %.16e\n", 
		i, j, k, outx, outy, outz, Az_ext[ext_index]);
      }
    }
  }
  fclose(outAzET);

  delete[] Ax_ext;
  delete[] Ay_ext;
  delete[] Az_ext;

  return;
}

void out_diffs (double* Bx,double* By,double* Bz,
		double* Bx0,double* By0,double* Bz0,
		double* Ax,double* Ay,double* Az)
{
  //This function outputs field differences
  //at various cuts through the grid,
  //all of which should be zero:
  //B (s.t. div(B)=0) - B (input)
  //B (s.t. div(B)=0) - curl(A)

  int bxindex, byindex, bzindex;
  int axpmindex, axmpindex, axmmindex;
  int aypmindex, aympindex, aymmindex;
  int azpmindex, azmpindex, azmmindex;

  double B;
  double Axpm, Axmp, Axmm;
  double Aypm, Aymp, Aymm;
  double Azpm, Azmp, Azmm;

  double diff;

  FILE *outBB0xi, *outBB0xj, *outBB0xk; 
  FILE *outBB0yi, *outBB0yj, *outBB0yk; 
  FILE *outBB0zi, *outBB0zj, *outBB0zk; 
  FILE *outBxcurlAi, *outBxcurlAj, *outBxcurlAk; 
  FILE *outBycurlAi, *outBycurlAj, *outBycurlAk; 
  FILE *outBzcurlAi, *outBzcurlAj, *outBzcurlAk; 
  ostringstream BB0xi, BB0xj, BB0xk; 
  ostringstream BB0yi, BB0yj, BB0yk; 
  ostringstream BB0zi, BB0zj, BB0zk; 
  ostringstream BxcurlAi, BxcurlAj, BxcurlAk; 
  ostringstream BycurlAi, BycurlAj, BycurlAk; 
  ostringstream BzcurlAi, BzcurlAj, BzcurlAk; 
  BB0xi << "BB0x_i" << setfill('0') << setw(3) << cutxindex << ".dat";
  BB0xj << "BB0x_j" << setfill('0') << setw(3) << cutyindex << ".dat";
  BB0xk << "BB0x_k" << setfill('0') << setw(3) << cutzindex << ".dat";
  BB0yi << "BB0y_i" << setfill('0') << setw(3) << cutxindex << ".dat";
  BB0yj << "BB0y_j" << setfill('0') << setw(3) << cutyindex << ".dat";
  BB0yk << "BB0y_k" << setfill('0') << setw(3) << cutzindex << ".dat";
  BB0zi << "BB0z_i" << setfill('0') << setw(3) << cutxindex << ".dat";
  BB0zj << "BB0z_j" << setfill('0') << setw(3) << cutyindex << ".dat";
  BB0zk << "BB0z_k" << setfill('0') << setw(3) << cutzindex << ".dat";
  BxcurlAi << "Bx_curlA_i" << setfill('0') << setw(3) << cutxindex << ".dat";
  BxcurlAj << "Bx_curlA_j" << setfill('0') << setw(3) << cutyindex << ".dat";
  BxcurlAk << "Bx_curlA_k" << setfill('0') << setw(3) << cutzindex << ".dat";
  BycurlAi << "By_curlA_i" << setfill('0') << setw(3) << cutxindex << ".dat";
  BycurlAj << "By_curlA_j" << setfill('0') << setw(3) << cutyindex << ".dat";
  BycurlAk << "By_curlA_k" << setfill('0') << setw(3) << cutzindex << ".dat";
  BzcurlAi << "Bz_curlA_i" << setfill('0') << setw(3) << cutxindex << ".dat";
  BzcurlAj << "Bz_curlA_j" << setfill('0') << setw(3) << cutyindex << ".dat";
  BzcurlAk << "Bz_curlA_k" << setfill('0') << setw(3) << cutzindex << ".dat";

  outBB0xi = fopen(BB0xi.str().c_str(), "w");
  for (j=0; j<Ny; j++)
  {
    for (k=0; k<Nz; k++)
    {
      i = cutxindex;
      bxindex = k+(Nz)*(j+(Ny)*i);

      diff = Bx[bxindex] - Bx0[bxindex];

      fprintf(outBB0xi, "%i %i %i %e\n", 
	      i, j, k, diff);
    }
  }
  fclose(outBB0xi);

  outBB0xj = fopen(BB0xj.str().c_str(), "w");
  for (i=0; i<Nx+1; i++)
  {
    for (k=0; k<Nz; k++)
    {
      j = cutyindex;
      bxindex = k+(Nz)*(j+(Ny)*i);

      diff = Bx[bxindex] - Bx0[bxindex];

      fprintf(outBB0xj, "%i %i %i %e\n", 
	      i, j, k, diff);
    }
  }
  fclose(outBB0xj);

  outBB0xk = fopen(BB0xk.str().c_str(), "w");
  for (i=0; i<Nx+1; i++)
  {
    for (j=0; j<Ny; j++)
    {
      k = cutzindex;
      bxindex = k+(Nz)*(j+(Ny)*i);

      diff = Bx[bxindex] - Bx0[bxindex];

      fprintf(outBB0xk, "%i %i %i %e\n", 
	      i, j, k, diff);
    }
  }
  fclose(outBB0xk);

  outBB0yi = fopen(BB0yi.str().c_str(), "w");
  for (j=0; j<Ny+1; j++)
  {
    for (k=0; k<Nz; k++)
    {
      i = cutxindex;
      byindex = k+(Nz)*(j+(Ny+1)*i);

      diff = By[byindex] - By0[byindex];

      fprintf(outBB0yi, "%i %i %i %e\n", 
	      i, j, k, diff);
    }
  }
  fclose(outBB0yi);

  outBB0yj = fopen(BB0yj.str().c_str(), "w");
  for (i=0; i<Nx; i++)
  {
    for (k=0; k<Nz; k++)
    {
      j = cutyindex;
      byindex = k+(Nz)*(j+(Ny+1)*i);

      diff = By[byindex] - By0[byindex];

      fprintf(outBB0yj, "%i %i %i %e\n", 
	      i, j, k, diff);
    }
  }
  fclose(outBB0yj);

  outBB0yk = fopen(BB0yk.str().c_str(), "w");
  for (i=0; i<Nx; i++)
  {
    for (j=0; j<Ny+1; j++)
    {
      k = cutzindex;
      byindex = k+(Nz)*(j+(Ny+1)*i);

      diff = By[byindex] - By0[byindex];

      fprintf(outBB0yk, "%i %i %i %e\n", 
	      i, j, k, diff);
    }
  }
  fclose(outBB0yk);

  outBB0zi = fopen(BB0zi.str().c_str(), "w");
  for (j=0; j<Ny; j++)
  {
    for (k=0; k<Nz+1; k++)
    {
      i = cutxindex;
      bzindex = k+(Nz+1)*(j+(Ny)*i);

      diff = Bz[bzindex] - Bz0[bzindex];

      fprintf(outBB0zi, "%i %i %i %e\n", 
	      i, j, k, diff);
    }
  }
  fclose(outBB0zi);

  outBB0zj = fopen(BB0zj.str().c_str(), "w");
  for (i=0; i<Nx; i++)
  {
    for (k=0; k<Nz+1; k++)
    {
      j = cutyindex;
      bzindex = k+(Nz+1)*(j+(Ny)*i);

      diff = Bz[bzindex] - Bz0[bzindex];

      fprintf(outBB0zj, "%i %i %i %e\n", 
	      i, j, k, diff);
    }
  }
  fclose(outBB0zj);

  outBB0zk = fopen(BB0zk.str().c_str(), "w");
  for (i=0; i<Nx; i++)
  {
    for (j=0; j<Ny; j++)
    {
      k = cutzindex;
      bzindex = k+(Nz+1)*(j+(Ny)*i);

      diff = Bz[bzindex] - Bz0[bzindex];

      fprintf(outBB0zk, "%i %i %i %e\n", 
	      i, j, k, diff);
    }
  }
  fclose(outBB0zk);


  outBxcurlAi = fopen(BxcurlAi.str().c_str(), "w");
  for (j=0; j<Ny; j++)
  {
    for (k=0; k<Nz; k++)
    {
      i = cutxindex;
      bxindex = k+(Nz)*(j+(Ny)*i);

      B = Bx0[bxindex];

      aymmindex = k+(Nz+1)*(j+(Ny)*i);
      aypmindex = (k+1)+(Nz+1)*(j+(Ny)*i);
      
      azmmindex = k+(Nz)*(j+(Ny+1)*i);
      azmpindex = k+(Nz)*((j+1)+(Ny+1)*i);

      Aymm = Ay[aymmindex];
      Aypm = Ay[aypmindex];

      Azmm = Az[azmmindex];
      Azmp = Az[azmpindex];

      diff = BcurlA_face(B,Azmp,Azmm,Aypm,Aymm,stepz,stepy);

      fprintf(outBxcurlAi, "%i %i %i %e\n", 
	      i, j, k, diff);
    }
  }
  fclose(outBxcurlAi);

  outBxcurlAj = fopen(BxcurlAj.str().c_str(), "w");
  for (i=0; i<Nx+1; i++)
  {
    for (k=0; k<Nz; k++)
    {
      j = cutyindex;
      bxindex = k+(Nz)*(j+(Ny)*i);

      B = Bx0[bxindex];

      aymmindex = k+(Nz+1)*(j+(Ny)*i);
      aypmindex = (k+1)+(Nz+1)*(j+(Ny)*i);
      
      azmmindex = k+(Nz)*(j+(Ny+1)*i);
      azmpindex = k+(Nz)*((j+1)+(Ny+1)*i);

      Aymm = Ay[aymmindex];
      Aypm = Ay[aypmindex];

      Azmm = Az[azmmindex];
      Azmp = Az[azmpindex];

      diff = BcurlA_face(B,Azmp,Azmm,Aypm,Aymm,stepz,stepy);

      fprintf(outBxcurlAj, "%i %i %i %e\n", 
	      i, j, k, diff);
    }
  }
  fclose(outBxcurlAj);

  outBxcurlAk = fopen(BxcurlAk.str().c_str(), "w");
  for (i=0; i<Nx+1; i++)
  {
    for (j=0; j<Ny; j++)
    {
      k = cutzindex;
      bxindex = k+(Nz)*(j+(Ny)*i);

      B = Bx0[bxindex];

      aymmindex = k+(Nz+1)*(j+(Ny)*i);
      aypmindex = (k+1)+(Nz+1)*(j+(Ny)*i);
      
      azmmindex = k+(Nz)*(j+(Ny+1)*i);
      azmpindex = k+(Nz)*((j+1)+(Ny+1)*i);

      Aymm = Ay[aymmindex];
      Aypm = Ay[aypmindex];

      Azmm = Az[azmmindex];
      Azmp = Az[azmpindex];

      diff = BcurlA_face(B,Azmp,Azmm,Aypm,Aymm,stepz,stepy);

      fprintf(outBxcurlAk, "%i %i %i %e\n", 
	      i, j, k, diff);
    }
  }
  fclose(outBxcurlAk);

  outBycurlAi = fopen(BycurlAi.str().c_str(), "w");
  for (j=0; j<Ny+1; j++)
  {
    for (k=0; k<Nz; k++)
    {
      i = cutxindex;
      byindex = k+(Nz)*(j+(Ny+1)*i);

      B = By0[byindex];

      axmmindex = k+(Nz+1)*(j+(Ny+1)*i);
      axmpindex = k+(Nz+1)*((j+1)+(Ny+1)*i);

      azmmindex = k+(Nz)*(j+(Ny+1)*i);
      azpmindex = (k+1)+(Nz)*(j+(Ny+1)*i);
      
      Axmm = Ax[axmmindex];
      Axmp = Ax[axmpindex];

      Azmm = Az[azmmindex];
      Azpm = Az[azpmindex];

      diff = BcurlA_face(B,Axmp,Axmm,Azpm,Azmm,stepx,stepz);

      fprintf(outBycurlAi, "%i %i %i %e\n", 
	      i, j, k, diff);
    }
  }
  fclose(outBycurlAi);

  outBycurlAj = fopen(BycurlAj.str().c_str(), "w");
  for (i=0; i<Nx; i++)
  {
    for (k=0; k<Nz; k++)
    {
      j = cutyindex;
      byindex = k+(Nz)*(j+(Ny+1)*i);

      B = By0[byindex];

      axmmindex = k+(Nz+1)*(j+(Ny+1)*i);
      axmpindex = k+(Nz+1)*((j+1)+(Ny+1)*i);

      azmmindex = k+(Nz)*(j+(Ny+1)*i);
      azpmindex = (k+1)+(Nz)*(j+(Ny+1)*i);
      
      Axmm = Ax[axmmindex];
      Axmp = Ax[axmpindex];

      Azmm = Az[azmmindex];
      Azpm = Az[azpmindex];

      diff = BcurlA_face(B,Axmp,Axmm,Azpm,Azmm,stepx,stepz);

      fprintf(outBycurlAj, "%i %i %i %e\n", 
	      i, j, k, diff);
    }
  }
  fclose(outBycurlAj);

  outBycurlAk = fopen(BycurlAk.str().c_str(), "w");
  for (i=0; i<Nx; i++)
  {
    for (j=0; j<Ny+1; j++)
    {
      k = cutzindex;
      byindex = k+(Nz)*(j+(Ny+1)*i);

      B = By0[byindex];

      axmmindex = k+(Nz+1)*(j+(Ny+1)*i);
      axmpindex = k+(Nz+1)*((j+1)+(Ny+1)*i);

      azmmindex = k+(Nz)*(j+(Ny+1)*i);
      azpmindex = (k+1)+(Nz)*(j+(Ny+1)*i);
      
      Axmm = Ax[axmmindex];
      Axmp = Ax[axmpindex];

      Azmm = Az[azmmindex];
      Azpm = Az[azpmindex];

      diff = BcurlA_face(B,Axmp,Axmm,Azpm,Azmm,stepx,stepz);

      fprintf(outBycurlAk, "%i %i %i %e\n", 
	      i, j, k, diff);
    }
  }
  fclose(outBycurlAk);

  outBzcurlAi = fopen(BzcurlAi.str().c_str(), "w");
  for (j=0; j<Ny; j++)
  {
    for (k=0; k<Nz+1; k++)
    {
      i = cutxindex;
      bzindex = k+(Nz+1)*(j+(Ny)*i);

      B = Bz0[bzindex];

      axmmindex = k+(Nz+1)*(j+(Ny+1)*i);
      axpmindex = (k+1)+(Nz+1)*(j+(Ny+1)*i);
      
      aymmindex = k+(Nz+1)*(j+(Ny)*i);
      aympindex = k+(Nz+1)*((j+1)+(Ny)*i);

      Axmm = Ax[axmmindex];
      Axpm = Ax[axpmindex];

      Aymm = Ay[aymmindex];
      Aymp = Ay[aympindex];

      diff = BcurlA_face(B,Aymp,Aymm,Axpm,Axmm,stepy,stepx);

      fprintf(outBzcurlAi, "%i %i %i %e\n", 
	      i, j, k, diff);
    }
  }
  fclose(outBzcurlAi);

  outBzcurlAj = fopen(BzcurlAj.str().c_str(), "w");
  for (i=0; i<Nx; i++)
  {
    for (k=0; k<Nz+1; k++)
    {
      j = cutyindex;
      bzindex = k+(Nz+1)*(j+(Ny)*i);

      B = Bz0[bzindex];

      axmmindex = k+(Nz+1)*(j+(Ny+1)*i);
      axpmindex = (k+1)+(Nz+1)*(j+(Ny+1)*i);
      
      aymmindex = k+(Nz+1)*(j+(Ny)*i);
      aympindex = k+(Nz+1)*((j+1)+(Ny)*i);

      Axmm = Ax[axmmindex];
      Axpm = Ax[axpmindex];

      Aymm = Ay[aymmindex];
      Aymp = Ay[aympindex];

      diff = BcurlA_face(B,Aymp,Aymm,Axpm,Axmm,stepy,stepx);

      fprintf(outBzcurlAj, "%i %i %i %e\n", 
	      i, j, k, diff);
    }
  }
  fclose(outBzcurlAj);

  outBzcurlAk = fopen(BzcurlAk.str().c_str(), "w");
  for (i=0; i<Nx; i++)
  {
    for (j=0; j<Ny; j++)
    {
      k = cutzindex;
      bzindex = k+(Nz+1)*(j+(Ny)*i);

      B = Bz0[bzindex];

      axmmindex = k+(Nz+1)*(j+(Ny+1)*i);
      axpmindex = (k+1)+(Nz+1)*(j+(Ny+1)*i);
      
      aymmindex = k+(Nz+1)*(j+(Ny)*i);
      aympindex = k+(Nz+1)*((j+1)+(Ny)*i);

      Axmm = Ax[axmmindex];
      Axpm = Ax[axpmindex];

      Aymm = Ay[aymmindex];
      Aymp = Ay[aympindex];

      diff = BcurlA_face(B,Aymp,Aymm,Axpm,Axmm,stepy,stepx);

      fprintf(outBzcurlAk, "%i %i %i %e\n", 
	      i, j, k, diff);
    }
  }
  fclose(outBzcurlAk);

  return;
}


//----------------------------------------------------------------------
// The following functions perform tests 
// on the fields generated by the program.
//----------------------------------------------------------------------

bool divB (double* Bx,double* By,double* Bz,int flag)
{
  //This function checks and outputs the divergence of B across
  //the whole grid. It is called both before and after divergence 
  //cleaning, so the flag is there to distinguish between these 
  //two calls.

  bool zero = true;
  int bxpindex, bxmindex, bypindex, bymindex, bzpindex, bzmindex;
  double divB;
  FILE *out_divBx, *out_divBy, *out_divBz, *out_divB;
  ostringstream divBx, divBy, divBz, divB_tot;

  if(flag == 1)
  {
    divBx << "divB0_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    divBy << "divB0_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    divBz << "divB0_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    divB_tot << "divB0.dat";
  }

  else if(flag == 2)
  {
    divBx << "divB_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    divBy << "divB_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    divBz << "divB_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    divB_tot << "divB.dat";
  }

  out_divBx = fopen(divBx.str().c_str(), "w");
  out_divBy = fopen(divBy.str().c_str(), "w");
  out_divBz = fopen(divBz.str().c_str(), "w");
  out_divB = fopen(divB_tot.str().c_str(), "w");

  for(i=0; i<Nx-1; i++)
  {
    for(j=0; j<Ny-1; j++)
    {
      for(k=0; k<Nz-1; k++)
      {
	bxmindex = k+(Nz)*(j+(Ny)*i);
	bxpindex = k+(Nz)*(j+(Ny)*(i+1));

	bymindex = k+(Nz)*(j+(Ny+1)*i);
	bypindex = k+(Nz)*((j+1)+(Ny+1)*i);

	bzmindex = k+(Nz+1)*(j+(Ny)*i);
	bzpindex = (k+1)+(Nz+1)*(j+(Ny)*i);

	divB = (Bx[bxpindex] - Bx[bxmindex])/stepx
	     + (By[bypindex] - By[bymindex])/stepy
	     + (Bz[bzpindex] - Bz[bzmindex])/stepz;

	fprintf(out_divB, "%i %i %i %e\n", 
		i, j, k, divB);

	if(i == cutxindex)
	{
	  fprintf(out_divBx, "%i %i %i %e\n", 
		  i, j, k, divB);
	}
	if(j == cutyindex)
	{
	  fprintf(out_divBy, "%i %i %i %e\n", 
		  i, j, k, divB);
	}
	if(k == cutzindex)
	{
	  fprintf(out_divBz, "%i %i %i %e\n", 
		  i, j, k, divB);
	}

	if(fabs(divB) > accerr)
	  zero = false;
      }
    }
  }

  fclose(out_divBx);
  fclose(out_divBy);
  fclose(out_divBz);
  fclose(out_divB);

  return zero;
}

double divB_cell (double Bxm,double Bxp,double Bym,double Byp,double Bzm,double Bzp)
{
  //This function returns the divergence of B for a single cell.

  double divBx = (Bxp - Bxm)/stepx;
  double divBy = (Byp - Bym)/stepy;
  double divBz = (Bzp - Bzm)/stepz;
  double divB = fabs(divBx + divBy + divBz);

  return divB;
}

void divA (double* Ax,double* Ay,double* Az,double* div_A,int flag)
{
  //This function checks and outputs the divergence of A across
  //the whole grid. It is called both before and after the gauge 
  //transformation, so the flag is there to distinguish between 
  //these two calls.

  cout << endl;
  if (flag == 1)
  {
    cout << "Divergence of A:" << endl;
  }  
  else if (flag == 2)
  {
    cout << "Divergence of A_c:" << endl;
  }  
  cout << endl;

  int divaindex;
  int axindex, ayindex, azindex;
  int axindex_next, ayindex_next, azindex_next;

  double Ax_here, Ay_here, Az_here, Ax_next, Ay_next, Az_next;

  FILE *out_divAi, *out_divAj, *out_divAk, *out_divA;

  ostringstream divAi, divAj, divAk, divA_tot;
  
  if (flag == 1)
  {
    divAi << "divA_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    divAj << "divA_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    divAk << "divA_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    divA_tot << "divA.dat";
  }
  else if (flag == 2)
  {
    divAi << "divAc_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    divAj << "divAc_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    divAk << "divAc_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    divA_tot << "divAc.dat";
  }

  for(i=0; i<Nx-1; i++)
  {
    for(j=0; j<Ny-1; j++)
    {
      for(k=0; k<Nz-1; k++)
      {
	divaindex = k+(Nz-1)*(j+(Ny-1)*i);

	axindex = (k+1)+(Nz+1)*((j+1)+(Ny+1)*i);
	ayindex = (k+1)+(Nz+1)*(j+(Ny)*(i+1));
	azindex = k+(Nz)*((j+1)+(Ny+1)*(i+1));

	axindex_next = (k+1)+(Nz+1)*((j+1)+(Ny+1)*(i+1));
	ayindex_next = (k+1)+(Nz+1)*((j+1)+(Ny)*(i+1));
	azindex_next = (k+1)+(Nz)*((j+1)+(Ny+1)*(i+1));

	Ax_here = Ax[axindex];
	Ay_here = Ay[ayindex];
	Az_here = Az[azindex];

	Ax_next = Ax[axindex_next];
	Ay_next = Ay[ayindex_next];
	Az_next = Az[azindex_next];

	div_A[divaindex] = (Ax_next-Ax_here)/stepx
	                 + (Ay_next-Ay_here)/stepy
	                 + (Az_next-Az_here)/stepz;
      }
    }
  }  

  out_divAi = fopen(divAi.str().c_str(), "w");
  for(j=0; j<Ny-1; j++)
  {
    for(k=0; k<Nz-1; k++)
    {  
      i = cutxindex;
      divaindex = k+(Nz-1)*(j+(Ny-1)*i);

      fprintf(out_divAi, "%i %i %i %e\n", 
	      i, j, k, div_A[divaindex]);
    }	
  }
  fclose(out_divAi);

  out_divAj = fopen(divAj.str().c_str(), "w");
  for(i=0; i<Nx-1; i++)
  {
    for(k=0; k<Nz-1; k++)
    {  
      j = cutyindex;
      divaindex = k+(Nz-1)*(j+(Ny-1)*i);

      fprintf(out_divAj, "%i %i %i %e\n", 
	      i, j, k, div_A[divaindex]);
    }	
  }
  fclose(out_divAj);

  out_divAk = fopen(divAk.str().c_str(), "w");
  for(i=0; i<Nx-1; i++)
  {
    for(j=0; j<Ny-1; j++)
    {
      k = cutzindex;
      divaindex = k+(Nz-1)*(j+(Ny-1)*i);

      fprintf(out_divAk, "%i %i %i %e\n", 
	      i, j, k, div_A[divaindex]);
    }	
  }
  fclose(out_divAk);

  out_divA = fopen(divA_tot.str().c_str(), "w");
  for(i=0; i<Nx-1; i++)
  {
    for(j=0; j<Ny-1; j++)
    {
      for(k=0; k<Nz-1; k++)
      {  
	divaindex = k+(Nz-1)*(j+(Ny-1)*i);

	fprintf(out_divA, "%i %i %i %e\n", 
		i, j, k, div_A[divaindex]);
      }	
    }
  }
  fclose(out_divA);

  return;
}

void curl (int Nx,int Ny,int Nz,double* Ax,double* Ay,double* Az,double* Bx,double* By,double* Bz)
{
  //This function calculates the curl for a single staggered cell.
  //Using the 12 values of A for a cell it calculates the six
  //values of B in that cell. Useful for testing.

  double Axpp, Axpm, Axmp, Axmm;
  double Aypp, Aypm, Aymp, Aymm;
  double Azpp, Azpm, Azmp, Azmm;

  int bxpindex, bxmindex, bypindex, bymindex, bzpindex, bzmindex;
  int axppindex, axpmindex, axmpindex, axmmindex;
  int ayppindex, aypmindex, aympindex, aymmindex;
  int azppindex, azpmindex, azmpindex, azmmindex;

  for(i=0; i<Nx; i++)
  {
    for(j=0; j<Ny; j++)
    {
      for(k=0; k<Nz; k++)
      {
	bxmindex = k+(Nz)*(j+(Ny)*i);
	bxpindex = k+(Nz)*(j+(Ny)*(i+1));

	bymindex = k+(Nz)*(j+(Ny+1)*i);
	bypindex = k+(Nz)*((j+1)+(Ny+1)*i);

	bzmindex = k+(Nz+1)*(j+(Ny)*i);
	bzpindex = (k+1)+(Nz+1)*(j+(Ny)*i);


	axmmindex = k+(Nz+1)*(j+(Ny+1)*i);
	axpmindex = k+(Nz+1)*((j+1)+(Ny+1)*i);
	axmpindex = (k+1)+(Nz+1)*(j+(Ny+1)*i);
	axppindex = (k+1)+(Nz+1)*((j+1)+(Ny+1)*i);

	aymmindex = k+(Nz+1)*(j+(Ny)*i);
	aypmindex = (k+1)+(Nz+1)*(j+(Ny)*i);
	aympindex = k+(Nz+1)*(j+(Ny)*(i+1));
	ayppindex = (k+1)+(Nz+1)*(j+(Ny)*(i+1));

	azmmindex = k+(Nz)*(j+(Ny+1)*i);
	azpmindex = k+(Nz)*(j+(Ny+1)*(i+1));
	azmpindex = k+(Nz)*((j+1)+(Ny+1)*i);
	azppindex = k+(Nz)*((j+1)+(Ny+1)*(i+1));


	Axmm = Ax[axmmindex];
	Axpm = Ax[axpmindex];
	Axmp = Ax[axmpindex];
	Axpp = Ax[axppindex];

	Aymm = Ay[aymmindex];
	Aypm = Ay[aypmindex];
	Aymp = Ay[aympindex];
	Aypp = Ay[ayppindex];

	Azmm = Az[azmmindex];
	Azpm = Az[azpmindex];
	Azmp = Az[azmpindex];
	Azpp = Az[azppindex];


	Bx[bxpindex] = (Azpp - Azpm)/stepy - (Aypp - Aymp)/stepz;
	Bx[bxmindex] = (Azmp - Azmm)/stepy - (Aypm - Aymm)/stepz;

	By[bypindex] = (Axpp - Axpm)/stepz - (Azpp - Azmp)/stepx;
	By[bymindex] = (Axmp - Axmm)/stepz - (Azpm - Azmm)/stepx;

	Bz[bzpindex] = (Aypp - Aypm)/stepx - (Axpp - Axmp)/stepy;
	Bz[bzmindex] = (Aymp - Aymm)/stepx - (Axpm - Axmm)/stepy;
      }
    }
  }

  return;
}

void laplacian (int Nx,int Ny,int Nz,double* array)
{
  //This function calculates and outputs the Laplacian
  //of a vector field. Here this is used to check the
  //properties of the convolution kernel.

  double *lap = new double[(Nx-2)*(Ny-2)*(Nz-2)];

  int lapindex, arrindex;
  int arrxindex_prev, arryindex_prev, arrzindex_prev;
  int arrxindex_next, arryindex_next, arrzindex_next;

  double Arr_here;
  double Arrx_prev, Arry_prev, Arrz_prev;
  double Arrx_next, Arry_next, Arrz_next;

  FILE *out_arr, *out_arri, *out_arrj, *out_arrk;

  ostringstream arr, arri, arrj, arrk;
  
  arr << "lap_arr.dat";
  arri << "lap_arr_i" << setfill('0') << setw(3) << cutxindex << ".dat";
  arrj << "lap_arr_j" << setfill('0') << setw(3) << cutyindex << ".dat";
  arrk << "lap_arr_k" << setfill('0') << setw(3) << cutzindex << ".dat";

  cout << endl;

  for(i=1; i<Nx-1; i++)
  {
    for(j=1; j<Ny-1; j++)
    {
      for(k=1; k<Nz-1; k++)
      {
	lapindex = k+(Nz-2)*(j+(Ny-2)*i);

	arrindex = k+(Nz)*(j+(Ny)*i);

	arrxindex_prev = k+(Nz)*(j+(Ny)*(i-1));
	arryindex_prev = k+(Nz)*((j-1)+(Ny)*i);
	arrzindex_prev = (k-1)+(Nz)*(j+(Ny)*i);

	arrxindex_next = k+(Nz)*(j+(Ny)*(i+1));
	arryindex_next = k+(Nz)*((j+1)+(Ny)*i);
	arrzindex_next = (k+1)+(Nz)*(j+(Ny)*i);

	Arr_here = array[arrindex];

	Arrx_prev = array[arrxindex_prev];
	Arry_prev = array[arryindex_prev];
	Arrz_prev = array[arrzindex_prev];

	Arrx_next = array[arrxindex_next];
	Arry_next = array[arryindex_next];
	Arrz_next = array[arrzindex_next];

	lap[lapindex] = (Arrx_prev + Arrx_next +
			 Arry_prev + Arry_next +
			 Arrz_prev + Arrz_next -
			 6*Arr_here)/step/step;

	if(fabs(lap[lapindex]) > accerr)
	{
	  cout << "Laplacian of Kernel Failure at: "
	       << i << " " << j << " " << k << endl;
	}
      }
    }
  }  

  cout << endl;

  out_arr = fopen(arr.str().c_str(), "w");
  for(i=0; i<Nx-2; i++)
  {
    for(j=0; j<Ny-2; j++)
    {
      for(k=0; k<Nz-2; k++)
      {  
	lapindex = k+(Nz-2)*(j+(Ny-2)*i);
	fprintf(out_arr, "%i %i %i %e\n", 
		i, j, k, lap[lapindex]);
      }	
    }
  }
  fclose(out_arr);

  out_arri = fopen(arri.str().c_str(), "w");
  for(j=0; j<Ny-2; j++)
  {
    for(k=0; k<Nz-2; k++)
    {  
      i = cutxindex;
      lapindex = k+(Nz-2)*(j+(Ny-2)*i);
      fprintf(out_arri, "%i %i %i %e\n", 
	      i, j, k, lap[lapindex]);
    }	
  }
  fclose(out_arri);

  out_arrj = fopen(arrj.str().c_str(), "w");
  for(i=0; i<Nx-2; i++)
  {
    for(k=0; k<Nz-2; k++)
    {  
      j = cutyindex;
      lapindex = k+(Nz-2)*(j+(Ny-2)*i);
      fprintf(out_arrj, "%i %i %i %e\n", 
	      i, j, k, lap[lapindex]);
    }	
  }
  fclose(out_arrj);

  out_arrk = fopen(arrk.str().c_str(), "w");
  for(i=0; i<Nx-2; i++)
  {
    for(j=0; j<Ny-2; j++)
    {
      k = cutzindex;
      lapindex = k+(Nz-2)*(j+(Ny-2)*i);
      fprintf(out_arrk, "%i %i %i %e\n", 
	      i, j, k, lap[lapindex]);
    }	
  }
  fclose(out_arrk);

  return;
}

double BcurlA_face (double B,double A1,double A2,double A3,double A4,
		    double step12,double step34)
{
  //This function returns the difference between a true value
  //of B and the value of B calculated via curl(A).

  double curlA, diff;

  curlA = (A1 - A2)/step34 - (A3 - A4)/step12;

  diff = B - curlA;

  return diff;
}

double curlA_cell (double Axmm,double Axpm,double Axmp,double Axpp,
		   double Aymm,double Aypm,double Aymp,double Aypp,
		   double Azmm,double Azpm,double Azmp,double Azpp,
		   double Bxm,double Bxp,double Bym,double Byp,double Bzm,double Bzp)
{
  //This function calculates curl(A) for a cell, calculates
  //the differences between B and curl(A), prints all this
  //information, and returns the largest difference.

  double curlAxp, curlAyp, curlAzp, curlAxm, curlAym, curlAzm;
  double diffxp, diffyp, diffzp, diffxm, diffym, diffzm;
  double diff = 0.0;

  curlAxp = (Azpp - Azpm)/stepy - (Aypp - Aymp)/stepz;
  curlAxm = (Azmp - Azmm)/stepy - (Aypm - Aymm)/stepz;

  curlAyp = (Axpp - Axpm)/stepz - (Azpp - Azmp)/stepx;
  curlAym = (Axmp - Axmm)/stepz - (Azpm - Azmm)/stepx;

  curlAzp = (Aypp - Aypm)/stepx - (Axpp - Axmp)/stepy;
  curlAzm = (Aymp - Aymm)/stepx - (Axpm - Axmm)/stepy;

  diffxp = fabs(Bxp - curlAxp);
  diffxm = fabs(Bxm - curlAxm);

  diffyp = fabs(Byp - curlAyp);
  diffym = fabs(Bym - curlAym);

  diffzp = fabs(Bzp - curlAzp);
  diffzm = fabs(Bzm - curlAzm);
  
  cout << i << " " << j << " " << k << endl;
  
  cout << setiosflags(ios::fixed) << setprecision(15);
  cout << setw(20) << Axmm 
       << setw(20) << Axpm 
       << setw(20) << Axmp 
       << setw(20) << Axpp << endl;
  cout << setw(20) << Aymm 
       << setw(20) << Aypm 
       << setw(20) << Aymp 
       << setw(20) << Aypp << endl;
  cout << setw(20) << Azmm 
       << setw(20) << Azpm 
       << setw(20) << Azmp 
       << setw(20) << Azpp << endl;
  cout << endl;

  cout << setw(20) << Bxm << setw(20) << Bxp << endl;
  cout << setw(20) << Bym << setw(20) << Byp << endl;
  cout << setw(20) << Bzm << setw(20) << Bzp << endl << endl;

  cout << setw(20) << curlAxm << setw(20) << curlAxp << endl;
  cout << setw(20) << curlAym << setw(20) << curlAyp << endl;
  cout << setw(20) << curlAzm << setw(20) << curlAzp << endl << endl;

  cout << setw(20) << diffxm << setw(20) << diffxp << endl;
  cout << setw(20) << diffym << setw(20) << diffyp << endl;
  cout << setw(20) << diffzm << setw(20) << diffzp << endl << endl;

  if (diffxp > diff)
  {
    diff = diffxp;
  }  
  if (diffxm > diff)
  {
    diff = diffxm;
  }
  if (diffyp > diff)
  {
    diff = diffyp;
  }
  if (diffym > diff)
  {
    diff = diffym;
  }
  if (diffzp > diff)
  {
    diff = diffzp;
  }
  if (diffzm > diff)
  {
    diff = diffzm;
  }
    
  return diff;
}

void curlA_test (double* Bx,double* By,double* Bz,
		 double* Ax,double* Ay,double* Az)
{
  //This function performs the overall test of whether or
  //not B=curl(A), going through the entire grid. 

  int bxpindex, bxmindex, bypindex, bymindex, bzpindex, bzmindex;
  int axppindex, axpmindex, axmpindex, axmmindex;
  int ayppindex, aypmindex, aympindex, aymmindex;
  int azppindex, azpmindex, azmpindex, azmmindex;

  double Bxp, Bxm, Byp, Bym, Bzp, Bzm;

  double Axpp, Axpm, Axmp, Axmm;
  double Aypp, Aypm, Aymp, Aymm;
  double Azpp, Azpm, Azmp, Azmm;

  double cell_diff, cell_divB, diff;
  double max_diff = 0.0;

  cout << "-----------------------------------------------" << endl;

  if (print_flag == 1)
  {
    cout << endl << "Before Coulomb Gauge" << endl << endl;
  }  
  else if (print_flag == 2)
  {
    cout << endl << "After Coulomb Gauge" << endl << endl;
  }  

  for(i=0; i<Nx; i++)
  {
    for(j=0; j<Ny; j++)
    {
      for(k=0; k<Nz; k++)
      {
	bxmindex = k+(Nz)*(j+(Ny)*i);
	bxpindex = k+(Nz)*(j+(Ny)*(i+1));

	bymindex = k+(Nz)*(j+(Ny+1)*i);
	bypindex = k+(Nz)*((j+1)+(Ny+1)*i);

	bzmindex = k+(Nz+1)*(j+(Ny)*i);
	bzpindex = (k+1)+(Nz+1)*(j+(Ny)*i);

	Bxm = Bx[bxmindex];
	Bxp = Bx[bxpindex];

	Bym = By[bymindex];
	Byp = By[bypindex];

	Bzm = Bz[bzmindex];
	Bzp = Bz[bzpindex];


	axmmindex = k+(Nz+1)*(j+(Ny+1)*i);
	axpmindex = k+(Nz+1)*((j+1)+(Ny+1)*i);
	axmpindex = (k+1)+(Nz+1)*(j+(Ny+1)*i);
	axppindex = (k+1)+(Nz+1)*((j+1)+(Ny+1)*i);

	aymmindex = k+(Nz+1)*(j+(Ny)*i);
	aypmindex = (k+1)+(Nz+1)*(j+(Ny)*i);
	aympindex = k+(Nz+1)*(j+(Ny)*(i+1));
	ayppindex = (k+1)+(Nz+1)*(j+(Ny)*(i+1));

	azmmindex = k+(Nz)*(j+(Ny+1)*i);
	azpmindex = k+(Nz)*(j+(Ny+1)*(i+1));
	azmpindex = k+(Nz)*((j+1)+(Ny+1)*i);
	azppindex = k+(Nz)*((j+1)+(Ny+1)*(i+1));


	Axmm = Ax[axmmindex];
	Axpm = Ax[axpmindex];
	Axmp = Ax[axmpindex];
	Axpp = Ax[axppindex];

	Aymm = Ay[aymmindex];
	Aypm = Ay[aypmindex];
	Aymp = Ay[aympindex];
	Aypp = Ay[ayppindex];

	Azmm = Az[azmmindex];
	Azpm = Az[azpmindex];
	Azmp = Az[azmpindex];
	Azpp = Az[azppindex];
	  
	cell_diff = curlA_cell(Axmm,Axpm,Axmp,Axpp,
			       Aymm,Aypm,Aymp,Aypp,
			       Azmm,Azpm,Azmp,Azpp,
			       Bxm,Bxp,Bym,Byp,Bzm,Bzp);

	cell_divB = divB_cell(Bxm,Bxp,Bym,Byp,Bzm,Bzp);
	diff = fabs(cell_divB-cell_diff);

	cout << scientific << cell_divB << fixed << endl;
	cout << scientific << diff << fixed << endl;
	cout << endl;

	if (cell_diff < accerr)
	{
	  cout << i << " " << j << " " << k << "   "
	       << "curlA: Success!" << endl
	       << endl << endl;
	}
	else
	{
	  cout << i << " " << j << " " << k << "   "
	       << "curlA: Failure :(" << endl
	       << endl << endl;
	}
	  
	if (diff > max_diff)
	{
	  max_diff = diff;
	}
      }
    }
  }

  cout << endl;
  cout << "Maximum difference between div(B) and (B-curl(A)): ";
  cout << scientific << max_diff << fixed << endl;
  cout << endl;

  return;
}

void powspec (double* Ax,double* Ay,double* Az,int flag)
{
  //This function calculates and outputs the power 
  //spectrum of the A field to test for spikes in the 
  //field. It is called both before and after the gauge
  //transformation, so the flag is there to distinguish
  //between these two calls.

  cout << endl;
  if (flag == 1)
  {
    cout << "Calculating the power spectrum of A" << endl;
  }
  else if (flag == 2)
  {
    cout << "Calculating the power spectrum of A_c" << endl;
  }
  cout << endl;

  double *Ax_ps = new double[Nx_fft*Ny_fft*Nz_fft];
  double *Ay_ps = new double[Nx_fft*Ny_fft*Nz_fft];
  double *Az_ps = new double[Nx_fft*Ny_fft*Nz_fft];

  fftw_complex *Ax_ps_fft = new fftw_complex[Nx_fft*Ny_fft*(Nz_fft/2+1)];
  fftw_complex *Ay_ps_fft = new fftw_complex[Nx_fft*Ny_fft*(Nz_fft/2+1)];
  fftw_complex *Az_ps_fft = new fftw_complex[Nx_fft*Ny_fft*(Nz_fft/2+1)];

  int axindex, ayindex, azindex, apsindex, aps_fftindex;

  double Ax_xN2, Ax_xN1, Ax_xNfft, Ax_xNfft1;  // N2 indicates the element with index N-2
  double Ax_yN1, Ax_yN, Ax_yNfft, Ax_yNfft1;   // N1 indicates the element with index N-1
  double Ax_zN1, Ax_zN, Ax_zNfft, Ax_zNfft1;   // N indicates the element with index N
  double Ay_xN1, Ay_xN, Ay_xNfft, Ay_xNfft1;   // Nfft indicates the element with index N_fft
  double Ay_yN2, Ay_yN1, Ay_yNfft, Ay_yNfft1;  // Nfft1 indicates the element with index N_fft+1
  double Ay_zN1, Ay_zN, Ay_zNfft, Ay_zNfft1;
  double Az_xN1, Az_xN, Az_xNfft, Az_xNfft1;
  double Az_yN1, Az_yN, Az_yNfft, Az_yNfft1;
  double Az_zN2, Az_zN1, Az_zNfft, Az_zNfft1;

  double wave_no, Axps_mag, Ayps_mag, Azps_mag;

  fftw_plan forward_Ax_ps, forward_Ay_ps, forward_Az_ps;
  forward_Ax_ps = fftw_plan_dft_r2c_3d(Nx_fft,Ny_fft,Nz_fft,Ax_ps,Ax_ps_fft,FFTW_ESTIMATE);
  forward_Ay_ps = fftw_plan_dft_r2c_3d(Nx_fft,Ny_fft,Nz_fft,Ay_ps,Ay_ps_fft,FFTW_ESTIMATE);
  forward_Az_ps = fftw_plan_dft_r2c_3d(Nx_fft,Ny_fft,Nz_fft,Az_ps,Az_ps_fft,FFTW_ESTIMATE);    

  FILE *out_Ax_ps, *out_Ay_ps, *out_Az_ps;
  FILE *outAps_xi, *outAps_xj, *outAps_xk;
  FILE *outAps_yi, *outAps_yj, *outAps_yk;
  FILE *outAps_zi, *outAps_zj, *outAps_zk;

  ostringstream Axps, Ayps, Azps;
  ostringstream Aps_xi, Aps_xj, Aps_xk;
  ostringstream Aps_yi, Aps_yj, Aps_yk;
  ostringstream Aps_zi, Aps_zj, Aps_zk;


  if (flag == 1)
  {
    Axps << "Ax_ps.dat";
    Ayps << "Ay_ps.dat";
    Azps << "Az_ps.dat";
    Aps_xi << "Aps_x_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    Aps_xj << "Aps_x_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    Aps_xk << "Aps_x_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    Aps_yi << "Aps_y_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    Aps_yj << "Aps_y_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    Aps_yk << "Aps_y_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    Aps_zi << "Aps_z_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    Aps_zj << "Aps_z_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    Aps_zk << "Aps_z_k" << setfill('0') << setw(3) << cutzindex << ".dat";
  }
  else if (flag == 2)
  {
    Axps << "Ac_x_ps.dat";
    Ayps << "Ac_y_ps.dat";
    Azps << "Ac_z_ps.dat";
    Aps_xi << "Apsc_x_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    Aps_xj << "Apsc_x_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    Aps_xk << "Apsc_x_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    Aps_yi << "Apsc_y_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    Aps_yj << "Apsc_y_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    Aps_yk << "Apsc_y_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    Aps_zi << "Apsc_z_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    Aps_zj << "Apsc_z_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    Aps_zk << "Apsc_z_k" << setfill('0') << setw(3) << cutzindex << ".dat";
  }
  
  //First, copy the A fields into the new arrays.

  for(i=0; i<Nx; i++)
  {
    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	axindex = k+(Nz+1)*(j+(Ny+1)*i);
	apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
	Ax_ps[apsindex] = Ax[axindex];
      }
    }
  }

  for(i=0; i<Nx+1; i++)
  {
    for(j=0; j<Ny; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	ayindex = k+(Nz+1)*(j+(Ny)*i);
	apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
	Ay_ps[apsindex] = Ay[ayindex];
      }
    }
  }

  for(i=0; i<Nx+1; i++)
  {
    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz; k++)
      {
	azindex = k+(Nz)*(j+(Ny+1)*i);
	apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
	Az_ps[apsindex] = Az[azindex];
      }
    }
  }

  //Now interpolate to fill the rest of
  //each array in such a way that the
  //last points in each direction are
  //the same as the first, giving the
  //data some level of periodicity.

  //Ax

  for (i=0; i<Nx; i++)
  {
    for (k=0; k<Nz+1; k++)
    {
      Ax_yN1 = Ax_ps[k+(Nz_fft)*((Ny-1)+(Ny_fft)*i)];
      Ax_yN = Ax_ps[k+(Nz_fft)*((Ny)+(Ny_fft)*i)];
      Ax_yNfft = Ax_ps[k+(Nz_fft)*(0+(Ny_fft)*i)];
      Ax_yNfft1 = Ax_ps[k+(Nz_fft)*(1+(Ny_fft)*i)];
      
      for (j=Ny+1; j<Ny_fft; j++)
      {
	apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
	Ax_ps[apsindex] = Ax_yN1*(j-(Ny))*(j-(Ny_fft))*(j-(Ny_fft+1))/((Ny-1)-(Ny))/((Ny-1)-(Ny_fft))/((Ny-1)-(Ny_fft+1))
	                + Ax_yN*(j-(Ny-1))*(j-(Ny_fft))*(j-(Ny_fft+1))/((Ny)-(Ny-1))/((Ny)-(Ny_fft))/((Ny)-(Ny_fft+1))
	                + Ax_yNfft*(j-(Ny-1))*(j-(Ny))*(j-(Ny_fft+1))/((Ny_fft)-(Ny-1))/((Ny_fft)-(Ny))/((Ny_fft)-(Ny_fft+1))
			+ Ax_yNfft1*(j-(Ny-1))*(j-(Ny))*(j-(Ny_fft))/((Ny_fft+1)-(Ny-1))/((Ny_fft+1)-(Ny))/((Ny_fft+1)-(Ny_fft));
      }
    }
  }

  for (i=0; i<Nx; i++)
  {
    for (j=0; j<Ny+1; j++)
    {
      Ax_zN1 = Ax_ps[(Nz-1)+(Nz_fft)*(j+(Ny_fft)*i)];
      Ax_zN = Ax_ps[(Nz)+(Nz_fft)*(j+(Ny_fft)*i)];
      Ax_zNfft = Ax_ps[0+(Nz_fft)*(j+(Ny_fft)*i)];
      Ax_zNfft1 = Ax_ps[1+(Nz_fft)*(j+(Ny_fft)*i)];

      for (k=Nz+1; k<Nz_fft; k++)
      {
	apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
	Ax_ps[apsindex] = Ax_zN1*(k-(Nz))*(k-(Nz_fft))*(k-(Nz_fft+1))/((Nz-1)-(Nz))/((Nz-1)-(Nz_fft))/((Nz-1)-(Nz_fft+1))
	                + Ax_zN*(k-(Nz-1))*(k-(Nz_fft))*(k-(Nz_fft+1))/((Nz)-(Nz-1))/((Nz)-(Nz_fft))/((Nz)-(Nz_fft+1))
	                + Ax_zNfft*(k-(Nz-1))*(k-(Nz))*(k-(Nz_fft+1))/((Nz_fft)-(Nz-1))/((Nz_fft)-(Nz))/((Nz_fft)-(Nz_fft+1))
	                + Ax_zNfft1*(k-(Nz-1))*(k-(Nz))*(k-(Nz_fft))/((Nz_fft+1)-(Nz-1))/((Nz_fft+1)-(Nz))/((Nz_fft+1)-(Nz_fft));
      }
    }
  }

  for (i=0; i<Nx; i++)
  {
    for (j=Ny+1; j<Ny_fft; j++)
    {
      Ax_zN1 = Ax_ps[(Nz-1)+(Nz_fft)*(j+(Ny_fft)*i)];
      Ax_zN = Ax_ps[(Nz)+(Nz_fft)*(j+(Ny_fft)*i)];
      Ax_zNfft = Ax_ps[0+(Nz_fft)*(j+(Ny_fft)*i)];
      Ax_zNfft1 = Ax_ps[1+(Nz_fft)*(j+(Ny_fft)*i)];

      for (k=Nz+1; k<Nz_fft; k++)
      {
	apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
	Ax_ps[apsindex] = Ax_zN1*(k-(Nz))*(k-(Nz_fft))*(k-(Nz_fft+1))/((Nz-1)-(Nz))/((Nz-1)-(Nz_fft))/((Nz-1)-(Nz_fft+1))
	                + Ax_zN*(k-(Nz-1))*(k-(Nz_fft))*(k-(Nz_fft+1))/((Nz)-(Nz-1))/((Nz)-(Nz_fft))/((Nz)-(Nz_fft+1))
	                + Ax_zNfft*(k-(Nz-1))*(k-(Nz))*(k-(Nz_fft+1))/((Nz_fft)-(Nz-1))/((Nz_fft)-(Nz))/((Nz_fft)-(Nz_fft+1))
	                + Ax_zNfft1*(k-(Nz-1))*(k-(Nz))*(k-(Nz_fft))/((Nz_fft+1)-(Nz-1))/((Nz_fft+1)-(Nz))/((Nz_fft+1)-(Nz_fft));
      }
    }
  }

  for(j=0; j<Ny_fft; j++)
  {
    for(k=0; k<Nz_fft; k++)
    {
      Ax_xN2 = Ax_ps[k+(Nz_fft)*(j+(Ny_fft)*(Nx-2))];
      Ax_xN1 = Ax_ps[k+(Nz_fft)*(j+(Ny_fft)*(Nx-1))];
      Ax_xNfft = Ax_ps[k+(Nz_fft)*(j+(Ny_fft)*0)];
      Ax_xNfft1 = Ax_ps[k+(Nz_fft)*(j+(Ny_fft)*1)];

      for(i=Nx; i<Nx_fft; i++)
      {
	apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
	Ax_ps[apsindex] = Ax_xN2*(i-(Nx-1))*(i-(Nx_fft))*(i-(Nx_fft+1))/((Nx-2)-(Nx-1))/((Nx-2)-(Nx_fft))/((Nx-2)-(Nx_fft+1))
	                + Ax_xN1*(i-(Nx-2))*(i-(Nx_fft))*(i-(Nx_fft+1))/((Nx-1)-(Nx-2))/((Nx-1)-(Nx_fft))/((Nx-1)-(Nx_fft+1))
	                + Ax_xNfft*(i-(Nx-2))*(i-(Nx-1))*(i-(Nx_fft+1))/((Nx_fft)-(Nx-2))/((Nx_fft)-(Nx-1))/((Nx_fft)-(Nx_fft+1))
	                + Ax_xNfft1*(i-(Nx-2))*(i-(Nx-1))*(i-(Nx_fft))/((Nx_fft+1)-(Nx-2))/((Nx_fft+1)-(Nx-1))/((Nx_fft+1)-(Nx_fft));
      }
    }
  }

  //Ay

  for (i=0; i<Nx+1; i++)
  {
    for (j=0; j<Ny; j++)
    {
      Ay_zN1 = Ay_ps[(Nz-1)+(Nz_fft)*(j+(Ny_fft)*i)];
      Ay_zN = Ay_ps[(Nz)+(Nz_fft)*(j+(Ny_fft)*i)];
      Ay_zNfft = Ay_ps[0+(Nz_fft)*(j+(Ny_fft)*i)];
      Ay_zNfft1 = Ay_ps[1+(Nz_fft)*(j+(Ny_fft)*i)];

      for (k=Nz+1; k<Nz_fft; k++)
      {
	apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
	Ay_ps[apsindex] = Ay_zN1*(k-(Nz))*(k-(Nz_fft))*(k-(Nz_fft+1))/((Nz-1)-(Nz))/((Nz-1)-(Nz_fft))/((Nz-1)-(Nz_fft+1))
	                + Ay_zN*(k-(Nz-1))*(k-(Nz_fft))*(k-(Nz_fft+1))/((Nz)-(Nz-1))/((Nz)-(Nz_fft))/((Nz)-(Nz_fft+1))
	                + Ay_zNfft*(k-(Nz-1))*(k-(Nz))*(k-(Nz_fft+1))/((Nz_fft)-(Nz-1))/((Nz_fft)-(Nz))/((Nz_fft)-(Nz_fft+1))
	                + Ay_zNfft1*(k-(Nz-1))*(k-(Nz))*(k-(Nz_fft))/((Nz_fft+1)-(Nz-1))/((Nz_fft+1)-(Nz))/((Nz_fft+1)-(Nz_fft));
      }
    }
  }

  for (j=0; j<Ny; j++)
  {
    for (k=0; k<Nz+1; k++)
    {
      Ay_xN1 = Ay_ps[k+(Nz_fft)*(j+(Ny_fft)*(Nx-1))];
      Ay_xN = Ay_ps[k+(Nz_fft)*(j+(Ny_fft)*(Nx))];
      Ay_xNfft = Ay_ps[k+(Nz_fft)*(j+(Ny_fft)*0)];
      Ay_xNfft1 = Ay_ps[k+(Nz_fft)*(j+(Ny_fft)*1)];

      for (i=Nx+1; i<Nx_fft; i++)
      {
	apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
	Ay_ps[apsindex] = Ay_xN1*(i-(Nx))*(i-(Nx_fft))*(i-(Nx_fft+1))/((Nx-1)-(Nx))/((Nx-1)-(Nx_fft))/((Nx-1)-(Nx_fft+1))
	                + Ay_xN*(i-(Nx-1))*(i-(Nx_fft))*(i-(Nx_fft+1))/((Nx)-(Nx-1))/((Nx)-(Nx_fft))/((Nx)-(Nx_fft+1))
	                + Ay_xNfft*(i-(Nx-1))*(i-(Nx))*(i-(Nx_fft+1))/((Nx_fft)-(Nx-1))/((Nx_fft)-(Nx))/((Nx_fft)-(Nx_fft+1))
                        + Ay_xNfft1*(i-(Nx-1))*(i-(Nx))*(i-(Nx_fft))/((Nx_fft+1)-(Nx-1))/((Nx_fft+1)-(Nx))/((Nx_fft+1)-(Nx_fft));
      }
    }
  }

  for (j=0; j<Ny; j++)
  {
    for (k=Nz+1; k<Nz_fft; k++)
    {
      Ay_xN1 = Ay_ps[k+(Nz_fft)*(j+(Ny_fft)*(Nx-1))];
      Ay_xN = Ay_ps[k+(Nz_fft)*(j+(Ny_fft)*(Nx))];
      Ay_xNfft = Ay_ps[k+(Nz_fft)*(j+(Ny_fft)*0)];
      Ay_xNfft1 = Ay_ps[k+(Nz_fft)*(j+(Ny_fft)*1)];

      for (i=Nx+1; i<Nx_fft; i++)
      {
	apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
	Ay_ps[apsindex] = Ay_xN1*(i-(Nx))*(i-(Nx_fft))*(i-(Nx_fft+1))/((Nx-1)-(Nx))/((Nx-1)-(Nx_fft))/((Nx-1)-(Nx_fft+1))
	                + Ay_xN*(i-(Nx-1))*(i-(Nx_fft))*(i-(Nx_fft+1))/((Nx)-(Nx-1))/((Nx)-(Nx_fft))/((Nx)-(Nx_fft+1))
	                + Ay_xNfft*(i-(Nx-1))*(i-(Nx))*(i-(Nx_fft+1))/((Nx_fft)-(Nx-1))/((Nx_fft)-(Nx))/((Nx_fft)-(Nx_fft+1))
	                + Ay_xNfft1*(i-(Nx-1))*(i-(Nx))*(i-(Nx_fft))/((Nx_fft+1)-(Nx-1))/((Nx_fft+1)-(Nx))/((Nx_fft+1)-(Nx_fft));
      }
    }
  }

  for(i=0; i<Nx_fft; i++)
  {
    for(k=0; k<Nz_fft; k++)
    {
      Ay_yN2 = Ay_ps[k+(Nz_fft)*((Ny-2)+(Ny_fft)*i)];
      Ay_yN1 = Ay_ps[k+(Nz_fft)*((Ny-1)+(Ny_fft)*i)];
      Ay_yNfft = Ay_ps[k+(Nz_fft)*(0+(Ny_fft)*i)];
      Ay_yNfft1 = Ay_ps[k+(Nz_fft)*(1+(Ny_fft)*i)];

      for(j=Ny; j<Ny_fft; j++)
      {
	apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
	Ay_ps[apsindex] = Ay_yN2*(j-(Ny-1))*(j-(Ny_fft))*(j-(Ny_fft+1))/((Ny-2)-(Ny-1))/((Ny-2)-(Ny_fft))/((Ny-2)-(Ny_fft+1))
	                + Ay_yN1*(j-(Ny-2))*(j-(Ny_fft))*(j-(Ny_fft+1))/((Ny-1)-(Ny-2))/((Ny-1)-(Ny_fft))/((Ny-1)-(Ny_fft+1))
	                + Ay_yNfft*(j-(Ny-2))*(j-(Ny-1))*(j-(Ny_fft+1))/((Ny_fft)-(Ny-2))/((Ny_fft)-(Ny-1))/((Ny_fft)-(Ny_fft+1))
	                + Ay_yNfft1*(j-(Ny-2))*(j-(Ny-1))*(j-(Ny_fft))/((Ny_fft+1)-(Ny-2))/((Ny_fft+1)-(Ny-1))/((Ny_fft+1)-(Ny_fft));
      }
    }
  }

  //Az

  for (j=0; j<Ny+1; j++)
  {
    for (k=0; k<Nz; k++)
    {
      Az_xN1 = Az_ps[k+(Nz_fft)*(j+(Ny_fft)*(Nx-1))];
      Az_xN = Az_ps[k+(Nz_fft)*(j+(Ny_fft)*(Nx))];
      Az_xNfft = Az_ps[k+(Nz_fft)*(j+(Ny_fft)*0)];
      Az_xNfft1 = Az_ps[k+(Nz_fft)*(j+(Ny_fft)*1)];

      for (i=Nx+1; i<Nx_fft; i++)
      {
	apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
	Az_ps[apsindex] = Az_xN1*(i-(Nx))*(i-(Nx_fft))*(i-(Nx_fft+1))/((Nx-1)-(Nx))/((Nx-1)-(Nx_fft))/((Nx-1)-(Nx_fft+1))
	                + Az_xN*(i-(Nx-1))*(i-(Nx_fft))*(i-(Nx_fft+1))/((Nx)-(Nx-1))/((Nx)-(Nx_fft))/((Nx)-(Nx_fft+1))
	                + Az_xNfft*(i-(Nx-1))*(i-(Nx))*(i-(Nx_fft+1))/((Nx_fft)-(Nx-1))/((Nx_fft)-(Nx))/((Nx_fft)-(Nx_fft+1))
	                + Az_xNfft1*(i-(Nx-1))*(i-(Nx))*(i-(Nx_fft))/((Nx_fft+1)-(Nx-1))/((Nx_fft+1)-(Nx))/((Nx_fft+1)-(Nx_fft));
      }
    }
  }

  for (i=0; i<Nx+1; i++)
  {
    for (k=0; k<Nz; k++)
    {
      Az_yN1 = Az_ps[k+(Nz_fft)*((Ny-1)+(Ny_fft)*i)];
      Az_yN = Az_ps[k+(Nz_fft)*((Ny)+(Ny_fft)*i)];
      Az_yNfft = Az_ps[k+(Nz_fft)*(0+(Ny_fft)*i)];
      Az_yNfft1 = Az_ps[k+(Nz_fft)*(1+(Ny_fft)*i)];

      for (j=Ny+1; j<Ny_fft; j++)
      {
	apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
	Az_ps[apsindex] = Az_yN1*(j-(Ny))*(j-(Ny_fft))*(j-(Ny_fft+1))/((Ny-1)-(Ny))/((Ny-1)-(Ny_fft))/((Ny-1)-(Ny_fft+1))
	                + Az_yN*(j-(Ny-1))*(j-(Ny_fft))*(j-(Ny_fft+1))/((Ny)-(Ny-1))/((Ny)-(Ny_fft))/((Ny)-(Ny_fft+1))
	                + Az_yNfft*(j-(Ny-1))*(j-(Ny))*(j-(Ny_fft+1))/((Ny_fft)-(Ny-1))/((Ny_fft)-(Ny))/((Ny_fft)-(Ny_fft+1))
                        + Az_yNfft1*(j-(Ny-1))*(j-(Ny))*(j-(Ny_fft))/((Ny_fft+1)-(Ny-1))/((Ny_fft+1)-(Ny))/((Ny_fft+1)-(Ny_fft));
      }
    }
  }

  for (i=Nx+1; i<Nx_fft; i++)
  {
    for (k=0; k<Nz; k++)
    {
      Az_yN1 = Az_ps[k+(Nz_fft)*((Ny-1)+(Ny_fft)*i)];
      Az_yN = Az_ps[k+(Nz_fft)*((Ny)+(Ny_fft)*i)];
      Az_yNfft = Az_ps[k+(Nz_fft)*(0+(Ny_fft)*i)];
      Az_yNfft1 = Az_ps[k+(Nz_fft)*(1+(Ny_fft)*i)];

      for (j=Ny+1; j<Ny_fft; j++)
      {
	apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
	Az_ps[apsindex] = Az_yN1*(k-(Ny))*(j-(Ny_fft))*(j-(Ny_fft+1))/((Ny-1)-(Ny))/((Ny-1)-(Ny_fft))/((Ny-1)-(Ny_fft+1))
	                + Az_yN*(k-(Ny-1))*(j-(Ny_fft))*(j-(Ny_fft+1))/((Ny)-(Ny-1))/((Ny)-(Ny_fft))/((Ny)-(Ny_fft+1))
	                + Az_yNfft*(k-(Ny-1))*(j-(Ny))*(j-(Ny_fft+1))/((Ny_fft)-(Ny-1))/((Ny_fft)-(Ny))/((Ny_fft)-(Ny_fft+1))
	                + Az_yNfft1*(k-(Ny-1))*(j-(Ny))*(j-(Ny_fft))/((Ny_fft+1)-(Ny-1))/((Ny_fft+1)-(Ny))/((Ny_fft+1)-(Ny_fft));
      }
    }
  }

  for(i=0; i<Nx_fft; i++)
  {
    for(j=0; j<Ny_fft; j++)
    {
      Az_zN2 = Az_ps[(Nz-2)+(Nz_fft)*(j+(Ny_fft)*i)];
      Az_zN1 = Az_ps[(Nz-1)+(Nz_fft)*(j+(Ny_fft)*i)];
      Az_zNfft = Az_ps[0+(Nz_fft)*(j+(Ny_fft)*i)];
      Az_zNfft1 = Az_ps[1+(Nz_fft)*(j+(Ny_fft)*i)];

      for(k=Nz; k<Nz_fft; k++)
      {
	apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
	Az_ps[apsindex] = Az_zN2*(k-(Nz-1))*(k-(Nz_fft))*(k-(Nz_fft+1))/((Nz-2)-(Nz-1))/((Nz-2)-(Nz_fft))/((Nz-2)-(Nz_fft+1))
	                + Az_zN1*(k-(Nz-2))*(k-(Nz_fft))*(k-(Nz_fft+1))/((Nz-1)-(Nz-2))/((Nz-1)-(Nz_fft))/((Nz-1)-(Nz_fft+1))
	                + Az_zNfft*(k-(Nz-2))*(k-(Nz-1))*(k-(Nz_fft+1))/((Nz_fft)-(Nz-2))/((Nz_fft)-(Nz-1))/((Nz_fft)-(Nz_fft+1))
	                + Az_zNfft1*(k-(Nz-2))*(k-(Nz-1))*(k-(Nz_fft))/((Nz_fft+1)-(Nz-2))/((Nz_fft+1)-(Nz-1))/((Nz_fft+1)-(Nz_fft));
      }
    }
  }

  //Print the larger, interpolated A fields.

  outAps_xi = fopen(Aps_xi.str().c_str(), "w");
  for(j=0; j<Ny_fft; j++)
  {
    for(k=0; k<Nz_fft; k++)
    { 
      i = cutxindex;
      apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
      fprintf(outAps_xi, "%i %i %i %lf\n", 
	      i, j, k, Ax_ps[apsindex]);
    }
  }
  fclose(outAps_xi);

  outAps_xj = fopen(Aps_xj.str().c_str(), "w");
  for(i=0; i<Nx_fft; i++)
  {
    for(k=0; k<Nz_fft; k++)
    {
      j = cutyindex;
      apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
      fprintf(outAps_xj, "%i %i %i %lf\n", 
	      i, j, k, Ax_ps[apsindex]);
    }	
  }
  fclose(outAps_xj);

  outAps_xk = fopen(Aps_xk.str().c_str(), "w");
  for(i=0; i<Nx_fft; i++)
  {
    for(j=0; j<Ny_fft; j++)
    {
      k = cutzindex;
      apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
      fprintf(outAps_xk, "%i %i %i %lf\n", 
	      i, j, k, Ax_ps[apsindex]);
    }	
  }
  fclose(outAps_xk);

  outAps_yi = fopen(Aps_yi.str().c_str(), "w");
  for(j=0; j<Ny_fft; j++)
  {
    for(k=0; k<Nz_fft; k++)
    { 
      i = cutxindex;
      apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
      fprintf(outAps_yi, "%i %i %i %lf\n", 
	      i, j, k, Ay_ps[apsindex]);
    }	
  }
  fclose(outAps_yi);

  outAps_yj = fopen(Aps_yj.str().c_str(), "w");
  for(i=0; i<Nx_fft; i++)
  {
    for(k=0; k<Nz_fft; k++)
    {  
      j = cutyindex;
      apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
      fprintf(outAps_yj, "%i %i %i %lf\n", 
	      i, j, k, Ay_ps[apsindex]);
    }	
  }
  fclose(outAps_yj);

  outAps_yk = fopen(Aps_yk.str().c_str(), "w");
  for(i=0; i<Nx_fft; i++)
  {
    for(j=0; j<Ny_fft; j++)
    {
      k = cutzindex;
      apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
      fprintf(outAps_yk, "%i %i %i %lf\n", 
	      i, j, k, Ay_ps[apsindex]);
    }	
  }
  fclose(outAps_yk);
  
  outAps_zi = fopen(Aps_zi.str().c_str(), "w");
  for(j=0; j<Ny_fft; j++)
  {
    for(k=0; k<Nz_fft; k++)
    {
      i = cutxindex;
      apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
      fprintf(outAps_zi, "%i %i %i %lf\n", 
	      i, j, k, Az_ps[apsindex]);
    }	
  }
  fclose(outAps_zi);

  outAps_zj = fopen(Aps_zj.str().c_str(), "w");
  for(i=0; i<Nx_fft; i++)
  {
    for(k=0; k<Nz_fft; k++)
    {
      j = cutyindex;
      apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
      fprintf(outAps_zj, "%i %i %i %lf\n", 
	      i, j, k, Az_ps[apsindex]);
    }	
  }
  fclose(outAps_zj);

  outAps_zk = fopen(Aps_zk.str().c_str(), "w");
  for(i=0; i<Nx_fft; i++)
  {
    for(j=0; j<Ny_fft; j++)
    {
      k = cutzindex;
      apsindex = k+(Nz_fft)*(j+(Ny_fft)*i);
      fprintf(outAps_zk, "%i %i %i %lf\n", 
	      i, j, k, Az_ps[apsindex]);
    }	
  }
  fclose(outAps_zk);

  //Perform the Fourier transforms.

  fftw_execute(forward_Ax_ps);
  fftw_execute(forward_Ay_ps);
  fftw_execute(forward_Az_ps);

  for(i=0; i<Nx_fft; i++) 
  {
    for(j=0; j<Ny_fft; j++) 
    {
      for(k=0; k<Nz_fft/2+1; k++) 
      {
	aps_fftindex = k+(Nz_fft/2+1)*(j+(Ny_fft)*i);

	Ax_ps_fft[aps_fftindex][0] *= 1.0/sqrt(Nx_fft)/sqrt(Ny_fft)/sqrt(Nz_fft);
	Ax_ps_fft[aps_fftindex][1] *= 1.0/sqrt(Nx_fft)/sqrt(Ny_fft)/sqrt(Nz_fft);

	Ay_ps_fft[aps_fftindex][0] *= 1.0/sqrt(Nx_fft)/sqrt(Ny_fft)/sqrt(Nz_fft);
	Ay_ps_fft[aps_fftindex][1] *= 1.0/sqrt(Nx_fft)/sqrt(Ny_fft)/sqrt(Nz_fft);

	Az_ps_fft[aps_fftindex][0] *= 1.0/sqrt(Nx_fft)/sqrt(Ny_fft)/sqrt(Nz_fft);
	Az_ps_fft[aps_fftindex][1] *= 1.0/sqrt(Nx_fft)/sqrt(Ny_fft)/sqrt(Nz_fft);
      }
    }
  }

  //Print the power spectrum.

  out_Ax_ps = fopen(Axps.str().c_str(), "w");
  for(i=0; i<Nx_fft/2+1; i++)
  {
    for(j=0; j<Ny_fft/2+1; j++)
    {
      for(k=0; k<Nz_fft/2+1; k++)
      {
	wave_no = sqrt(i*i+j*j+k*k);
	aps_fftindex = k+(Nz_fft/2+1)*(j+(Ny_fft)*i);
	Axps_mag = Ax_ps_fft[aps_fftindex][0]*Ax_ps_fft[aps_fftindex][0]
	         + Ax_ps_fft[aps_fftindex][1]*Ax_ps_fft[aps_fftindex][1];
	fprintf(out_Ax_ps, "%i %i %i %e %e\n",
		i, j, k, wave_no, Axps_mag);
      }
    }
  }
  fclose(out_Ax_ps);

  out_Ay_ps = fopen(Ayps.str().c_str(), "w");
  for(i=0; i<Nx_fft/2+1; i++)
  {
    for(j=0; j<Ny_fft/2+1; j++)
    {
      for(k=0; k<Nz_fft/2+1; k++)
      {
	wave_no = sqrt(i*i+j*j+k*k);
	aps_fftindex = k+(Nz_fft/2+1)*(j+(Ny_fft)*i);
	Ayps_mag = Ay_ps_fft[aps_fftindex][0]*Ay_ps_fft[aps_fftindex][0]
	         + Ay_ps_fft[aps_fftindex][1]*Ay_ps_fft[aps_fftindex][1];
	fprintf(out_Ay_ps, "%i %i %i %e %e\n",
		i, j, k, wave_no, Ayps_mag);
      }
    }
  }
  fclose(out_Ay_ps);

  out_Az_ps = fopen(Azps.str().c_str(), "w");
  for(i=0; i<Nx_fft/2+1; i++)
  {
    for(j=0; j<Ny_fft/2+1; j++)
    {
      for(k=0; k<Nz_fft/2+1; k++)
      {
	wave_no = sqrt(i*i+j*j+k*k);
	aps_fftindex = k+(Nz_fft/2+1)*(j+(Ny_fft)*i);
	Azps_mag = Az_ps_fft[aps_fftindex][0]*Az_ps_fft[aps_fftindex][0]
	         + Az_ps_fft[aps_fftindex][1]*Az_ps_fft[aps_fftindex][1];
	fprintf(out_Az_ps, "%i %i %i %e %e\n",
		i, j, k, wave_no, Azps_mag);
      }
    }
  }
  fclose(out_Az_ps);

  delete[] Ax_ps;
  delete[] Ay_ps;
  delete[] Az_ps;

  delete[] Ax_ps_fft;
  delete[] Ay_ps_fft;
  delete[] Az_ps_fft;  

  fftw_destroy_plan(forward_Ax_ps);
  fftw_destroy_plan(forward_Ay_ps);
  fftw_destroy_plan(forward_Az_ps);

  return;
}

void derivs (double* Ax,double* Ay,double* Az,int flag)
{
  //This function calculates and outputs the second, third,
  //and fourth derivatives of the A field to test the 
  //properties of the field. It also prints the 
  //root-mean-square of each derivative. It is called both 
  //before and after the gauge transformation, so the flag 
  //is there to distinguish between these two calls.

  cout << endl;
  if (flag == 1)
  {
    cout << "Calculating the derivatives of A" << endl;
  }
  else if (flag == 2)
  {
    cout << "Calculating the derivatives of A_c" << endl;
  }
  cout << endl;

  double *AxD2 = new double[(Nx-4)*(Ny-3)*(Nz-3)];
  double *AyD2 = new double[(Nx-3)*(Ny-4)*(Nz-3)];
  double *AzD2 = new double[(Nx-3)*(Ny-3)*(Nz-4)];
  double *AxD3 = new double[(Nx-4)*(Ny-3)*(Nz-3)];
  double *AyD3 = new double[(Nx-3)*(Ny-4)*(Nz-3)];
  double *AzD3 = new double[(Nx-3)*(Ny-3)*(Nz-4)];
  double *AxD4 = new double[(Nx-4)*(Ny-3)*(Nz-3)];
  double *AyD4 = new double[(Nx-3)*(Ny-4)*(Nz-3)];
  double *AzD4 = new double[(Nx-3)*(Ny-3)*(Nz-4)];

  int Ntot_derx = (Nx-4)*(Ny-3)*(Nz-3);
  int Ntot_dery = (Nx-3)*(Ny-4)*(Nz-3);
  int Ntot_derz = (Nx-3)*(Ny-3)*(Nz-4);

  int axderindex, ayderindex, azderindex;
  int axindex_xneg2, axindex_xneg1, axindex_x0, axindex_xpos1, axindex_xpos2;
  int axindex_yneg2, axindex_yneg1, axindex_y0, axindex_ypos1, axindex_ypos2;
  int axindex_zneg2, axindex_zneg1, axindex_z0, axindex_zpos1, axindex_zpos2;
  int ayindex_xneg2, ayindex_xneg1, ayindex_x0, ayindex_xpos1, ayindex_xpos2;
  int ayindex_yneg2, ayindex_yneg1, ayindex_y0, ayindex_ypos1, ayindex_ypos2;
  int ayindex_zneg2, ayindex_zneg1, ayindex_z0, ayindex_zpos1, ayindex_zpos2;
  int azindex_xneg2, azindex_xneg1, azindex_x0, azindex_xpos1, azindex_xpos2;
  int azindex_yneg2, azindex_yneg1, azindex_y0, azindex_ypos1, azindex_ypos2;
  int azindex_zneg2, azindex_zneg1, azindex_z0, azindex_zpos1, azindex_zpos2;

  double Ax_xneg2, Ax_xneg1, Ax_x0, Ax_xpos1, Ax_xpos2;
  double Ax_yneg2, Ax_yneg1, Ax_y0, Ax_ypos1, Ax_ypos2;
  double Ax_zneg2, Ax_zneg1, Ax_z0, Ax_zpos1, Ax_zpos2;
  double Ay_xneg2, Ay_xneg1, Ay_x0, Ay_xpos1, Ay_xpos2;
  double Ay_yneg2, Ay_yneg1, Ay_y0, Ay_ypos1, Ay_ypos2;
  double Ay_zneg2, Ay_zneg1, Ay_z0, Ay_zpos1, Ay_zpos2;
  double Az_xneg2, Az_xneg1, Az_x0, Az_xpos1, Az_xpos2;
  double Az_yneg2, Az_yneg1, Az_y0, Az_ypos1, Az_ypos2;
  double Az_zneg2, Az_zneg1, Az_z0, Az_zpos1, Az_zpos2;

  double xterm, yterm, zterm;
  double AxD2_tot = 0.0;
  double AxD3_tot = 0.0;
  double AxD4_tot = 0.0;
  double AyD2_tot = 0.0;
  double AyD3_tot = 0.0;
  double AyD4_tot = 0.0;
  double AzD2_tot = 0.0;
  double AzD3_tot = 0.0;
  double AzD4_tot = 0.0;

  double AxD2_RMS, AxD3_RMS, AxD4_RMS;
  double AyD2_RMS, AyD3_RMS, AyD4_RMS;
  double AzD2_RMS, AzD3_RMS, AzD4_RMS;

  FILE *out_AxiDer, *out_AxjDer, *out_AxkDer;
  FILE *out_AyiDer, *out_AyjDer, *out_AykDer;
  FILE *out_AziDer, *out_AzjDer, *out_AzkDer;
  FILE *out_AxDer, *out_AyDer, *out_AzDer;

  ostringstream AxiDer, AxjDer, AxkDer;
  ostringstream AyiDer, AyjDer, AykDer;
  ostringstream AziDer, AzjDer, AzkDer;
  ostringstream AxDer, AyDer, AzDer;

  if (flag == 1)
  {
    AxiDer << "Ax_derivs_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    AxjDer << "Ax_derivs_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    AxkDer << "Ax_derivs_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    AyiDer << "Ay_derivs_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    AyjDer << "Ay_derivs_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    AykDer << "Ay_derivs_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    AziDer << "Az_derivs_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    AzjDer << "Az_derivs_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    AzkDer << "Az_derivs_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    AxDer << "Ax_derivs.dat";
    AyDer << "Ay_derivs.dat";
    AzDer << "Az_derivs.dat";
  }
  else if (flag == 2)
  {
    AxiDer << "Axc_derivs_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    AxjDer << "Axc_derivs_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    AxkDer << "Axc_derivs_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    AyiDer << "Ayc_derivs_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    AyjDer << "Ayc_derivs_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    AykDer << "Ayc_derivs_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    AziDer << "Azc_derivs_i" << setfill('0') << setw(3) << cutxindex << ".dat";
    AzjDer << "Azc_derivs_j" << setfill('0') << setw(3) << cutyindex << ".dat";
    AzkDer << "Azc_derivs_k" << setfill('0') << setw(3) << cutzindex << ".dat";
    AxDer << "Axc_derivs.dat";
    AyDer << "Ayc_derivs.dat";
    AzDer << "Azc_derivs.dat";
  }

  //Second Derivative

  for (i=2; i<Nx-2; i++)
  {
    for (j=2; j<Ny-1; j++)
    {
      for (k=2; k<Nz-1; k++)
      {
	axindex_xneg2 = k+(Nz+1)*(j+(Ny+1)*(i-2));
	axindex_xneg1 = k+(Nz+1)*(j+(Ny+1)*(i-1));
	axindex_x0 = k+(Nz+1)*(j+(Ny+1)*i);
	axindex_xpos1 = k+(Nz+1)*(j+(Ny+1)*(i+1));
	axindex_xpos2 = k+(Nz+1)*(j+(Ny+1)*(i+2));

	axindex_yneg2 = k+(Nz+1)*((j-2)+(Ny+1)*i);
	axindex_yneg1 = k+(Nz+1)*((j-1)+(Ny+1)*i);
	axindex_y0 = k+(Nz+1)*(j+(Ny+1)*i);
	axindex_ypos1 = k+(Nz+1)*((j+1)+(Ny+1)*i);
	axindex_ypos2 = k+(Nz+1)*((j+2)+(Ny+1)*i);

	axindex_zneg2 = (k-2)+(Nz+1)*(j+(Ny+1)*i);
	axindex_zneg1 = (k-1)+(Nz+1)*(j+(Ny+1)*i);
	axindex_z0 = k+(Nz+1)*(j+(Ny+1)*i);
	axindex_zpos1 = (k+1)+(Nz+1)*(j+(Ny+1)*i);
	axindex_zpos2 = (k+2)+(Nz+1)*(j+(Ny+1)*i);

	Ax_xneg2 = Ax[axindex_xneg2];
	Ax_xneg1 = Ax[axindex_xneg1];
	Ax_x0 = Ax[axindex_x0];
	Ax_xpos1 = Ax[axindex_xpos1];
	Ax_xpos2 = Ax[axindex_xpos2];

	Ax_yneg2 = Ax[axindex_yneg2];
	Ax_yneg1 = Ax[axindex_yneg1];
	Ax_y0 = Ax[axindex_y0];
	Ax_ypos1 = Ax[axindex_ypos1];
	Ax_ypos2 = Ax[axindex_ypos2];

	Ax_zneg2 = Ax[axindex_zneg2];
	Ax_zneg1 = Ax[axindex_zneg1];
	Ax_z0 = Ax[axindex_z0];
	Ax_zpos1 = Ax[axindex_zpos1];
	Ax_zpos2 = Ax[axindex_zpos2];

	xterm = (-1.0*Ax_xneg2
		 +16.0*Ax_xneg1
		 -30.0*Ax_x0
		 +16.0*Ax_xpos1
		 -1.0*Ax_xpos2)/12.0/stepx/stepx;

	yterm = (-1.0*Ax_yneg2
		 +16.0*Ax_yneg1
		 -30.0*Ax_y0
		 +16.0*Ax_ypos1
		 -1.0*Ax_ypos2)/12.0/stepy/stepy;

	zterm = (-1.0*Ax_zneg2
		 +16.0*Ax_zneg1
		 -30.0*Ax_z0
		 +16.0*Ax_zpos1
		 -1.0*Ax_zpos2)/12.0/stepz/stepz;


	axderindex = (k-2)+(Nz-3)*((j-2)+(Ny-3)*(i-2));

	AxD2[axderindex] = xterm*xterm + yterm*yterm + zterm*zterm;
      }
    }
  }

  for (i=2; i<Nx-1; i++)
  {
    for (j=2; j<Ny-2; j++)
    {
      for (k=2; k<Nz-1; k++)
      {
	ayindex_xneg2 = k+(Nz+1)*(j+(Ny)*(i-2));
	ayindex_xneg1 = k+(Nz+1)*(j+(Ny)*(i-1));
	ayindex_x0 = k+(Nz+1)*(j+(Ny)*i);
	ayindex_xpos1 = k+(Nz+1)*(j+(Ny)*(i+1));
	ayindex_xpos2 = k+(Nz+1)*(j+(Ny)*(i+2));

	ayindex_yneg2 = k+(Nz+1)*((j-2)+(Ny)*i);
	ayindex_yneg1 = k+(Nz+1)*((j-1)+(Ny)*i);
	ayindex_y0 = k+(Nz+1)*(j+(Ny)*i);
	ayindex_ypos1 = k+(Nz+1)*((j+1)+(Ny)*i);
	ayindex_ypos2 = k+(Nz+1)*((j+2)+(Ny)*i);

	ayindex_zneg2 = (k-2)+(Nz+1)*(j+(Ny)*i);
	ayindex_zneg1 = (k-1)+(Nz+1)*(j+(Ny)*i);
	ayindex_z0 = k+(Nz+1)*(j+(Ny)*i);
	ayindex_zpos1 = (k+1)+(Nz+1)*(j+(Ny)*i);
	ayindex_zpos2 = (k+2)+(Nz+1)*(j+(Ny)*i);

	Ay_xneg2 = Ay[ayindex_xneg2];
	Ay_xneg1 = Ay[ayindex_xneg1];
	Ay_x0 = Ay[ayindex_x0];
	Ay_xpos1 = Ay[ayindex_xpos1];
	Ay_xpos2 = Ay[ayindex_xpos2];

	Ay_yneg2 = Ay[ayindex_yneg2];
	Ay_yneg1 = Ay[ayindex_yneg1];
	Ay_y0 = Ay[ayindex_y0];
	Ay_ypos1 = Ay[ayindex_ypos1];
	Ay_ypos2 = Ay[ayindex_ypos2];

	Ay_zneg2 = Ay[ayindex_zneg2];
	Ay_zneg1 = Ay[ayindex_zneg1];
	Ay_z0 = Ay[ayindex_z0];
	Ay_zpos1 = Ay[ayindex_zpos1];
	Ay_zpos2 = Ay[ayindex_zpos2];

	xterm = (-1.0*Ay_xneg2
		 +16.0*Ay_xneg1
		 -30.0*Ay_x0
		 +16.0*Ay_xpos1
		 -1.0*Ay_xpos2)/12.0/stepx/stepx;

	yterm = (-1.0*Ay_yneg2
		 +16.0*Ay_yneg1
		 -30.0*Ay_y0
		 +16.0*Ay_ypos1
		 -1.0*Ay_ypos2)/12.0/stepy/stepy;

	zterm = (-1.0*Ay_zneg2
		 +16.0*Ay_zneg1
		 -30.0*Ay_z0
		 +16.0*Ay_zpos1
		 -1.0*Ay_zpos2)/12.0/stepz/stepz;

	ayderindex = (k-2)+(Nz-3)*((j-2)+(Ny-4)*(i-2));

	AyD2[ayderindex] = xterm*xterm + yterm*yterm + zterm*zterm;
      }
    }
  }

  for (i=2; i<Nx-1; i++)
  {
    for (j=2; j<Ny-1; j++)
    {
      for (k=2; k<Nz-2; k++)
      {
	azindex_xneg2 = k+(Nz)*(j+(Ny+1)*(i-2));
	azindex_xneg1 = k+(Nz)*(j+(Ny+1)*(i-1));
	azindex_x0 = k+(Nz)*(j+(Ny+1)*i);
	azindex_xpos1 = k+(Nz)*(j+(Ny+1)*(i+1));
	azindex_xpos2 = k+(Nz)*(j+(Ny+1)*(i+2));

	azindex_yneg2 = k+(Nz)*((j-2)+(Ny+1)*i);
	azindex_yneg1 = k+(Nz)*((j-1)+(Ny+1)*i);
	azindex_y0 = k+(Nz)*(j+(Ny+1)*i);
	azindex_ypos1 = k+(Nz)*((j+1)+(Ny+1)*i);
	azindex_ypos2 = k+(Nz)*((j+2)+(Ny+1)*i);

	azindex_zneg2 = (k-2)+(Nz)*(j+(Ny+1)*i);
	azindex_zneg1 = (k-1)+(Nz)*(j+(Ny+1)*i);
	azindex_z0 = k+(Nz)*(j+(Ny+1)*i);
	azindex_zpos1 = (k+1)+(Nz)*(j+(Ny+1)*i);
	azindex_zpos2 = (k+2)+(Nz)*(j+(Ny+1)*i);

	Az_xneg2 = Az[azindex_xneg2];
	Az_xneg1 = Az[azindex_xneg1];
	Az_x0 = Az[azindex_x0];
	Az_xpos1 = Az[azindex_xpos1];
	Az_xpos2 = Az[azindex_xpos2];

	Az_yneg2 = Az[azindex_yneg2];
	Az_yneg1 = Az[azindex_yneg1];
	Az_y0 = Az[azindex_y0];
	Az_ypos1 = Az[azindex_ypos1];
	Az_ypos2 = Az[azindex_ypos2];

	Az_zneg2 = Az[azindex_zneg2];
	Az_zneg1 = Az[azindex_zneg1];
	Az_z0 = Az[azindex_z0];
	Az_zpos1 = Az[azindex_zpos1];
	Az_zpos2 = Az[azindex_zpos2];

	xterm = (-1.0*Az_xneg2
		 +16.0*Az_xneg1
		 -30.0*Az_x0
		 +16.0*Az_xpos1
		 -1.0*Az_xpos2)/12.0/stepx/stepx;

	yterm = (-1.0*Az_yneg2
		 +16.0*Az_yneg1
		 -30.0*Az_y0
		 +16.0*Az_ypos1
		 -1.0*Az_ypos2)/12.0/stepy/stepy;

	zterm = (-1.0*Az_zneg2
		 +16.0*Az_zneg1
		 -30.0*Az_z0
		 +16.0*Az_zpos1
		 -1.0*Az_zpos2)/12.0/stepz/stepz;

	azderindex = (k-2)+(Nz-4)*((j-2)+(Ny-3)*(i-2));

	AzD2[azderindex] = xterm*xterm + yterm*yterm + zterm*zterm;
      }
    }
  }

  //Third Derivative

  for (i=2; i<Nx-2; i++)
  {
    for (j=2; j<Ny-1; j++)
    {
      for (k=2; k<Nz-1; k++)
      {
	axindex_xneg2 = k+(Nz+1)*(j+(Ny+1)*(i-2));
	axindex_xneg1 = k+(Nz+1)*(j+(Ny+1)*(i-1));
	axindex_xpos1 = k+(Nz+1)*(j+(Ny+1)*(i+1));
	axindex_xpos2 = k+(Nz+1)*(j+(Ny+1)*(i+2));

	axindex_yneg2 = k+(Nz+1)*((j-2)+(Ny+1)*i);
	axindex_yneg1 = k+(Nz+1)*((j-1)+(Ny+1)*i);
	axindex_ypos1 = k+(Nz+1)*((j+1)+(Ny+1)*i);
	axindex_ypos2 = k+(Nz+1)*((j+2)+(Ny+1)*i);

	axindex_zneg2 = (k-2)+(Nz+1)*(j+(Ny+1)*i);
	axindex_zneg1 = (k-1)+(Nz+1)*(j+(Ny+1)*i);
	axindex_zpos1 = (k+1)+(Nz+1)*(j+(Ny+1)*i);
	axindex_zpos2 = (k+2)+(Nz+1)*(j+(Ny+1)*i);

	Ax_xneg2 = Ax[axindex_xneg2];
	Ax_xneg1 = Ax[axindex_xneg1];
	Ax_xpos1 = Ax[axindex_xpos1];
	Ax_xpos2 = Ax[axindex_xpos2];

	Ax_yneg2 = Ax[axindex_yneg2];
	Ax_yneg1 = Ax[axindex_yneg1];
	Ax_ypos1 = Ax[axindex_ypos1];
	Ax_ypos2 = Ax[axindex_ypos2];

	Ax_zneg2 = Ax[axindex_zneg2];
	Ax_zneg1 = Ax[axindex_zneg1];
	Ax_zpos1 = Ax[axindex_zpos1];
	Ax_zpos2 = Ax[axindex_zpos2];

	xterm = (-1.0*Ax_xneg2
		 +2.0*Ax_xneg1
		 -2.0*Ax_xpos1
		 +1.0*Ax_xpos2)/2.0/pow(stepx,3);

	yterm = (-1.0*Ax_yneg2
		 +2.0*Ax_yneg1
		 -2.0*Ax_ypos1
		 +1.0*Ax_ypos2)/2.0/pow(stepy,3);

	zterm = (-1.0*Ax_zneg2
		 +2.0*Ax_zneg1
		 -2.0*Ax_zpos1
		 +1.0*Ax_zpos2)/2.0/pow(stepz,3);

	axderindex = (k-2)+(Nz-3)*((j-2)+(Ny-3)*(i-2));

	AxD3[axderindex] = xterm*xterm + yterm*yterm + zterm*zterm;
      }
    }
  }

  for (i=2; i<Nx-1; i++)
  {
    for (j=2; j<Ny-2; j++)
    {
      for (k=2; k<Nz-1; k++)
      {
	ayindex_xneg2 = k+(Nz+1)*(j+(Ny)*(i-2));
	ayindex_xneg1 = k+(Nz+1)*(j+(Ny)*(i-1));
	ayindex_xpos1 = k+(Nz+1)*(j+(Ny)*(i+1));
	ayindex_xpos2 = k+(Nz+1)*(j+(Ny)*(i+2));

	ayindex_yneg2 = k+(Nz+1)*((j-2)+(Ny)*i);
	ayindex_yneg1 = k+(Nz+1)*((j-1)+(Ny)*i);
	ayindex_ypos1 = k+(Nz+1)*((j+1)+(Ny)*i);
	ayindex_ypos2 = k+(Nz+1)*((j+2)+(Ny)*i);

	ayindex_zneg2 = (k-2)+(Nz+1)*(j+(Ny)*i);
	ayindex_zneg1 = (k-1)+(Nz+1)*(j+(Ny)*i);
	ayindex_zpos1 = (k+1)+(Nz+1)*(j+(Ny)*i);
	ayindex_zpos2 = (k+2)+(Nz+1)*(j+(Ny)*i);

	Ay_xneg2 = Ay[ayindex_xneg2];
	Ay_xneg1 = Ay[ayindex_xneg1];
	Ay_xpos1 = Ay[ayindex_xpos1];
	Ay_xpos2 = Ay[ayindex_xpos2];

	Ay_yneg2 = Ay[ayindex_yneg2];
	Ay_yneg1 = Ay[ayindex_yneg1];
	Ay_ypos1 = Ay[ayindex_ypos1];
	Ay_ypos2 = Ay[ayindex_ypos2];

	Ay_zneg2 = Ay[ayindex_zneg2];
	Ay_zneg1 = Ay[ayindex_zneg1];
	Ay_zpos1 = Ay[ayindex_zpos1];
	Ay_zpos2 = Ay[ayindex_zpos2];

	xterm = (-1.0*Ay_xneg2
		 +2.0*Ay_xneg1
		 -2.0*Ay_xpos1
		 +1.0*Ay_xpos2)/2.0/pow(stepx,3);

	yterm = (-1.0*Ay_yneg2
		 +2.0*Ay_yneg1
		 -2.0*Ay_ypos1
		 +1.0*Ay_ypos2)/2.0/pow(stepy,3);

	zterm = (-1.0*Ay_zneg2
		 +2.0*Ay_zneg1
		 -2.0*Ay_zpos1
		 +1.0*Ay_zpos2)/2.0/pow(stepz,3);

	ayderindex = (k-2)+(Nz-3)*((j-2)+(Ny-4)*(i-2));

	AyD3[ayderindex] = xterm*xterm + yterm*yterm + zterm*zterm;
      }
    }
  }

  for (i=2; i<Nx-1; i++)
  {
    for (j=2; j<Ny-1; j++)
    {
      for (k=2; k<Nz-2; k++)
      {
	azindex_xneg2 = k+(Nz)*(j+(Ny+1)*(i-2));
	azindex_xneg1 = k+(Nz)*(j+(Ny+1)*(i-1));
	azindex_xpos1 = k+(Nz)*(j+(Ny+1)*(i+1));
	azindex_xpos2 = k+(Nz)*(j+(Ny+1)*(i+2));

	azindex_yneg2 = k+(Nz)*((j-2)+(Ny+1)*i);
	azindex_yneg1 = k+(Nz)*((j-1)+(Ny+1)*i);
	azindex_ypos1 = k+(Nz)*((j+1)+(Ny+1)*i);
	azindex_ypos2 = k+(Nz)*((j+2)+(Ny+1)*i);

	azindex_zneg2 = (k-2)+(Nz)*(j+(Ny+1)*i);
	azindex_zneg1 = (k-1)+(Nz)*(j+(Ny+1)*i);
	azindex_zpos1 = (k+1)+(Nz)*(j+(Ny+1)*i);
	azindex_zpos2 = (k+2)+(Nz)*(j+(Ny+1)*i);

	Az_xneg2 = Az[azindex_xneg2];
	Az_xneg1 = Az[azindex_xneg1];
	Az_xpos1 = Az[azindex_xpos1];
	Az_xpos2 = Az[azindex_xpos2];

	Az_yneg2 = Az[azindex_yneg2];
	Az_yneg1 = Az[azindex_yneg1];
	Az_ypos1 = Az[azindex_ypos1];
	Az_ypos2 = Az[azindex_ypos2];

	Az_zneg2 = Az[azindex_zneg2];
	Az_zneg1 = Az[azindex_zneg1];
	Az_zpos1 = Az[azindex_zpos1];
	Az_zpos2 = Az[azindex_zpos2];

	xterm = (-1.0*Az_xneg2
		 +2.0*Az_xneg1
		 -2.0*Az_xpos1
		 +1.0*Az_xpos2)/2.0/pow(stepx,3);

	yterm = (-1.0*Az_yneg2
		 +2.0*Az_yneg1
		 -2.0*Az_ypos1
		 +1.0*Az_ypos2)/2.0/pow(stepy,3);

	zterm = (-1.0*Az_zneg2
		 +2.0*Az_zneg1
		 -2.0*Az_zpos1
		 +1.0*Az_zpos2)/2.0/pow(stepz,3);

	azderindex = (k-2)+(Nz-4)*((j-2)+(Ny-3)*(i-2));

	AzD3[azderindex] = xterm*xterm + yterm*yterm + zterm*zterm;
      }
    }
  }

  //Fourth Derivative

  for (i=2; i<Nx-2; i++)
  {
    for (j=2; j<Ny-1; j++)
    {
      for (k=2; k<Nz-1; k++)
      {
	axindex_xneg2 = k+(Nz+1)*(j+(Ny+1)*(i-2));
	axindex_xneg1 = k+(Nz+1)*(j+(Ny+1)*(i-1));
	axindex_x0 = k+(Nz+1)*(j+(Ny+1)*i);
	axindex_xpos1 = k+(Nz+1)*(j+(Ny+1)*(i+1));
	axindex_xpos2 = k+(Nz+1)*(j+(Ny+1)*(i+2));

	axindex_yneg2 = k+(Nz+1)*((j-2)+(Ny+1)*i);
	axindex_yneg1 = k+(Nz+1)*((j-1)+(Ny+1)*i);
	axindex_y0 = k+(Nz+1)*(j+(Ny+1)*i);
	axindex_ypos1 = k+(Nz+1)*((j+1)+(Ny+1)*i);
	axindex_ypos2 = k+(Nz+1)*((j+2)+(Ny+1)*i);

	axindex_zneg2 = (k-2)+(Nz+1)*(j+(Ny+1)*i);
	axindex_zneg1 = (k-1)+(Nz+1)*(j+(Ny+1)*i);
	axindex_z0 = k+(Nz+1)*(j+(Ny+1)*i);
	axindex_zpos1 = (k+1)+(Nz+1)*(j+(Ny+1)*i);
	axindex_zpos2 = (k+2)+(Nz+1)*(j+(Ny+1)*i);

	Ax_xneg2 = Ax[axindex_xneg2];
	Ax_xneg1 = Ax[axindex_xneg1];
	Ax_x0 = Ax[axindex_x0];
	Ax_xpos1 = Ax[axindex_xpos1];
	Ax_xpos2 = Ax[axindex_xpos2];

	Ax_yneg2 = Ax[axindex_yneg2];
	Ax_yneg1 = Ax[axindex_yneg1];
	Ax_y0 = Ax[axindex_y0];
	Ax_ypos1 = Ax[axindex_ypos1];
	Ax_ypos2 = Ax[axindex_ypos2];

	Ax_zneg2 = Ax[axindex_zneg2];
	Ax_zneg1 = Ax[axindex_zneg1];
	Ax_z0 = Ax[axindex_z0];
	Ax_zpos1 = Ax[axindex_zpos1];
	Ax_zpos2 = Ax[axindex_zpos2];

	xterm = ( 1.0*Ax_xneg2
		 -4.0*Ax_xneg1
		 +6.0*Ax_x0
	         -4.0*Ax_xpos1
		 +1.0*Ax_xpos2)/pow(stepx,4);

	yterm = ( 1.0*Ax_yneg2
		 -4.0*Ax_yneg1
		 +6.0*Ax_y0
	         -4.0*Ax_ypos1
		 +1.0*Ax_ypos2)/pow(stepy,4);

	zterm = ( 1.0*Ax_zneg2
		 -4.0*Ax_zneg1
		 +6.0*Ax_z0
	         -4.0*Ax_zpos1
		 +1.0*Ax_zpos2)/pow(stepz,4);

	axderindex = (k-2)+(Nz-3)*((j-2)+(Ny-3)*(i-2));

	AxD4[axderindex] = xterm*xterm + yterm*yterm + zterm*zterm;
      }
    }
  }

  for (i=2; i<Nx-1; i++)
  {
    for (j=2; j<Ny-2; j++)
    {
      for (k=2; k<Nz-1; k++)
      {
	ayindex_xneg2 = k+(Nz+1)*(j+(Ny)*(i-2));
	ayindex_xneg1 = k+(Nz+1)*(j+(Ny)*(i-1));
	ayindex_x0 = k+(Nz+1)*(j+(Ny)*i);
	ayindex_xpos1 = k+(Nz+1)*(j+(Ny)*(i+1));
	ayindex_xpos2 = k+(Nz+1)*(j+(Ny)*(i+2));

	ayindex_yneg2 = k+(Nz+1)*((j-2)+(Ny)*i);
	ayindex_yneg1 = k+(Nz+1)*((j-1)+(Ny)*i);
	ayindex_y0 = k+(Nz+1)*(j+(Ny)*i);
	ayindex_ypos1 = k+(Nz+1)*((j+1)+(Ny)*i);
	ayindex_ypos2 = k+(Nz+1)*((j+2)+(Ny)*i);

	ayindex_zneg2 = (k-2)+(Nz+1)*(j+(Ny)*i);
	ayindex_zneg1 = (k-1)+(Nz+1)*(j+(Ny)*i);
	ayindex_z0 = k+(Nz+1)*(j+(Ny)*i);
	ayindex_zpos1 = (k+1)+(Nz+1)*(j+(Ny)*i);
	ayindex_zpos2 = (k+2)+(Nz+1)*(j+(Ny)*i);

	Ay_xneg2 = Ay[ayindex_xneg2];
	Ay_xneg1 = Ay[ayindex_xneg1];
	Ay_x0 = Ay[ayindex_x0];
	Ay_xpos1 = Ay[ayindex_xpos1];
	Ay_xpos2 = Ay[ayindex_xpos2];

	Ay_yneg2 = Ay[ayindex_yneg2];
	Ay_yneg1 = Ay[ayindex_yneg1];
	Ay_y0 = Ay[ayindex_y0];
	Ay_ypos1 = Ay[ayindex_ypos1];
	Ay_ypos2 = Ay[ayindex_ypos2];

	Ay_zneg2 = Ay[ayindex_zneg2];
	Ay_zneg1 = Ay[ayindex_zneg1];
	Ay_z0 = Ay[ayindex_z0];
	Ay_zpos1 = Ay[ayindex_zpos1];
	Ay_zpos2 = Ay[ayindex_zpos2];

	xterm = ( 1.0*Ay_xneg2
		 -4.0*Ay_xneg1
		 +6.0*Ay_x0
		 -4.0*Ay_xpos1
		 +1.0*Ay_xpos2)/pow(stepx,4);

	yterm = ( 1.0*Ay_yneg2
		 -4.0*Ay_yneg1
		 +6.0*Ay_y0
		 -4.0*Ay_ypos1
		 +1.0*Ay_ypos2)/pow(stepy,4);

	zterm = ( 1.0*Ay_zneg2
		 -4.0*Ay_zneg1
		 +6.0*Ay_z0
		 -4.0*Ay_zpos1
		 +1.0*Ay_zpos2)/pow(stepz,4);

	ayderindex = (k-2)+(Nz-3)*((j-2)+(Ny-4)*(i-2));

	AyD4[ayderindex] = xterm*xterm + yterm*yterm + zterm*zterm;
      }
    }
  }

  for (i=2; i<Nx-1; i++)
  {
    for (j=2; j<Ny-1; j++)
    {
      for (k=2; k<Nz-2; k++)
      {
	azindex_xneg2 = k+(Nz)*(j+(Ny+1)*(i-2));
	azindex_xneg1 = k+(Nz)*(j+(Ny+1)*(i-1));
	azindex_x0 = k+(Nz)*(j+(Ny+1)*i);
	azindex_xpos1 = k+(Nz)*(j+(Ny+1)*(i+1));
	azindex_xpos2 = k+(Nz)*(j+(Ny+1)*(i+2));

	azindex_yneg2 = k+(Nz)*((j-2)+(Ny+1)*i);
	azindex_yneg1 = k+(Nz)*((j-1)+(Ny+1)*i);
	azindex_y0 = k+(Nz)*(j+(Ny+1)*i);
	azindex_ypos1 = k+(Nz)*((j+1)+(Ny+1)*i);
	azindex_ypos2 = k+(Nz)*((j+2)+(Ny+1)*i);

	azindex_zneg2 = (k-2)+(Nz)*(j+(Ny+1)*i);
	azindex_zneg1 = (k-1)+(Nz)*(j+(Ny+1)*i);
	azindex_z0 = k+(Nz)*(j+(Ny+1)*i);
	azindex_zpos1 = (k+1)+(Nz)*(j+(Ny+1)*i);
	azindex_zpos2 = (k+2)+(Nz)*(j+(Ny+1)*i);

	Az_xneg2 = Az[azindex_xneg2];
	Az_xneg1 = Az[azindex_xneg1];
	Az_x0 = Az[azindex_x0];
	Az_xpos1 = Az[azindex_xpos1];
	Az_xpos2 = Az[azindex_xpos2];

	Az_yneg2 = Az[azindex_yneg2];
	Az_yneg1 = Az[azindex_yneg1];
	Az_y0 = Az[azindex_y0];
	Az_ypos1 = Az[azindex_ypos1];
	Az_ypos2 = Az[azindex_ypos2];

	Az_zneg2 = Az[azindex_zneg2];
	Az_zneg1 = Az[azindex_zneg1];
	Az_z0 = Az[azindex_z0];
	Az_zpos1 = Az[azindex_zpos1];
	Az_zpos2 = Az[azindex_zpos2];

	xterm = ( 1.0*Az_xneg2
		 -4.0*Az_xneg1
		 +6.0*Az_x0
		 -4.0*Az_xpos1
		 +1.0*Az_xpos2)/pow(stepx,4);

	yterm = ( 1.0*Az_yneg2
		 -4.0*Az_yneg1
		 +6.0*Az_y0
		 -4.0*Az_ypos1
		 +1.0*Az_ypos2)/pow(stepy,4);

	zterm = ( 1.0*Az_zneg2
		 -4.0*Az_zneg1
		 +6.0*Az_z0
		 -4.0*Az_zpos1
		 +1.0*Az_zpos2)/pow(stepz,4);

	azderindex = (k-2)+(Nz-4)*((j-2)+(Ny-3)*(i-2));

	AzD4[azderindex] = xterm*xterm + yterm*yterm + zterm*zterm;
      }
    }
  }

  //Print the higher derivatives.

  out_AxiDer = fopen(AxiDer.str().c_str(), "w");
  for (j=2; j<Ny-1; j++)
  {
    for (k=2; k<Nz-1; k++)
    {
      i = cutxindex;
      axderindex = (k-2)+(Nz-3)*((j-2)+(Ny-3)*(i-2));
      fprintf(out_AxiDer, "%i %i %i %e %e %e\n",
	      i, j, k, AxD2[axderindex], AxD3[axderindex], AxD4[axderindex]);
    }
  }
  fclose(out_AxiDer);

  out_AxjDer = fopen(AxjDer.str().c_str(), "w");
  for (i=2; i<Nx-2; i++)
  {
    for (k=2; k<Nz-1; k++)
    {
      j = cutyindex;
      axderindex = (k-2)+(Nz-3)*((j-2)+(Ny-3)*(i-2));
      fprintf(out_AxjDer, "%i %i %i %e %e %e\n",
	      i, j, k, AxD2[axderindex], AxD3[axderindex], AxD4[axderindex]);
    }
  }
  fclose(out_AxjDer);

  out_AxkDer = fopen(AxkDer.str().c_str(), "w");
  for (i=2; i<Nx-2; i++)
  {
    for (j=2; j<Ny-1; j++)
    {
      k = cutzindex;
      axderindex = (k-2)+(Nz-3)*((j-2)+(Ny-3)*(i-2));
      fprintf(out_AxkDer, "%i %i %i %e %e %e\n",
	      i, j, k, AxD2[axderindex], AxD3[axderindex], AxD4[axderindex]);
    }
  }
  fclose(out_AxkDer);

  out_AyiDer = fopen(AyiDer.str().c_str(), "w");
  for (j=2; j<Ny-2; j++)
  {
    for (k=2; k<Nz-1; k++)
    {
      i = cutxindex;
      ayderindex = (k-2)+(Nz-3)*((j-2)+(Ny-4)*(i-2));
      fprintf(out_AyiDer, "%i %i %i %e %e %e\n",
	      i, j, k, AyD2[ayderindex], AyD3[ayderindex], AyD4[ayderindex]);
    }
  }
  fclose(out_AyiDer);

  out_AyjDer = fopen(AyjDer.str().c_str(), "w");
  for (i=2; i<Nx-1; i++)
  {
    for (k=2; k<Nz-1; k++)
    {
      j = cutyindex;
      ayderindex = (k-2)+(Nz-3)*((j-2)+(Ny-4)*(i-2));
      fprintf(out_AyjDer, "%i %i %i %e %e %e\n",
	      i, j, k, AyD2[ayderindex], AyD3[ayderindex], AyD4[ayderindex]);
    }
  }
  fclose(out_AyjDer);

  out_AykDer = fopen(AykDer.str().c_str(), "w");
  for (i=2; i<Nx-1; i++)
  {
    for (j=2; j<Ny-2; j++)
    {
      k = cutzindex;
      ayderindex = (k-2)+(Nz-3)*((j-2)+(Ny-4)*(i-2));
      fprintf(out_AykDer, "%i %i %i %e %e %e\n",
	      i, j, k, AyD2[ayderindex], AyD3[ayderindex], AyD4[ayderindex]);
    }
  }
  fclose(out_AykDer);

  out_AziDer = fopen(AziDer.str().c_str(), "w");
  for (j=2; j<Ny-1; j++)
  {
    for (k=2; k<Nz-2; k++)
    {
      i = cutxindex;
      azderindex = (k-2)+(Nz-4)*((j-2)+(Ny-3)*(i-2));
      fprintf(out_AziDer, "%i %i %i %e %e %e\n",
	      i, j, k, AzD2[azderindex], AzD3[azderindex], AzD4[azderindex]);
    }
  }
  fclose(out_AziDer);

  out_AzjDer = fopen(AzjDer.str().c_str(), "w");
  for (i=2; i<Nx-1; i++)
  {
    for (k=2; k<Nz-2; k++)
    {
      j = cutyindex;
      azderindex = (k-2)+(Nz-4)*((j-2)+(Ny-3)*(i-2));
      fprintf(out_AzjDer, "%i %i %i %e %e %e\n",
	      i, j, k, AzD2[azderindex], AzD3[azderindex], AzD4[azderindex]);
    }
  }
  fclose(out_AzjDer);

  out_AzkDer = fopen(AzkDer.str().c_str(), "w");
  for (i=2; i<Nx-1; i++)
  {
    for (j=2; j<Ny-1; j++)
    {
      k = cutzindex;
      azderindex = (k-2)+(Nz-4)*((j-2)+(Ny-3)*(i-2));
      fprintf(out_AzkDer, "%i %i %i %e %e %e\n",
	      i, j, k, AzD2[azderindex], AzD3[azderindex], AzD4[azderindex]);
    }
  }
  fclose(out_AzkDer);

  //Print the full higher derivatives.

  out_AxDer = fopen(AxDer.str().c_str(), "w");
  for (i=2; i<Nx-2; i++)
  {
    for (j=2; j<Ny-1; j++)
    {
      for (k=2; k<Nz-1; k++)
      {
	axderindex = (k-2)+(Nz-3)*((j-2)+(Ny-3)*(i-2));
	fprintf(out_AxDer, "%i %i %i %e %e %e\n",
		i, j, k, AxD2[axderindex], AxD3[axderindex], AxD4[axderindex]);
      }
    }
  }
  fclose(out_AxDer);

  out_AyDer = fopen(AyDer.str().c_str(), "w");
  for (i=2; i<Nx-1; i++)
  {
    for (j=2; j<Ny-2; j++)
    {
      for (k=2; k<Nz-1; k++)
      {
	ayderindex = (k-2)+(Nz-3)*((j-2)+(Ny-4)*(i-2));
	fprintf(out_AyDer, "%i %i %i %e %e %e\n",
		i, j, k, AyD2[ayderindex], AyD3[ayderindex], AyD4[ayderindex]);
      }
    }
  }
  fclose(out_AyDer);

  out_AzDer = fopen(AzDer.str().c_str(), "w");
  for (i=2; i<Nx-1; i++)
  {
    for (j=2; j<Ny-1; j++)
    {
      for (k=2; k<Nz-2; k++)
      {
	azderindex = (k-2)+(Nz-4)*((j-2)+(Ny-3)*(i-2));
	fprintf(out_AzDer, "%i %i %i %e %e %e\n",
		i, j, k, AzD2[azderindex], AzD3[azderindex], AzD4[azderindex]);
      }
    }
  }
  fclose(out_AzDer);

  //Calculate the root-mean-square of each derivative.
  
  for (i=2; i<Nx-2; i++)
  {
    for (j=2; j<Ny-1; j++)
    {
      for (k=2; k<Nz-1; k++)
      {
	axderindex = (k-2)+(Nz-3)*((j-2)+(Ny-3)*(i-2));

	AxD2_tot += AxD2[axderindex];
	AxD3_tot += AxD3[axderindex];
	AxD4_tot += AxD4[axderindex];
      }
    }
  }

  AxD2_RMS = sqrt(AxD2_tot/Ntot_derx);
  AxD3_RMS = sqrt(AxD3_tot/Ntot_derx);
  AxD4_RMS = sqrt(AxD4_tot/Ntot_derx);
  
  for (i=2; i<Nx-1; i++)
  {
    for (j=2; j<Ny-2; j++)
    {
      for (k=2; k<Nz-1; k++)
      {
	ayderindex = (k-2)+(Nz-3)*((j-2)+(Ny-4)*(i-2));

	AyD2_tot += AyD2[ayderindex];
	AyD3_tot += AyD3[ayderindex];
	AyD4_tot += AyD4[ayderindex];
      }
    }
  }

  AyD2_RMS = sqrt(AyD2_tot/Ntot_dery);
  AyD3_RMS = sqrt(AyD3_tot/Ntot_dery);
  AyD4_RMS = sqrt(AyD4_tot/Ntot_dery);
  
  for (i=2; i<Nx-1; i++)
  {
    for (j=2; j<Ny-1; j++)
    {
      for (k=2; k<Nz-2; k++)
      {
	azderindex = (k-2)+(Nz-4)*((j-2)+(Ny-3)*(i-2));

	AzD2_tot += AzD2[azderindex];
	AzD3_tot += AzD3[azderindex];
	AzD4_tot += AzD4[azderindex];
      }
    }
  }

  AzD2_RMS = sqrt(AzD2_tot/Ntot_derz);
  AzD3_RMS = sqrt(AzD3_tot/Ntot_derz);
  AzD4_RMS = sqrt(AzD4_tot/Ntot_derz);
  
  //Output the RMS values.

  cout << endl;
  cout << "Root-Mean-Square Values of Higher Derivatives:" << endl;
  cout << endl;
  cout << setiosflags(ios::fixed) << setprecision(10);
  if (flag ==1)
  {
    cout << "Second Derivative of A_x :  " << setw(15) << AxD2_RMS << endl;
    cout << "Second Derivative of A_y :  " << setw(15) << AyD2_RMS << endl;
    cout << "Second Derivative of A_z :  " << setw(15) << AzD2_RMS << endl;
    cout << endl;
    cout << "Third Derivative of A_x  :  " << setw(15) << AxD3_RMS << endl;
    cout << "Third Derivative of A_y  :  " << setw(15) << AyD3_RMS << endl;
    cout << "Third Derivative of A_z  :  " << setw(15) << AzD3_RMS << endl;
    cout << endl;
    cout << "Fourth Derivative of A_x :  " << setw(15) << AxD4_RMS << endl;
    cout << "Fourth Derivative of A_y :  " << setw(15) << AyD4_RMS << endl;
    cout << "Fourth Derivative of A_z :  " << setw(15) << AzD4_RMS << endl;
    cout << endl;
  }
  else if (flag ==2)
  {
    cout << "Second Derivative of Ac_x : " << setw(15) << AxD2_RMS << endl;
    cout << "Second Derivative of Ac_y : " << setw(15) << AyD2_RMS << endl;
    cout << "Second Derivative of Ac_z : " << setw(15) << AzD2_RMS << endl;
    cout << endl;
    cout << "Third Derivative of Ac_x  : " << setw(15) << AxD3_RMS << endl;
    cout << "Third Derivative of Ac_y  : " << setw(15) << AyD3_RMS << endl;
    cout << "Third Derivative of Ac_z  : " << setw(15) << AzD3_RMS << endl;
    cout << endl;
    cout << "Fourth Derivative of Ac_x : " << setw(15) << AxD4_RMS << endl;
    cout << "Fourth Derivative of Ac_y : " << setw(15) << AyD4_RMS << endl;
    cout << "Fourth Derivative of Ac_z : " << setw(15) << AzD4_RMS << endl;
    cout << endl;
  }

  delete[] AxD2;
  delete[] AyD2;
  delete[] AzD2;
  delete[] AxD3;
  delete[] AyD3;
  delete[] AzD3;
  delete[] AxD4;
  delete[] AyD4;
  delete[] AzD4;

  return;
}

void Coulomb (double* Ax,double* Ay,double* Az,char *argv[])
{
  //This function performs the gauge transformation on the
  //A field to put the field into the Coulomb gauge,
  //defined by div(A)=0.

  double *psi = new double [(Nx_fft+1)*(Ny_fft+1)*(Nz_fft+1)];
  double *kernel = new double [(Nx_fft2)*(Ny_fft2)*(Nz_fft2)];
  double *divA_fft = new double [(Nx_fft2)*(Ny_fft2)*(Nz_fft2)];
  fftw_complex *kernel_c = fftw_alloc_complex((Nx_fft2)*(Ny_fft2)*(Nz_fft+1));
  fftw_complex *divA_c = fftw_alloc_complex((Nx_fft2)*(Ny_fft2)*(Nz_fft+1));

  int Nx_ker, Ny_ker, Nz_ker;
  int idist, jdist, kdist, psii, psij, psik;
  int axindex, ayindex, azindex;
  int psi_index, kernel_index;
  int axlow, axhigh, aylow, ayhigh, azlow, azhigh;
  int diva_index, corner, indexlow, indexhigh;
  double rda, cda;

  FILE *in_kernel;

  FILE *outkeri, *outkerj, *outkerk, *outker;
  ostringstream keri, kerj, kerk, kerfull;
  keri << "kernel_i" << setfill('0') << setw(3) << cutxindex << ".dat";
  kerj << "kernel_j" << setfill('0') << setw(3) << cutyindex << ".dat";
  kerk << "kernel_k" << setfill('0') << setw(3) << cutzindex << ".dat";
  kerfull << "kernel.dat";

  fftw_plan forward_plan1, forward_plan2, reverse_plan;

  forward_plan1 = fftw_plan_dft_r2c_3d(Nx_fft2,Ny_fft2,Nz_fft2,kernel,kernel_c,FFTW_ESTIMATE);
  forward_plan2 = fftw_plan_dft_r2c_3d(Nx_fft2,Ny_fft2,Nz_fft2,divA_fft,divA_c,FFTW_ESTIMATE);
  reverse_plan = fftw_plan_dft_c2r_3d(Nx_fft2,Ny_fft2,Nz_fft2,divA_c,divA_fft,FFTW_ESTIMATE);
  
  cout << "About to do Coulomb piece!" << endl;

  //Build the convolution kernel.

  if (kernel_flag == 1)
  {
    //Construct 1/r kernel.

    kernel[0]=self_convolution/4.0/M_PI/step;

    for(i=0; i<Nx_fft2; i++) 
    {
      for(j=0; j<Ny_fft2; j++) 
      {
	for(k=0; k<Nz_fft2; k++) 
	{
	  kernel_index = k+Nz_fft2*(j+Ny_fft2*i);
	  idist = min(i,Nx_fft2-i);
	  jdist = min(j,Ny_fft2-j);
	  kdist = min(k,Nz_fft2-k);

	  if (i!=0 || j!=0 || k!=0)
	  {
	    kernel[kernel_index] = 1.0/4.0/M_PI/step/sqrt(idist*idist+jdist*jdist+kdist*kdist);
	  }
	}
      }
    }

    cout << "Constructed inv_r" << endl;
  }

  else if (kernel_flag == 2)
  {
    //Create the general kernel.
      
    in_kernel = fopen(argv[4], "r");

    fscanf(in_kernel, "%i %i %i\n", &Nx_ker, &Ny_ker, &Nz_ker);
    
    if (Nx_ker < (Nx_fft+1) || Ny_ker < (Ny_fft+1) || Nz_ker < (Nz_fft+1))
    {
      cout << endl;
      cout << "Under-Sized Kernel" << endl;
      return;
    }
    
    else if (Nx_ker == (Nx_fft+1) && Ny_ker == (Ny_fft+1) && Nz_ker == (Nz_fft+1))
    {
      cout << endl;
      cout << "Correctly Sized Kernel" << endl;
      cout << endl;

      for(k=0; k<Nz_fft+1; k++)
      {
	for(j=0; j<Ny_fft+1; j++)
	{
	  for(i=0; i<Nx_fft+1; i++)
	  {
	    psi_index = k+(Nz_fft+1)*(j+(Ny_fft+1)*i);	
	    fscanf(in_kernel, "%lf\n", &psi[psi_index]);
	  }
	}
      }

      fclose(in_kernel);
    }

    else if (Nx_ker > (Nx_fft+1) || Ny_ker > (Ny_fft+1) || Nz_ker > (Nz_fft+1))
    {
      cout << endl;
      cout << "Over-Sized Kernel" << endl;
      cout << endl;

      double *kernel_full = new double[(Nx_ker)*(Ny_ker)*(Nz_ker)];

      for(k=0; k<Nz_ker; k++)
      {
	for(j=0; j<Ny_ker; j++)
	{
	  for(i=0; i<Nx_ker; i++)
	  {
	    kernel_index = k+(Nz_ker)*(j+(Ny_ker)*i);
	    fscanf(in_kernel, "%lf\n", &kernel_full[kernel_index]);
	  }
	}
      }

      fclose(in_kernel);

      for(i=0; i<Nx_fft+1; i++)
      {
	for(j=0; j<Ny_fft+1; j++)
	{
	  for(k=0; k<Nz_fft+1; k++)
	  {
	    psi_index = k+(Nz_fft+1)*(j+(Ny_fft+1)*i);	
	    kernel_index = k+(Nz_ker)*(j+(Ny_ker)*i);
	    psi[psi_index] = kernel_full[kernel_index];
	  }
	}
      }

      delete[] kernel_full;
    }

    for(i=0; i<Nx_fft2; i++)
    {
      for(j=0; j<Ny_fft2; j++)
      {
	for(k=0; k<Nz_fft2; k++)
	{
	  psii = min(i,Nx_fft2-i);
	  psij = min(j,Ny_fft2-j);
	  psik = min(k,Nz_fft2-k);
	  psi_index = psik+(Nz_fft+1)*(psij+(Ny_fft+1)*psii);	
	  kernel_index = k+Nz_fft2*(j+Ny_fft2*i);
	  kernel[kernel_index] = psi[psi_index]/step;   //Divide by step for proper units
	}
      }
    }    

    //Print out the kernel

    outkeri = fopen(keri.str().c_str(), "w");
    for(j=0; j<Ny_fft+1; j++)
    {
      for(k=0; k<Nz_fft+1; k++)
      {  
	i = cutxindex;
	psi_index = k+(Nz_fft+1)*(j+(Ny_fft+1)*i);	
	fprintf(outkeri, "%i %i %i %e\n", 
		i, j, k, psi[psi_index]);
      }	
    }
    fclose(outkeri);

    outkerj = fopen(kerj.str().c_str(), "w");
    for(i=0; i<Nx_fft+1; i++)
    {
      for(k=0; k<Nz_fft+1; k++)
      {  
	j = cutyindex;
	psi_index = k+(Nz_fft+1)*(j+(Ny_fft+1)*i);	
	fprintf(outkerj, "%i %i %i %e\n", 
		i, j, k, psi[psi_index]);
      }	
    }
    fclose(outkerj);

    outkerk = fopen(kerk.str().c_str(), "w");
    for(i=0; i<Nx_fft+1; i++)
    {
      for(j=0; j<Ny_fft+1; j++)
      {
	k = cutzindex;
	psi_index = k+(Nz_fft+1)*(j+(Ny_fft+1)*i);	
	fprintf(outkerk, "%i %i %i %e\n", 
		i, j, k, psi[psi_index]);
      }	
    }
    fclose(outkerk);

    outker = fopen(kerfull.str().c_str(), "w");
    fprintf(outker, "%i %i %i\n", 
	    Nx_ker, Ny_ker, Nz_ker);
    for(i=0; i<Nx_fft+1; i++)
    {
      for(j=0; j<Ny_fft+1; j++)
      {
	for(k=0; k<Nz_fft+1; k++)
	{
	  psi_index = k+(Nz_fft+1)*(j+(Ny_fft+1)*i);	
	  fprintf(outker, "%.19lf\n", 
		  psi[psi_index]);
	}
      }	
    }
    fclose(outker);


    cout << endl;
    cout << "Laplacian of the kernel:" << endl;
    cout << endl;

    laplacian(Nx_fft+1,Ny_fft+1,Nz_fft+1,psi);

    cout << "Constructed kernel" << endl;
  }

  //Initialize div(A).

  for(i=0; i<Nx_fft2; i++)
  {
    for(j=0; j<Ny_fft2; j++)
    {
      for(k=0; k<Nz_fft2; k++) 
      {
	diva_index = k+(Nz_fft2)*(j+(Ny_fft2)*i);
	divA_fft[diva_index] = 0.0;
      }
    }
  }
  //divA_fft[0+Nz_fft2*(0+Ny_fft2*0)] = 1.0;

  for(l=0; l<ngauge; l++)
  {
    cout << "Iteration: " << l+1 << endl;

    //We need to calculate the divergence of the A-field
    //Allow for a one unit offset with respect to the 
    //interior of the grid, to encompass the boundary terms
    
    for(i=0; i<Nx-1; i++) 
    {
      for(j=0; j<Ny-1; j++)
      {
	for(k=0; k<Nz-1; k++) 
	{
	  //Recall: A vectors have N points in that (normal) direction, N+1 tangentially
	  // Shift by one in tangential directions for divA, average i and i+1 in normal direction 
	  axlow = (k+1)+(Nz+1)*((j+1)+(Ny+1)*i); 
	  axhigh = (k+1)+(Nz+1)*((j+1)+(Ny+1)*(i+1)); 
	  aylow = (k+1)+(Nz+1)*(j+(Ny)*(i+1));
	  ayhigh = (k+1)+(Nz+1)*((j+1)+(Ny)*(i+1));
	  azlow = k+(Nz)*((j+1)+(Ny+1)*(i+1));
	  azhigh = (k+1)+(Nz)*((j+1)+(Ny+1)*(i+1));

	  //This is if we FFT on the grid itself
	  //   diva_index = k+(N-1)*(j+(N_1)*i);
	  //This is if we use a grid of fixed size N_fft
	  //   diva_index = k+(N_fft)*(j+(N_fft)*i);
	  diva_index = (k+1)+(Nz_fft2)*((j+1)+(Ny_fft2)*(i+1));

	  divA_fft[diva_index] = ((Ax[axhigh]-Ax[axlow])/stepx
	                        + (Ay[ayhigh]-Ay[aylow])/stepy
				+ (Az[azhigh]-Az[azlow])/stepz)*dV;            
	}
      }
    }
    
    corner = 1+(Nz_fft2)*(1+(Ny_fft2)*1);

    cout << setiosflags(ios::fixed) << setprecision(4);
    cout << "divA: " << stepx << " " << stepy << " " << stepz << " ";
    cout << setiosflags(ios::fixed) << setprecision(15);
    cout << setw(20) << divA_fft[corner]
	 << setw(20) << divA_fft[corner+1]
	 << setw(20) << divA_fft[corner+256]
	 << setw(20) << divA_fft[corner+257]
	 << setw(20) << divA_fft[cutzindex+(Nz_fft2)*(cutyindex+(Ny_fft2)*cutxindex)] << endl;

    //Call the FFTW to solve nabla^2(phi) = div(A); the result below is actually -phi, not +phi
    //This phi is the gauge-fixing potential.

    fftw_execute(forward_plan1);
    fftw_execute(forward_plan2);

    for(i=0; i<Nx_fft2*Ny_fft2*(Nz_fft+1); i++)
    {
      rda = divA_c[i][0]*kernel_c[i][0] - divA_c[i][1]*kernel_c[i][1];
      cda = divA_c[i][0]*kernel_c[i][1] + divA_c[i][1]*kernel_c[i][0];

      divA_c[i][0] = rda;
      divA_c[i][1] = cda;
    }

    fftw_execute(reverse_plan);

    for(i=0; i<Nx_fft2*Ny_fft2*Nz_fft2; i++) 
    {
      divA_fft[i] *= 1.0/Nx_fft2/Ny_fft2/Nz_fft2;
    }

    //At this point, the array divA_fft actually contains -phi

    cout << setiosflags(ios::fixed) << setprecision(4);
    cout << "-phi: " << stepx << " " << stepy << " " << stepz << " ";
    cout << setiosflags(ios::fixed) << setprecision(15);
    cout << setw(20) << divA_fft[corner]
	 << setw(20) << divA_fft[corner+1]
	 << setw(20) << divA_fft[corner+256]
	 << setw(20) << divA_fft[corner+257]
	 << setw(20) << divA_fft[cutzindex+(Nz_fft2)*(cutyindex+(Ny_fft2)*cutxindex)] << endl;

    //Finally, we have A_c = A - grad(phi), which we evaluate as A + grad(-phi)

    for(i=0; i<Nx; i++) 
    {
      for(j=0; j<Ny+1; j++) 
      {
	for(k=0; k<Nz+1; k++) 
	{
	  axindex = k+(Nz+1)*(j+(Ny+1)*i);
	  indexlow = k+(Nz_fft2)*(j+(Ny_fft2)*i);
	  indexhigh = k+(Nz_fft2)*(j+(Ny_fft2)*(i+1));
        
	  Ax[axindex] += (divA_fft[indexhigh]-divA_fft[indexlow])/stepx;
	}
      }
    }

    for(i=0; i<Nx+1; i++) 
    {
      for(j=0; j<Ny; j++) 
      {
	for(k=0; k<Nz+1; k++) 
	{
	  ayindex = k+(Nz+1)*(j+(Ny)*i);
	  indexlow = k+(Nz_fft2)*(j+(Ny_fft2)*i);
	  indexhigh = k+(Nz_fft2)*((j+1)+(Ny_fft2)*i);
        
	  Ay[ayindex] += (divA_fft[indexhigh]-divA_fft[indexlow])/stepy;
	}
      }
    }

    for(i=0; i<Nx+1; i++) 
    {
      for(j=0; j<Ny+1; j++) 
      {
	for(k=0; k<Nz; k++) 
	{
	  azindex = k+(Nz)*(j+(Ny+1)*i);
	  indexlow = k+(Nz_fft2)*(j+(Ny_fft2)*i);
	  indexhigh = (k+1)+(Nz_fft2)*(j+(Ny_fft2)*i);
        
	  Az[azindex] += (divA_fft[indexhigh]-divA_fft[indexlow])/stepz;
	}
      }
    }

    //This is a check, because we just calculated div(A) again.

    for(i=0; i<Nx-1; i++) 
    {
      for(j=0; j<Ny-1; j++)
      {
	for(k=0; k<Nz-1; k++) 
	{
	  //Recall: A vectors have N points in that (normal) direction, N+1 tangentially
	  //Shift by one in tangential directions for divA, average i and i+1 in normal direction 
	  axlow = (k+1)+(Nz+1)*((j+1)+(Ny+1)*i); 
	  axhigh = (k+1)+(Nz+1)*((j+1)+(Ny+1)*(i+1)); 
	  aylow = (k+1)+(Nz+1)*(j+(Ny)*(i+1));
	  ayhigh = (k+1)+(Nz+1)*((j+1)+(Ny)*(i+1));
	  azlow = k+(Nz)*((j+1)+(Ny+1)*(i+1));
	  azhigh = (k+1)+(Nz)*((j+1)+(Ny+1)*(i+1));

	  //This is if we FFT on the grid itself
	  //   int diva_index = k+(N-1)*(j+(N_1)*i);
	  //This is if we use a grid of fixed size N_fft
	  //   int diva_index = k+(N_fft)*(j+(N_fft)*i);
	  diva_index = (k+1)+(Nz_fft2)*((j+1)+(Ny_fft2)*(i+1));

	  divA_fft[diva_index] = (Ax[axhigh]-Ax[axlow])/stepx
	                       + (Ay[ayhigh]-Ay[aylow])/stepy
	                       + (Az[azhigh]-Az[azlow])/stepz;          
	}
      }
    }
  
    cout << setiosflags(ios::fixed) << setprecision(4);
    cout << "divA: " << stepx << " " << stepy << " " << stepz << " ";
    cout << setiosflags(ios::fixed) << setprecision(15);
    cout << setw(20) << divA_fft[corner]
	 << setw(20) << divA_fft[corner+1]
	 << setw(20) << divA_fft[corner+256]
	 << setw(20) << divA_fft[corner+257]
	 << setw(20) << divA_fft[cutzindex+(Nz_fft2)*(cutyindex+(Ny_fft2)*cutxindex)] << endl;

    cout<<"End of Coulomb piece!"<<endl;
    cout << endl << endl;
  }

  delete[] psi;
  delete[] kernel;
  delete[] divA_fft;
  //fftw_free(kernel_c);
  //fftw_free(divA_c);
  fftw_destroy_plan(forward_plan1);
  fftw_destroy_plan(forward_plan2);
  fftw_destroy_plan(reverse_plan);

  return;
}

void extrap_tests (double* Ax_ext,double* Ay_ext,double* Az_ext,int flag)
{
  //Extrapolation Tests

  int n_ghost = 3;
  int n_ext = 0;
  int ext_index, index1, index2, index3, index4;
  int ext2_index, ext3_index;

  int ext_order = 2;
  int ext_flag = 4;

  int axindex, ayindex, azindex;
  double outx, outy, outz;

  //Larger, linearly extrapolated fields
  double *Ax_ext2 = new double[(Nx+1+n_ext)*(Ny+1)*(Nz+1)];
  double *Ay_ext2 = new double[(Nx+1+n_ext)*(Ny+1)*(Nz+1)];
  double *Az_ext2 = new double[(Nx+1+n_ext)*(Ny+1)*(Nz+1)];

  //Staggered, linearly extrapolated fields
  double *Ax_ext3 = new double[(Nx+n_ext)*(Ny+1)*(Nz+1)];
  double *Ay_ext3 = new double[(Nx+1+n_ext)*(Ny)*(Nz+1)];
  double *Az_ext3 = new double[(Nx+1+n_ext)*(Ny+1)*(Nz)];

  //Staggered curl of staggered, extrapolated fields
  double *Bx_ext3 = new double[(Nx+1+n_ext)*(Ny)*(Nz)];
  double *By_ext3 = new double[(Nx+n_ext)*(Ny+1)*(Nz)];
  double *Bz_ext3 = new double[(Nx+n_ext)*(Ny)*(Nz+1)];

  FILE *outBxExtest, *outByExtest, *outBzExtest;
  FILE *outAxExtest, *outAyExtest, *outAzExtest;
  FILE *outBxi, *outBxj, *outBxk; 
  FILE *outByi, *outByj, *outByk;
  FILE *outBzi, *outBzj, *outBzk;
  FILE *outAxi, *outAxj, *outAxk;
  FILE *outAyi, *outAyj, *outAyk;
  FILE *outAzi, *outAzj, *outAzk;

  ostringstream BxExtest, ByExtest, BzExtest;
  ostringstream AxExtest, AyExtest, AzExtest;
  ostringstream Bxi, Bxj, Bxk;
  ostringstream Byi, Byj, Byk;
  ostringstream Bzi, Bzj, Bzk;
  ostringstream Axi, Axj, Axk;
  ostringstream Ayi, Ayj, Ayk;
  ostringstream Azi, Azj, Azk;

  if (flag == 1)
  {
    BxExtest << "Bx0_extrap.dat";
    ByExtest << "By0_extrap.dat";
    BzExtest << "Bz0_extrap.dat";
    AxExtest << "Ax0_extrap.dat";
    AyExtest << "Ay0_extrap.dat";
    AzExtest << "Az0_extrap.dat";
    Bxi << "Bx0.x.dat";
    Bxj << "Bx0.y.dat";
    Bxk << "Bx0.z.dat";
    Byi << "By0.x.dat";
    Byj << "By0.y.dat";
    Byk << "By0.z.dat";
    Bzi << "Bz0.x.dat";
    Bzj << "Bz0.y.dat";
    Bzk << "Bz0.z.dat";
    Axi << "Ax0.x.dat";
    Axj << "Ax0.y.dat";
    Axk << "Ax0.z.dat";
    Ayi << "Ay0.x.dat";
    Ayj << "Ay0.y.dat";
    Ayk << "Ay0.z.dat";
    Azi << "Az0.x.dat";
    Azj << "Az0.y.dat";
    Azk << "Az0.z.dat";
  }
  else if (flag == 2)
  {
    BxExtest << "Bx_extrap.dat";
    ByExtest << "By_extrap.dat";
    BzExtest << "Bz_extrap.dat";
    AxExtest << "Ax_extrap.dat";
    AyExtest << "Ay_extrap.dat";
    AzExtest << "Az_extrap.dat";
    Bxi << "Bx.x.dat";
    Bxj << "Bx.y.dat";
    Bxk << "Bx.z.dat";
    Byi << "By.x.dat";
    Byj << "By.y.dat";
    Byk << "By.z.dat";
    Bzi << "Bz.x.dat";
    Bzj << "Bz.y.dat";
    Bzk << "Bz.z.dat";
    Axi << "Ax.x.dat";
    Axj << "Ax.y.dat";
    Axk << "Ax.z.dat";
    Ayi << "Ay.x.dat";
    Ayj << "Ay.y.dat";
    Ayk << "Ay.z.dat";
    Azi << "Az.x.dat";
    Azj << "Az.y.dat";
    Azk << "Az.z.dat";
  }

  //First initialize the bigger arrays in
  //preparation for the necessary
  //extrapolations, staggerings, etc.

  for(i=0; i<Nx+1+n_ext; i++)
  {  
    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	Ax_ext2[ext_index] = poison;
	Ay_ext2[ext_index] = poison;
	Az_ext2[ext_index] = poison;
      }
    }
  }

  for(i=0; i<Nx+n_ext; i++)
  {  
    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	Ax_ext3[ext_index] = poison;
      }
    }
  }
  for(i=0; i<Nx+1+n_ext; i++)
  {  
    for(j=0; j<Ny; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	ext_index = k+(Nz+1)*(j+(Ny)*i);
	Ay_ext3[ext_index] = poison;
      }
    }
  }
  for(i=0; i<Nx+1+n_ext; i++)
  {  
    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz; k++)
      {
	ext_index = k+(Nz)*(j+(Ny+1)*i);
	Az_ext3[ext_index] = poison;
      }
    }
  }

  for(i=0; i<Nx+1+n_ext; i++)
  {  
    for(j=0; j<Ny; j++)
    {
      for(k=0; k<Nz; k++)
      {
	ext_index = k+(Nz)*(j+(Ny)*i);
	Bx_ext3[ext_index] = poison;
      }
    }
  }
  for(i=0; i<Nx+n_ext; i++)
  {  
    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz; k++)
      {
	ext_index = k+(Nz)*(j+(Ny+1)*i);
	By_ext3[ext_index] = poison;
      }
    }
  }
  for(i=0; i<Nx+n_ext; i++)
  {  
    for(j=0; j<Ny; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	ext_index = k+(Nz+1)*(j+(Ny)*i);
	Bz_ext3[ext_index] = poison;
      }
    }
  }

  //Populate the bigger arrays in
  //preparation for the necessary
  //extrapolations

  for(i=0; i<Nx+1; i++)
  {  
    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	axindex = k+(Nz+1)*(j+(Ny+1)*i);
	ext_index = k+(Nz+1)*(j+(Ny+1)*(i+n_ext));
	Ax_ext2[ext_index] = Ax_ext[axindex];
	Ay_ext2[ext_index] = Ay_ext[axindex];
	Az_ext2[ext_index] = Az_ext[axindex];
      }
    }
  }

  //After reading in the A fields, IllinoisGRMHD
  //extrapolates the ghost zones. Thus this is 
  //done here as well.

  if (ext_flag == 1)
  {
    //Lower x-Face

    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	i = n_ext+(n_ghost);
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	index1 = k+(Nz+1)*(j+(Ny+1)*(i+1));
	index2 = k+(Nz+1)*(j+(Ny+1)*(i+2));
	Ax_ext2[ext_index] = 2*Ax_ext2[index1] 
	                   - 1*Ax_ext2[index2];

	for(l=0; l<n_ghost; l++)
	{
	  i = n_ext+(n_ghost)-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*(j+(Ny+1)*(i+1));
	  index2 = k+(Nz+1)*(j+(Ny+1)*(i+2));
	  index3 = k+(Nz+1)*(j+(Ny+1)*(i+3));

	  Ax_ext2[ext_index] = 2*Ax_ext2[index1] 
	                     - 1*Ax_ext2[index2];

	  if (ext_order == 1)
	  {
	    Ay_ext2[ext_index] = 2*Ay_ext2[index1] 
	                       - 1*Ay_ext2[index2];

	    Az_ext2[ext_index] = 2*Az_ext2[index1] 
	                       - 1*Az_ext2[index2];
	  }
	  else if (ext_order == 2)
	  {
	    Ay_ext2[ext_index] = 3*Ay_ext2[index1] 
	                       - 3*Ay_ext2[index2]
	                       + 1*Ay_ext2[index3];

	    Az_ext2[ext_index] = 3*Az_ext2[index1] 
	                       - 3*Az_ext2[index2]
	                       + 1*Az_ext2[index3];
	  }
	}
      }
    }
  
    //Upper x-Face

    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  i = (Nx+1+n_ext)-(n_ghost)+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*(j+(Ny+1)*(i-1));
	  index2 = k+(Nz+1)*(j+(Ny+1)*(i-2));
	  index3 = k+(Nz+1)*(j+(Ny+1)*(i-3));

	  Ax_ext2[ext_index] = 2*Ax_ext2[index1] 
	                     - 1*Ax_ext2[index2];

	  if (ext_order == 1)
	  {
	    Ay_ext2[ext_index] = 2*Ay_ext2[index1] 
	                       - 1*Ay_ext2[index2];

	    Az_ext2[ext_index] = 2*Az_ext2[index1] 
	                       - 1*Az_ext2[index2];
	  }
	  else if (ext_order == 2)
	  {
	    Ay_ext2[ext_index] = 3*Ay_ext2[index1] 
	                       - 3*Ay_ext2[index2]
	                       + 1*Ay_ext2[index3];

	    Az_ext2[ext_index] = 3*Az_ext2[index1] 
	                       - 3*Az_ext2[index2]
		               + 1*Az_ext2[index3];
	  }
	}
      }
    }
  
    //Lower y-Face

    for(i=0; i<Nx+1+n_ext; i++)
    {
      for(k=0; k<Nz+1; k++)
      {
	j = n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	index1 = k+(Nz+1)*((j+1)+(Ny+1)*i);
	index2 = k+(Nz+1)*((j+2)+(Ny+1)*i);
	Ay_ext2[ext_index] = 2*Ay_ext2[index1] 
	                   - 1*Ay_ext2[index2];

	for(l=0; l<n_ghost; l++)
	{
	  j = n_ghost-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*((j+1)+(Ny+1)*i);
	  index2 = k+(Nz+1)*((j+2)+(Ny+1)*i);
	  index3 = k+(Nz+1)*((j+3)+(Ny+1)*i);

	  Ay_ext2[ext_index] = 2*Ay_ext2[index1] 
	                     - 1*Ay_ext2[index2];

	  if (ext_order == 1)
	  {
	    Ax_ext2[ext_index] = 2*Ax_ext2[index1] 
	                       - 1*Ax_ext2[index2];

	    Az_ext2[ext_index] = 2*Az_ext2[index1] 
	                       - 1*Az_ext2[index2];
	  }
	  else if (ext_order == 2)
	  {
	    Ax_ext2[ext_index] = 3*Ax_ext2[index1] 
	                       - 3*Ax_ext2[index2]
		               + 1*Ax_ext2[index3];

	    Az_ext2[ext_index] = 3*Az_ext2[index1] 
		               - 3*Az_ext2[index2]
		               + 1*Az_ext2[index3];
	  }
	}
      }
    }
  
    //Upper y-Face

    for(i=0; i<Nx+1+n_ext; i++)
    {
      for(k=0; k<Nz+1; k++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  j = (Ny+1)-(n_ghost)+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*((j-1)+(Ny+1)*i);
	  index2 = k+(Nz+1)*((j-2)+(Ny+1)*i);
	  index3 = k+(Nz+1)*((j-3)+(Ny+1)*i);

	  Ay_ext2[ext_index] = 2*Ay_ext2[index1] 
	                     - 1*Ay_ext2[index2];

	  if (ext_order == 1)
	  {
	    Ax_ext2[ext_index] = 2*Ax_ext2[index1] 
	                       - 1*Ax_ext2[index2];

	    Az_ext2[ext_index] = 2*Az_ext2[index1] 
	                       - 1*Az_ext2[index2];
	  }
	  else if (ext_order == 2)
	  {
	    Ax_ext2[ext_index] = 3*Ax_ext2[index1] 
	                       - 3*Ax_ext2[index2]
	                       + 1*Ax_ext2[index3];

	    Az_ext2[ext_index] = 3*Az_ext2[index1] 
	                       - 3*Az_ext2[index2]
		               + 1*Az_ext2[index3];
	  }
	}
      }
    }
  
    //Lower z-Face

    for(i=0; i<Nx+1+n_ext; i++)
    {
      for(j=0; j<Ny+1; j++)
      {
	k = n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	index1 = (k+1)+(Nz+1)*(j+(Ny+1)*i);
	index2 = (k+2)+(Nz+1)*(j+(Ny+1)*i);
	Az_ext2[ext_index] = 2*Az_ext2[index1] 
	                   - 1*Az_ext2[index2];

	for(l=0; l<n_ghost; l++)
	{
	  k = n_ghost-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = (k+1)+(Nz+1)*(j+(Ny+1)*i);
	  index2 = (k+2)+(Nz+1)*(j+(Ny+1)*i);
	  index3 = (k+3)+(Nz+1)*(j+(Ny+1)*i);

	  Az_ext2[ext_index] = 2*Az_ext2[index1] 
	                     - 1*Az_ext2[index2];

	  if (ext_order == 1)
	  {
	    Ax_ext2[ext_index] = 2*Ax_ext2[index1] 
	                       - 1*Ax_ext2[index2];

	    Ay_ext2[ext_index] = 2*Ay_ext2[index1] 
	                       - 1*Ay_ext2[index2];
	  }
	  else if (ext_order == 2)
	  {
	    Ax_ext2[ext_index] = 3*Ax_ext2[index1] 
	                       - 3*Ax_ext2[index2]
	                       + 1*Ax_ext2[index3];

	    Ay_ext2[ext_index] = 3*Ay_ext2[index1] 
	                       - 3*Ay_ext2[index2]
	                       + 1*Ay_ext2[index3];
	  }
	}
      }
    }
    
    //Upper z-Face

    for(i=0; i<Nx+1+n_ext; i++)
    {
      for(j=0; j<Ny+1; j++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  k = (Nz+1)-(n_ghost)+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = (k-1)+(Nz+1)*(j+(Ny+1)*i);
	  index2 = (k-2)+(Nz+1)*(j+(Ny+1)*i);
	  index3 = (k-3)+(Nz+1)*(j+(Ny+1)*i);

	  Az_ext2[ext_index] = 2*Az_ext2[index1] 
	                     - 1*Az_ext2[index2];

	  if (ext_order == 1)
	  {
	    Ax_ext2[ext_index] = 2*Ax_ext2[index1] 
	                       - 1*Ax_ext2[index2];

	    Ay_ext2[ext_index] = 2*Ay_ext2[index1] 
		               - 1*Ay_ext2[index2];
	  }
	  else if (ext_order == 2)
	  {
	    Ax_ext2[ext_index] = 3*Ax_ext2[index1] 
	                       - 3*Ax_ext2[index2]
	                       + 1*Ax_ext2[index3];

	    Ay_ext2[ext_index] = 3*Ay_ext2[index1] 
	                       - 3*Ay_ext2[index2]
	                       + 1*Ay_ext2[index3];
	  }
	}
      }
    }
  }
  
  else if (ext_flag == 2)
  {
    //Lower x-Face

    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	i = n_ext+(n_ghost);
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	index1 = k+(Nz+1)*(j+(Ny+1)*(i+1));
	Ax_ext2[ext_index] = Ax_ext2[index1];

	for(l=0; l<n_ghost; l++)
	{
	  i = n_ext+(n_ghost)-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*(j+(Ny+1)*(i+1));
	  index2 = k+(Nz+1)*(j+(Ny+1)*(i+2));

	  Ax_ext2[ext_index] = Ax_ext2[index1];
	  
	  Ay_ext2[ext_index] = 2*Ay_ext2[index1] 
	                     - 1*Ay_ext2[index2];

	  Az_ext2[ext_index] = 2*Az_ext2[index1] 
	                     - 1*Az_ext2[index2];
	  
	  /*
	  Ay_ext2[ext_index] = Ay_ext2[index1];


	  Az_ext2[ext_index] = Az_ext2[index1];
	  */
	}
      }
    }
  
    //Upper x-Face

    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  i = (Nx+1+n_ext)-(n_ghost)+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*(j+(Ny+1)*(i-1));
	  index2 = k+(Nz+1)*(j+(Ny+1)*(i-2));

	  Ax_ext2[ext_index] = Ax_ext2[index1];

	  Ay_ext2[ext_index] = 2*Ay_ext2[index1] 
	                     - 1*Ay_ext2[index2];

	  Az_ext2[ext_index] = 2*Az_ext2[index1] 
	                     - 1*Az_ext2[index2];
	}
      }
    }
  
    //Lower y-Face

    for(i=0; i<Nx+1+n_ext; i++)
    {
      for(k=0; k<Nz+1; k++)
      {
	j = n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	index1 = k+(Nz+1)*((j+1)+(Ny+1)*i);
	Ay_ext2[ext_index] = Ay_ext2[index1];

	for(l=0; l<n_ghost; l++)
	{
	  j = n_ghost-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*((j+1)+(Ny+1)*i);
	  index2 = k+(Nz+1)*((j+2)+(Ny+1)*i);

	  Ay_ext2[ext_index] = Ay_ext2[index1];

	  Ax_ext2[ext_index] = 2*Ax_ext2[index1] 
	                     - 1*Ax_ext2[index2];

	  Az_ext2[ext_index] = 2*Az_ext2[index1] 
	                     - 1*Az_ext2[index2];
	}
      }
    }
  
    //Upper y-Face

    for(i=0; i<Nx+1+n_ext; i++)
    {
      for(k=0; k<Nz+1; k++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  j = (Ny+1)-(n_ghost)+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*((j-1)+(Ny+1)*i);
	  index2 = k+(Nz+1)*((j-2)+(Ny+1)*i);

	  Ay_ext2[ext_index] = Ay_ext2[index1];

	  Ax_ext2[ext_index] = 2*Ax_ext2[index1] 
	                     - 1*Ax_ext2[index2];

	  Az_ext2[ext_index] = 2*Az_ext2[index1] 
	                     - 1*Az_ext2[index2];
	}
      }
    }
  
    //Lower z-Face

    for(i=0; i<Nx+1+n_ext; i++)
    {
      for(j=0; j<Ny+1; j++)
      {
	k = n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	index1 = (k+1)+(Nz+1)*(j+(Ny+1)*i);
	Az_ext2[ext_index] = Az_ext2[index1];

	for(l=0; l<n_ghost; l++)
	{
	  k = n_ghost-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = (k+1)+(Nz+1)*(j+(Ny+1)*i);
	  index2 = (k+2)+(Nz+1)*(j+(Ny+1)*i);

	  Az_ext2[ext_index] = Az_ext2[index1];

	  Ax_ext2[ext_index] = 2*Ax_ext2[index1] 
	                     - 1*Ax_ext2[index2];

	  Ay_ext2[ext_index] = 2*Ay_ext2[index1] 
	                     - 1*Ay_ext2[index2];
	}
      }
    }
  
    //Upper z-Face

    for(i=0; i<Nx+1+n_ext; i++)
    {
      for(j=0; j<Ny+1; j++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  k = (Nz+1)-(n_ghost)+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = (k-1)+(Nz+1)*(j+(Ny+1)*i);
	  index2 = (k-2)+(Nz+1)*(j+(Ny+1)*i);

	  Az_ext2[ext_index] = Az_ext2[index1];

	  Ax_ext2[ext_index] = 2*Ax_ext2[index1] 
	                     - 1*Ax_ext2[index2];

	  Ay_ext2[ext_index] = 2*Ay_ext2[index1] 
	                     - 1*Ay_ext2[index2];
	}
      }
    }
  }
  
  else if (ext_flag == 3)
  {
    //Lower x-Face

    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	i = n_ext+n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	Ax_ext2[ext_index] = 0.0;

	for(l=0; l<n_ghost; l++)
	{
	  i = n_ext+(n_ghost)-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*(j+(Ny+1)*(i+1));

	  Ax_ext2[ext_index] = 0.0;
	  
	  Ay_ext2[ext_index] = Ay_ext2[index1];

	  Az_ext2[ext_index] = Az_ext2[index1];
	}
      }
    }

    //Upper x-Face

    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  i = (Nx+1+n_ext)-(n_ghost)+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*(j+(Ny+1)*(i-1));

	  Ax_ext2[ext_index] = 0.0;

	  Ay_ext2[ext_index] = Ay_ext2[index1];

	  Az_ext2[ext_index] = Az_ext2[index1];
	}
      }
    }
  
    //Lower y-Face

    for(i=0; i<Nx+1+n_ext; i++)
    {
      for(k=0; k<Nz+1; k++)
      {
	j = n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	Ay_ext2[ext_index] = 0.0;

	for(l=0; l<n_ghost; l++)
	{
	  j = n_ghost-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*((j+1)+(Ny+1)*i);

	  Ay_ext2[ext_index] = 0.0;

	  Ax_ext2[ext_index] = Ax_ext2[index1];

	  Az_ext2[ext_index] = Az_ext2[index1];
	}
      }
    }
  
    //Upper y-Face

    for(i=0; i<Nx+1+n_ext; i++)
    {
      for(k=0; k<Nz+1; k++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  j = (Ny+1)-(n_ghost)+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*((j-1)+(Ny+1)*i);

	  Ay_ext2[ext_index] = 0.0;

	  Ax_ext2[ext_index] = Ax_ext2[index1];

	  Az_ext2[ext_index] = Az_ext2[index1];
	}
      }
    }
  
    //Lower z-Face

    for(i=0; i<Nx+1+n_ext; i++)
    {
      for(j=0; j<Ny+1; j++)
      {
	k = n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	Az_ext2[ext_index] = 0.0;

	for(l=0; l<n_ghost; l++)
	{
	  k = n_ghost-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = (k+1)+(Nz+1)*(j+(Ny+1)*i);

	  Az_ext2[ext_index] = 0.0;

	  Ax_ext2[ext_index] = Ax_ext2[index1];

	  Ay_ext2[ext_index] = Ay_ext2[index1];
	}
      }
    }
  
    //Upper z-Face

    for(i=0; i<Nx+1+n_ext; i++)
    {
      for(j=0; j<Ny+1; j++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  k = (Nz+1)-(n_ghost)+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = (k-1)+(Nz+1)*(j+(Ny+1)*i);

	  Az_ext2[ext_index] = 0.0;

	  Ax_ext2[ext_index] = Ax_ext2[index1];

	  Ay_ext2[ext_index] = Ay_ext2[index1];
	}
      }
    }
  }
  
  else if (ext_flag == 4)
  {
    //Lower x-Face

    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	i = n_ext+n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	index1 = k+(Nz+1)*(j+(Ny+1)*(i+1));
	index2 = k+(Nz+1)*(j+(Ny+1)*(i+2));	
	index3 = k+(Nz+1)*(j+(Ny+1)*(i+3));
	Ax_ext2[ext_index] = 3*Ax_ext2[index1] 
	                   - 3*Ax_ext2[index2]
	                   + 1*Ax_ext2[index3];

	for(l=0; l<n_ghost; l++)
	{
	  i = n_ext+n_ghost-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*(j+(Ny+1)*(i+1));
	  index2 = k+(Nz+1)*(j+(Ny+1)*(i+2));
	  index3 = k+(Nz+1)*(j+(Ny+1)*(i+3));
	  index4 = k+(Nz+1)*(j+(Ny+1)*(i+4));

	  Ax_ext2[ext_index] = 3*Ax_ext2[index1] 
	                     - 3*Ax_ext2[index2]
	                     + 1*Ax_ext2[index3];

	  Ay_ext2[ext_index] = 4*Ay_ext2[index1] 
	                     - 6*Ay_ext2[index2]
	                     + 4*Ay_ext2[index3]
	                     - 1*Ay_ext2[index4];

	  Az_ext2[ext_index] = 4*Az_ext2[index1] 
	                     - 6*Az_ext2[index2]
	                     + 4*Az_ext2[index3]
	                     - 1*Az_ext2[index4];
	}
      }
    }
  
    //Upper x-Face

    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  i = (Nx+1+n_ext)-n_ghost+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*(j+(Ny+1)*(i-1));
	  index2 = k+(Nz+1)*(j+(Ny+1)*(i-2));
	  index3 = k+(Nz+1)*(j+(Ny+1)*(i-3));
	  index4 = k+(Nz+1)*(j+(Ny+1)*(i-4));

	  Ax_ext2[ext_index] = 3*Ax_ext2[index1] 
	                     - 3*Ax_ext2[index2]
	                     + 1*Ax_ext2[index3];

	  Ay_ext2[ext_index] = 4*Ay_ext2[index1] 
	                     - 6*Ay_ext2[index2]
	                     + 4*Ay_ext2[index3]
	                     - 1*Ay_ext2[index4];

	  Az_ext2[ext_index] = 4*Az_ext2[index1] 
	                     - 6*Az_ext2[index2]
	                     + 4*Az_ext2[index3]
	                     - 1*Az_ext2[index4];
	}
      }
    }
  
    //Lower y-Face

    for(i=0; i<Nx+1+n_ext; i++)
    {
      for(k=0; k<Nz+1; k++)
      {
	j = n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	index1 = k+(Nz+1)*((j+1)+(Ny+1)*i);
	index2 = k+(Nz+1)*((j+2)+(Ny+1)*i);
	index3 = k+(Nz+1)*((j+3)+(Ny+1)*i);
	Ay_ext2[ext_index] = 3*Ay_ext2[index1] 
	                   - 3*Ay_ext2[index2]
	                   + 1*Ay_ext2[index3];

	for(l=0; l<n_ghost; l++)
	{
	  j = n_ghost-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*((j+1)+(Ny+1)*i);
	  index2 = k+(Nz+1)*((j+2)+(Ny+1)*i);
	  index3 = k+(Nz+1)*((j+3)+(Ny+1)*i);
	  index4 = k+(Nz+1)*((j+4)+(Ny+1)*i);

	  Ay_ext2[ext_index] = 3*Ay_ext2[index1] 
	                     - 3*Ay_ext2[index2]
	                     + 1*Ay_ext2[index3];

	  Az_ext2[ext_index] = 4*Az_ext2[index1] 
	                     - 6*Az_ext2[index2]
	                     + 4*Az_ext2[index3]
	                     - 1*Az_ext2[index4];

	  Ax_ext2[ext_index] = 4*Ax_ext2[index1] 
	                     - 6*Ax_ext2[index2]
	                     + 4*Ax_ext2[index3]
	                     - 1*Ax_ext2[index4];
	}
      }
    }
  
    //Upper y-Face

    for(i=0; i<Nx+1+n_ext; i++)
    {
      for(k=0; k<Nz+1; k++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  j = (Ny+1)-n_ghost+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*((j-1)+(Ny+1)*i);
	  index2 = k+(Nz+1)*((j-2)+(Ny+1)*i);
	  index3 = k+(Nz+1)*((j-3)+(Ny+1)*i);
	  index4 = k+(Nz+1)*((j-4)+(Ny+1)*i);

	  Ay_ext2[ext_index] = 3*Ay_ext2[index1] 
	                     - 3*Ay_ext2[index2]
	                     + 1*Ay_ext2[index3];

	  Az_ext2[ext_index] = 4*Az_ext2[index1] 
	                     - 6*Az_ext2[index2]
	                     + 4*Az_ext2[index3]
	                     - 1*Az_ext2[index4];

	  Ax_ext2[ext_index] = 4*Ax_ext2[index1] 
	                     - 6*Ax_ext2[index2]
	                     + 4*Ax_ext2[index3]
	                     - 1*Ax_ext2[index4];
	}
      }
    }
  
    //Lower z-Face

    for(i=0; i<Nx+1+n_ext; i++)
    {
      for(j=0; j<Ny+1; j++)
      {
	k = n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	index1 = (k+1)+(Nz+1)*(j+(Ny+1)*i);
	index2 = (k+2)+(Nz+1)*(j+(Ny+1)*i);
	index3 = (k+3)+(Nz+1)*(j+(Ny+1)*i);
	Az_ext2[ext_index] = 3*Az_ext2[index1] 
	                   - 3*Az_ext2[index2]
	                   + 1*Az_ext2[index3];

	for(l=0; l<n_ghost; l++)
	{
	  k = n_ghost-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = (k+1)+(Nz+1)*(j+(Ny+1)*i);
	  index2 = (k+2)+(Nz+1)*(j+(Ny+1)*i);
	  index3 = (k+3)+(Nz+1)*(j+(Ny+1)*i);
	  index4 = (k+4)+(Nz+1)*(j+(Ny+1)*i);

	  Az_ext2[ext_index] = 3*Az_ext2[index1] 
	                     - 3*Az_ext2[index2]
	                     + 1*Az_ext2[index3];

	  Ax_ext2[ext_index] = 4*Ax_ext2[index1] 
	                     - 6*Ax_ext2[index2]
	                     + 4*Ax_ext2[index3]
	                     - 1*Ax_ext2[index4];

	  Ay_ext2[ext_index] = 4*Ay_ext2[index1] 
	                     - 6*Ay_ext2[index2]
	                     + 4*Ay_ext2[index3]
	                     - 1*Ay_ext2[index4];
	}
      }
    }
    
    //Upper z-Face

    for(i=0; i<Nx+1+n_ext; i++)
    {
      for(j=0; j<Ny+1; j++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  k = (Nz+1)-(n_ghost)+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = (k-1)+(Nz+1)*(j+(Ny+1)*i);
	  index2 = (k-2)+(Nz+1)*(j+(Ny+1)*i);
	  index3 = (k-3)+(Nz+1)*(j+(Ny+1)*i);
	  index4 = (k-4)+(Nz+1)*(j+(Ny+1)*i);

	  Az_ext2[ext_index] = 3*Az_ext2[index1] 
	                     - 3*Az_ext2[index2]
	                     + 1*Az_ext2[index3];

	  Ax_ext2[ext_index] = 4*Ax_ext2[index1] 
	                     - 6*Ax_ext2[index2]
	                     + 4*Ax_ext2[index3]
	                     - 1*Ax_ext2[index4];

	  Ay_ext2[ext_index] = 4*Ay_ext2[index1] 
	                     - 6*Ay_ext2[index2]
	                     + 4*Ay_ext2[index3]
	                     - 1*Ay_ext2[index4];
	}
      }
    }
  }
  
  //Now extrapolate.

  for(l=0; l<n_ext; l++)
  {
    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	i = (n_ext-1)-l;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	index1 = k+(Nz+1)*(j+(Ny+1)*(i+1));
	index2 = k+(Nz+1)*(j+(Ny+1)*(i+2));
	index3 = k+(Nz+1)*(j+(Ny+1)*(i+3));

	if (ext_order == 1)
	{
	  Ax_ext2[ext_index] = 2*Ax_ext2[index1] 
	                     - 1*Ax_ext2[index2];
	  Ay_ext2[ext_index] = 2*Ay_ext2[index1] 
	                     - 1*Ay_ext2[index2];
	  Az_ext2[ext_index] = 2*Az_ext2[index1] 
	                     - 1*Az_ext2[index2];
	}
	else if (ext_order == 2)
	{
	  Ax_ext2[ext_index] = 3*Ax_ext2[index1] 
	                     - 3*Ax_ext2[index2]
	                     + 1*Ax_ext2[index3];
	  Ay_ext2[ext_index] = 3*Ay_ext2[index1] 
	                     - 3*Ay_ext2[index2]
	                     + 1*Ay_ext2[index3];
	  Az_ext2[ext_index] = 3*Az_ext2[index1] 
	                     - 3*Az_ext2[index2]
	                     + 1*Az_ext2[index3];
	}
      }
    }
  }

  //Make the extrapolated fields smaller, making
  //them staggered so the curl can be performed.

  for(i=0; i<Nx+n_ext; i++)
  {  
    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	ext2_index = k+(Nz+1)*(j+(Ny+1)*(i+1));
	ext3_index = k+(Nz+1)*(j+(Ny+1)*i);
	Ax_ext3[ext3_index] = Ax_ext2[ext2_index];
      }
    }
  }

  for(i=0; i<Nx+1+n_ext; i++)
  {  
    for(j=0; j<Ny; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	ext2_index = k+(Nz+1)*((j+1)+(Ny+1)*i);
	ext3_index = k+(Nz+1)*(j+(Ny)*i);
	Ay_ext3[ext3_index] = Ay_ext2[ext2_index];
      }
    }
  }

  for(i=0; i<Nx+1+n_ext; i++)
  {  
    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz; k++)
      {
	ext2_index = (k+1)+(Nz+1)*(j+(Ny+1)*i);
	ext3_index = k+(Nz)*(j+(Ny+1)*i);
	Az_ext3[ext3_index] = Az_ext2[ext2_index];
      }
    }
  }

  //Now that the A fields are staggered, take
  //the curl to get the B fields.

  curl(Nx+n_ext,Ny,Nz,Ax_ext3,Ay_ext3,Az_ext3,Bx_ext3,By_ext3,Bz_ext3);

  //Output the extrapolated fields for testing.

  outBxExtest = fopen(BxExtest.str().c_str(), "w");
  for(k=0; k<Nz; k++)
  {
    for(j=0; j<Ny; j++)
    {
      for(i=0; i<Nx+1+n_ext; i++)
      {  
	outx = xinit + i*dx - (n_ext-1)*dx;
	outy = yinit + j*dy;
	outz = zinit + k*dz;
	ext_index = k+(Nz)*(j+(Ny)*i);
	fprintf(outBxExtest, "%i %i %i %.16e %.16e %.16e %.16e\n", 
		i, j, k, outx, outy, outz, Bx_ext3[ext_index]);
      }
    }
  }
  fclose(outBxExtest);

  outByExtest = fopen(ByExtest.str().c_str(), "w");
  for(k=0; k<Nz; k++)
  {  
    for(j=0; j<Ny+1; j++)
    {
      for(i=0; i<Nx+n_ext; i++)
      {
	outx = xinit + i*dx - (n_ext)*dx;
	outy = yinit + j*dy;
	outz = zinit + k*dz;
	ext_index = k+(Nz)*(j+(Ny+1)*i);
	fprintf(outByExtest, "%i %i %i %.16e %.16e %.16e %.16e\n", 
		i, j, k, outx, outy, outz, By_ext3[ext_index]);
      }
    }
  }
  fclose(outByExtest);

  outBzExtest = fopen(BzExtest.str().c_str(), "w");
  for(k=0; k<Nz+1; k++)
  {  
    for(j=0; j<Ny; j++)
    {
      for(i=0; i<Nx+n_ext; i++)
      {
	outx = xinit + i*dx - (n_ext)*dx;
	outy = yinit + j*dy;
	outz = zinit + k*dz;
	ext_index = k+(Nz+1)*(j+(Ny)*i);
	fprintf(outBzExtest, "%i %i %i %.16e %.16e %.16e %.16e\n", 
		i, j, k, outx, outy, outz, Bz_ext3[ext_index]);
      }
    }
  }
  fclose(outBzExtest);

  outAxExtest = fopen(AxExtest.str().c_str(), "w");
  for(k=0; k<Nz+1; k++)
  {
    for(j=0; j<Ny+1; j++)
    {
      for(i=0; i<Nx+n_ext; i++)
      {  
	outx = xinit + i*dx - (n_ext)*dx;
	outy = yinit + j*dy;
	outz = zinit + k*dz;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	fprintf(outAxExtest, "%i %i %i %.16e %.16e %.16e %.16e\n", 
		i, j, k, outx, outy, outz, Ax_ext3[ext_index]);
      }
    }
  }
  fclose(outAxExtest);

  outAyExtest = fopen(AyExtest.str().c_str(), "w");
  for(k=0; k<Nz+1; k++)
  {  
    for(j=0; j<Ny; j++)
    {
      for(i=0; i<Nx+1+n_ext; i++)
      {
	outx = xinit + i*dx - (n_ext-1)*dx;
	outy = yinit + j*dy;
	outz = zinit + k*dz;
	ext_index = k+(Nz+1)*(j+(Ny)*i);
	fprintf(outAyExtest, "%i %i %i %.16e %.16e %.16e %.16e\n", 
		i, j, k, outx, outy, outz, Ay_ext3[ext_index]);
      }
    }
  }
  fclose(outAyExtest);

  outAzExtest = fopen(AzExtest.str().c_str(), "w");
  for(k=0; k<Nz; k++)
  {  
    for(j=0; j<Ny+1; j++)
    {
      for(i=0; i<Nx+1+n_ext; i++)
      {
	outx = xinit + i*dx - (n_ext-1)*dx;
	outy = yinit + j*dy;
	outz = zinit + k*dz;
	ext_index = k+(Nz)*(j+(Ny+1)*i);
	fprintf(outAzExtest, "%i %i %i %.16e %.16e %.16e %.16e\n", 
		i, j, k, outx, outy, outz, Az_ext3[ext_index]);
      }
    }
  }
  fclose(outAzExtest);

  //Output various 1D cuts for testing.

  outBxi = fopen(Bxi.str().c_str(), "w");
  for(i=0; i<Nx+1+n_ext; i++)
  {
    j = cutyindex-1;
    k = cutzindex-1;
    outx = xinit + i*dx - n_ext*dx;
    outy = yinit + j*dy;
    outz = zinit + k*dz;
    ext_index = k+(Nz)*(j+(Ny)*i);
    fprintf(outBxi, "%i %i %i %.16e %.16e %.16e %.16e\n", 
	    i, j, k, outx, outy, outz, Bx_ext3[ext_index]);
  }
  fclose(outBxi);
  
  outBxj = fopen(Bxj.str().c_str(), "w");
  for(j=0; j<Ny; j++)
  {
    k = cutzindex-1;
    i = cutxindex;
    outx = xinit + i*dx;
    outy = yinit + (j+1)*dy;
    outz = zinit + k*dz;
    ext_index = k+(Nz)*(j+(Ny)*(i+n_ext));
    fprintf(outBxj, "%i %i %i %.16e %.16e %.16e %.16e\n", 
	    i, j, k, outx, outy, outz, Bx_ext3[ext_index]);
  }
  fclose(outBxj);
  
  outBxk = fopen(Bxk.str().c_str(), "w");
  for(k=0; k<Nz; k++)
  {
    i = cutxindex-zdiffindex;
    j = cutyindex-1-zdiffindex;
    outx = xinit + i*dx;
    outy = yinit + j*dy;
    outz = zinit + (k+1)*dz;
    ext_index = k+(Nz)*(j+(Ny)*(i+n_ext));
    fprintf(outBxk, "%i %i %i %.16e %.16e %.16e %.16e\n", 
	    i, j, k, outx, outy, outz, Bx_ext3[ext_index]);
  }
  fclose(outBxk);
  
  outByi = fopen(Byi.str().c_str(), "w");
  for(i=0; i<Nx+n_ext; i++)
  {
    j = cutyindex;
    k = cutzindex-1;
    outx = xinit + (i+1)*dx - n_ext*dx;
    outy = yinit + j*dy;
    outz = zinit + k*dz;
    ext_index = k+(Nz)*(j+(Ny+1)*i);
    fprintf(outByi, "%i %i %i %.16e %.16e %.16e %.16e\n", 
	    i, j, k, outx, outy, outz, By_ext3[ext_index]);
  }
  fclose(outByi);
  
  outByj = fopen(Byj.str().c_str(), "w");
  for(j=0; j<Ny+1; j++)
  {
    k = cutzindex-1;
    i = cutxindex-1;
    outx = xinit + i*dx;
    outy = yinit + j*dy;
    outz = zinit + k*dz;
    ext_index = k+(Nz)*(j+(Ny+1)*(i+n_ext));
    fprintf(outByj, "%i %i %i %.16e %.16e %.16e %.16e\n", 
	    i, j, k, outx, outy, outz, By_ext3[ext_index]);
  }
  fclose(outByj);
  
  outByk = fopen(Byk.str().c_str(), "w");
  for(k=0; k<Nz; k++)
  {
    i = cutxindex-1-zdiffindex;
    j = cutyindex-zdiffindex;
    outx = xinit + i*dx;
    outy = yinit + j*dy;
    outz = zinit + (k+1)*dz;
    ext_index = k+(Nz)*(j+(Ny+1)*(i+n_ext));
    fprintf(outByk, "%i %i %i %.16e %.16e %.16e %.16e\n", 
	    i, j, k, outx, outy, outz, By_ext3[ext_index]);
  }
  fclose(outByk);
  
  outBzi = fopen(Bzi.str().c_str(), "w");
  for(i=0; i<Nx+n_ext; i++)
  {
    j = cutyindex-1;
    k = cutzindex;
    outx = xinit + (i+1)*dx - n_ext*dx;
    outy = yinit + j*dy;
    outz = zinit + k*dz;
    ext_index = k+(Nz+1)*(j+(Ny)*i);
    fprintf(outBzi, "%i %i %i %.16e %.16e %.16e %.16e\n", 
	    i, j, k, outx, outy, outz, Bz_ext3[ext_index]);
  }
  fclose(outBzi);
  
  outBzj = fopen(Bzj.str().c_str(), "w");
  for(j=0; j<Ny; j++)
  {
    k = cutzindex;
    i = cutxindex-1;
    outx = xinit + i*dx;
    outy = yinit + (j+1)*dy;
    outz = zinit + k*dz;
    ext_index = k+(Nz+1)*(j+(Ny)*(i+n_ext));
    fprintf(outBzj, "%i %i %i %.16e %.16e %.16e %.16e\n", 
	    i, j, k, outx, outy, outz, Bz_ext3[ext_index]);
  }
  fclose(outBzj);
  
  outBzk = fopen(Bzk.str().c_str(), "w");
  for(k=0; k<Nz+1; k++)
  {
    i = cutxindex-1-zdiffindex;
    j = cutyindex-1-zdiffindex;
    outx = xinit + i*dx;
    outy = yinit + j*dy;
    outz = zinit + k*dz;
    ext_index = k+(Nz+1)*(j+(Ny)*(i+n_ext));
    fprintf(outBzk, "%i %i %i %.16e %.16e %.16e %.16e\n", 
	    i, j, k, outx, outy, outz, Bz_ext3[ext_index]);
  }
  fclose(outBzk);
  
  outAxi = fopen(Axi.str().c_str(), "w");
  for(i=0; i<Nx+1+n_ext; i++)
  {
    j = cutyindex;
    k = cutzindex;
    outx = xinit + i*dx - n_ext*dx;
    outy = yinit + j*dy;
    outz = zinit + k*dz;
    ext_index = k+(Nz+1)*(j+(Ny+1)*i);
    fprintf(outAxi, "%i %i %i %.16e %.16e %.16e %.16e\n", 
	    i, j, k, outx, outy, outz, Ax_ext2[ext_index]);
  }
  fclose(outAxi);
  
  outAxj = fopen(Axj.str().c_str(), "w");
  for(j=0; j<Ny+1; j++)
  {
    k = cutzindex;
    i = cutxindex;
    outx = xinit + i*dx;
    outy = yinit + j*dy;
    outz = zinit + k*dz;
    ext_index = k+(Nz+1)*(j+(Ny+1)*(i+n_ext));
    fprintf(outAxj, "%i %i %i %.16e %.16e %.16e %.16e\n", 
	    i, j, k, outx, outy, outz, Ax_ext2[ext_index]);
  }
  fclose(outAxj);
  
  outAxk = fopen(Axk.str().c_str(), "w");
  for(k=0; k<Nz+1; k++)
  {
    i = cutxindex-zdiffindex;
    j = cutyindex-zdiffindex;
    outx = xinit + i*dx;
    outy = yinit + j*dy;
    outz = zinit + k*dz;
    ext_index = k+(Nz+1)*(j+(Ny+1)*(i+n_ext));
    fprintf(outAxk, "%i %i %i %.16e %.16e %.16e %.16e\n", 
	    i, j, k, outx, outy, outz, Ax_ext2[ext_index]);
  }
  fclose(outAxk);
  
  outAyi = fopen(Ayi.str().c_str(), "w");
  for(i=0; i<Nx+1+n_ext; i++)
  {
    j = cutyindex;
    k = cutzindex;
    outx = xinit + i*dx - n_ext*dx;
    outy = yinit + j*dy;
    outz = zinit + k*dz;
    ext_index = k+(Nz+1)*(j+(Ny+1)*i);
    fprintf(outAyi, "%i %i %i %.16e %.16e %.16e %.16e\n", 
	    i, j, k, outx, outy, outz, Ay_ext2[ext_index]);
  }
  fclose(outAyi);
  
  outAyj = fopen(Ayj.str().c_str(), "w");
  for(j=0; j<Ny+1; j++)
  {
    k = cutzindex;
    i = cutxindex;
    outx = xinit + i*dx;
    outy = yinit + j*dy;
    outz = zinit + k*dz;
    ext_index = k+(Nz+1)*(j+(Ny+1)*(i+n_ext));
    fprintf(outAyj, "%i %i %i %.16e %.16e %.16e %.16e\n", 
	    i, j, k, outx, outy, outz, Ay_ext2[ext_index]);
  }
  fclose(outAyj);
  
  outAyk = fopen(Ayk.str().c_str(), "w");
  for(k=0; k<Nz+1; k++)
  {
    i = cutxindex-zdiffindex;
    j = cutyindex-zdiffindex;
    outx = xinit + i*dx;
    outy = yinit + j*dy;
    outz = zinit + k*dz;
    ext_index = k+(Nz+1)*(j+(Ny+1)*(i+n_ext));
    fprintf(outAyk, "%i %i %i %.16e %.16e %.16e %.16e\n", 
	    i, j, k, outx, outy, outz, Ay_ext2[ext_index]);
  }
  fclose(outAyk);
  
  outAzi = fopen(Azi.str().c_str(), "w");
  for(i=0; i<Nx+1+n_ext; i++)
  {
    j = cutyindex;
    k = cutzindex;
    outx = xinit + i*dx - n_ext*dx;
    outy = yinit + j*dy;
    outz = zinit + k*dz;
    ext_index = k+(Nz+1)*(j+(Ny+1)*i);
    fprintf(outAzi, "%i %i %i %.16e %.16e %.16e %.16e\n", 
	    i, j, k, outx, outy, outz, Az_ext2[ext_index]);
  }
  fclose(outAzi);
  
  outAzj = fopen(Azj.str().c_str(), "w");
  for(j=0; j<Ny+1; j++)
  {
    k = cutzindex;
    i = cutxindex;
    outx = xinit + i*dx;
    outy = yinit + j*dy;
    outz = zinit + k*dz;
    ext_index = k+(Nz+1)*(j+(Ny+1)*(i+n_ext));
    fprintf(outAzj, "%i %i %i %.16e %.16e %.16e %.16e\n", 
	    i, j, k, outx, outy, outz, Az_ext2[ext_index]);
  }
  fclose(outAzj);
  
  outAzk = fopen(Azk.str().c_str(), "w");
  for(k=0; k<Nz+1; k++)
  {
    i = cutxindex-zdiffindex;
    j = cutyindex-zdiffindex;
    outx = xinit + i*dx;
    outy = yinit + j*dy;
    outz = zinit + k*dz;
    ext_index = k+(Nz+1)*(j+(Ny+1)*(i+n_ext));
    fprintf(outAzk, "%i %i %i %.16e %.16e %.16e %.16e\n", 
	    i, j, k, outx, outy, outz, Az_ext2[ext_index]);
  }
  fclose(outAzk);
  

  delete[] Ax_ext2;
  delete[] Ay_ext2;
  delete[] Az_ext2;
  delete[] Ax_ext3;
  delete[] Ay_ext3;
  delete[] Az_ext3;
  delete[] Bx_ext3;
  delete[] By_ext3;
  delete[] Bz_ext3;

  return;
}

void extrap_ghost (double* Ax_ext,double* Ay_ext,double* Az_ext)
{
  //This function handles the ghost cells of the grid. It is
  //currently tied closely to how IllinoisGRMHD handles its
  //ghost cells. This method of solving for A does not assume
  //any ghost values, so they must be extrapolated afterwards.
  //Due to the staggering of the fields, tangential A-field
  //components must be extrapolated to one order higher than
  //normal components. For example, on an x-face of the grid,
  //if we use quadratic extrapolation for Ax, we must use 
  //cubic extrapolation for Ay and Az.

  int ext_index, index1, index2, index3, index4;

  if (ext_order_flag == 0)
  {
    //Lower x-Face

    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	i = n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	Ax_ext[ext_index] = 0.0;

	for(l=0; l<n_ghost; l++)
	{
	  i = n_ghost-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*(j+(Ny+1)*(i+1));

	  Ax_ext[ext_index] = 0.0;
	  
	  Ay_ext[ext_index] = Ay_ext[index1];

	  Az_ext[ext_index] = Az_ext[index1];
	}
      }
    }
  
    //Upper x-Face

    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  i = (Nx+1)-n_ghost+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*(j+(Ny+1)*(i-1));

	  Ax_ext[ext_index] = 0.0;
	  
	  Ay_ext[ext_index] = Ay_ext[index1];

	  Az_ext[ext_index] = Az_ext[index1];
	}
      }
    }
  
    //Lower y-Face

    for(i=0; i<Nx+1; i++)
    {
      for(k=0; k<Nz+1; k++)
      {
	j = n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	Ay_ext[ext_index] = 0.0;

	for(l=0; l<n_ghost; l++)
	{
	  j = n_ghost-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*((j+1)+(Ny+1)*i);

	  Ay_ext[ext_index] = 0.0;
	  
	  Az_ext[ext_index] = Az_ext[index1];

	  Ax_ext[ext_index] = Ax_ext[index1];
	}
      }
    }
  
    //Upper y-Face

    for(i=0; i<Nx+1; i++)
    {
      for(k=0; k<Nz+1; k++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  j = (Ny+1)-n_ghost+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*((j-1)+(Ny+1)*i);

	  Ay_ext[ext_index] = 0.0;
	  
	  Az_ext[ext_index] = Az_ext[index1];

	  Ax_ext[ext_index] = Ax_ext[index1];
	}
      }
    }
  
    //Lower z-Face

    for(i=0; i<Nx+1; i++)
    {
      for(j=0; j<Ny+1; j++)
      {
	k = n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	Az_ext[ext_index] = 0.0;

	for(l=0; l<n_ghost; l++)
	{
	  k = n_ghost-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = (k+1)+(Nz+1)*(j+(Ny+1)*i);

	  Az_ext[ext_index] = 0.0;
	  
	  Ax_ext[ext_index] = Ax_ext[index1];

	  Ay_ext[ext_index] = Ay_ext[index1];
	}
      }
    }
    
    //Upper z-Face

    for(i=0; i<Nx+1; i++)
    {
      for(j=0; j<Ny+1; j++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  k = (Nz+1)-(n_ghost)+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = (k-1)+(Nz+1)*(j+(Ny+1)*i);

	  Az_ext[ext_index] = 0.0;
	  
	  Ax_ext[ext_index] = Ax_ext[index1];

	  Ay_ext[ext_index] = Ay_ext[index1];
	}
      }
    }
  }

  else if (ext_order_flag == 1)
  {
    //Lower x-Face

    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	i = n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	index1 = k+(Nz+1)*(j+(Ny+1)*(i+1));
	Ax_ext[ext_index] = Ax_ext[index1];

	for(l=0; l<n_ghost; l++)
	{
	  i = n_ghost-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*(j+(Ny+1)*(i+1));
	  index2 = k+(Nz+1)*(j+(Ny+1)*(i+2));

	  Ax_ext[ext_index] = Ax_ext[index1];
	  
	  Ay_ext[ext_index] = 2*Ay_ext[index1] 
	                    - 1*Ay_ext[index2];

	  Az_ext[ext_index] = 2*Az_ext[index1] 
	                    - 1*Az_ext[index2];
	}
      }
    }
  
    //Upper x-Face

    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  i = (Nx+1)-n_ghost+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*(j+(Ny+1)*(i-1));
	  index2 = k+(Nz+1)*(j+(Ny+1)*(i-2));

	  Ax_ext[ext_index] = Ax_ext[index1];
	  
	  Ay_ext[ext_index] = 2*Ay_ext[index1] 
	                    - 1*Ay_ext[index2];

	  Az_ext[ext_index] = 2*Az_ext[index1] 
	                    - 1*Az_ext[index2];
	}
      }
    }
  
    //Lower y-Face

    for(i=0; i<Nx+1; i++)
    {
      for(k=0; k<Nz+1; k++)
      {
	j = n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	index1 = k+(Nz+1)*((j+1)+(Ny+1)*i);
	Ay_ext[ext_index] = Ay_ext[index1];

	for(l=0; l<n_ghost; l++)
	{
	  j = n_ghost-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*((j+1)+(Ny+1)*i);
	  index2 = k+(Nz+1)*((j+2)+(Ny+1)*i);

	  Ay_ext[ext_index] = Ay_ext[index1];
	  
	  Az_ext[ext_index] = 2*Az_ext[index1] 
	                    - 1*Az_ext[index2];

	  Ax_ext[ext_index] = 2*Ax_ext[index1] 
	                    - 1*Ax_ext[index2];
	}
      }
    }
  
    //Upper y-Face

    for(i=0; i<Nx+1; i++)
    {
      for(k=0; k<Nz+1; k++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  j = (Ny+1)-n_ghost+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*((j-1)+(Ny+1)*i);
	  index2 = k+(Nz+1)*((j-2)+(Ny+1)*i);

	  Ay_ext[ext_index] = Ay_ext[index1];
	  
	  Az_ext[ext_index] = 2*Az_ext[index1] 
	                    - 1*Az_ext[index2];

	  Ax_ext[ext_index] = 2*Ax_ext[index1] 
	                    - 1*Ax_ext[index2];
	}
      }
    }
  
    //Lower z-Face

    for(i=0; i<Nx+1; i++)
    {
      for(j=0; j<Ny+1; j++)
      {
	k = n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	index1 = (k+1)+(Nz+1)*(j+(Ny+1)*i);
	Az_ext[ext_index] = Az_ext[index1];

	for(l=0; l<n_ghost; l++)
	{
	  k = n_ghost-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = (k+1)+(Nz+1)*(j+(Ny+1)*i);
	  index2 = (k+2)+(Nz+1)*(j+(Ny+1)*i);

	  Az_ext[ext_index] = Az_ext[index1];
	  
	  Ax_ext[ext_index] = 2*Ax_ext[index1] 
	                    - 1*Ax_ext[index2];

	  Ay_ext[ext_index] = 2*Ay_ext[index1] 
	                    - 1*Ay_ext[index2];
	}
      }
    }
    
    //Upper z-Face

    for(i=0; i<Nx+1; i++)
    {
      for(j=0; j<Ny+1; j++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  k = (Nz+1)-(n_ghost)+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = (k-1)+(Nz+1)*(j+(Ny+1)*i);
	  index2 = (k-2)+(Nz+1)*(j+(Ny+1)*i);

	  Az_ext[ext_index] = Az_ext[index1];
	  
	  Ax_ext[ext_index] = 2*Ax_ext[index1] 
	                    - 1*Ax_ext[index2];

	  Ay_ext[ext_index] = 2*Ay_ext[index1] 
	                    - 1*Ay_ext[index2];
	}
      }
    }
  }

  else if (ext_order_flag == 2)
  {
    //Lower x-Face

    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	i = n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	index1 = k+(Nz+1)*(j+(Ny+1)*(i+1));
	index2 = k+(Nz+1)*(j+(Ny+1)*(i+2));
	Ax_ext[ext_index] = 2*Ax_ext[index1] 
	                  - 1*Ax_ext[index2];

	for(l=0; l<n_ghost; l++)
	{
	  i = n_ghost-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*(j+(Ny+1)*(i+1));
	  index2 = k+(Nz+1)*(j+(Ny+1)*(i+2));
	  index3 = k+(Nz+1)*(j+(Ny+1)*(i+3));

	  Ax_ext[ext_index] = 2*Ax_ext[index1] 
	                    - 1*Ax_ext[index2];

	  Ay_ext[ext_index] = 3*Ay_ext[index1] 
	                    - 3*Ay_ext[index2]
	                    + 1*Ay_ext[index3];

	  Az_ext[ext_index] = 3*Az_ext[index1] 
	                    - 3*Az_ext[index2]
	                    + 1*Az_ext[index3];
	}
      }
    }
  
    //Upper x-Face

    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  i = (Nx+1)-n_ghost+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*(j+(Ny+1)*(i-1));
	  index2 = k+(Nz+1)*(j+(Ny+1)*(i-2));
	  index3 = k+(Nz+1)*(j+(Ny+1)*(i-3));

	  Ax_ext[ext_index] = 2*Ax_ext[index1] 
	                    - 1*Ax_ext[index2];

	  Ay_ext[ext_index] = 3*Ay_ext[index1] 
	                    - 3*Ay_ext[index2]
	                    + 1*Ay_ext[index3];

	  Az_ext[ext_index] = 3*Az_ext[index1] 
	                    - 3*Az_ext[index2]
	                    + 1*Az_ext[index3];
	}
      }
    }
  
    //Lower y-Face

    for(i=0; i<Nx+1; i++)
    {
      for(k=0; k<Nz+1; k++)
      {
	j = n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	index1 = k+(Nz+1)*((j+1)+(Ny+1)*i);
	index2 = k+(Nz+1)*((j+2)+(Ny+1)*i);
	Ay_ext[ext_index] = 2*Ay_ext[index1] 
	                  - 1*Ay_ext[index2];

	for(l=0; l<n_ghost; l++)
	{
	  j = n_ghost-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*((j+1)+(Ny+1)*i);
	  index2 = k+(Nz+1)*((j+2)+(Ny+1)*i);
	  index3 = k+(Nz+1)*((j+3)+(Ny+1)*i);

	  Ay_ext[ext_index] = 2*Ay_ext[index1] 
	                    - 1*Ay_ext[index2];

	  Az_ext[ext_index] = 3*Az_ext[index1] 
	                    - 3*Az_ext[index2]
	                    + 1*Az_ext[index3];

	  Ax_ext[ext_index] = 3*Ax_ext[index1] 
	                    - 3*Ax_ext[index2]
	                    + 1*Ax_ext[index3];
	}
      }
    }
  
    //Upper y-Face

    for(i=0; i<Nx+1; i++)
    {
      for(k=0; k<Nz+1; k++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  j = (Ny+1)-n_ghost+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*((j-1)+(Ny+1)*i);
	  index2 = k+(Nz+1)*((j-2)+(Ny+1)*i);
	  index3 = k+(Nz+1)*((j-3)+(Ny+1)*i);

	  Ay_ext[ext_index] = 2*Ay_ext[index1] 
	                    - 1*Ay_ext[index2];

	  Az_ext[ext_index] = 3*Az_ext[index1] 
	                    - 3*Az_ext[index2]
	                    + 1*Az_ext[index3];

	  Ax_ext[ext_index] = 3*Ax_ext[index1] 
	                    - 3*Ax_ext[index2]
	                    + 1*Ax_ext[index3];
	}
      }
    }
  
    //Lower z-Face

    for(i=0; i<Nx+1; i++)
    {
      for(j=0; j<Ny+1; j++)
      {
	k = n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	index1 = (k+1)+(Nz+1)*(j+(Ny+1)*i);
	index2 = (k+2)+(Nz+1)*(j+(Ny+1)*i);
	Az_ext[ext_index] = 2*Az_ext[index1] 
	                  - 1*Az_ext[index2];

	for(l=0; l<n_ghost; l++)
	{
	  k = n_ghost-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = (k+1)+(Nz+1)*(j+(Ny+1)*i);
	  index2 = (k+2)+(Nz+1)*(j+(Ny+1)*i);
	  index3 = (k+3)+(Nz+1)*(j+(Ny+1)*i);

	  Az_ext[ext_index] = 2*Az_ext[index1] 
	                    - 1*Az_ext[index2];

	  Ax_ext[ext_index] = 3*Ax_ext[index1] 
	                    - 3*Ax_ext[index2]
	                    + 1*Ax_ext[index3];

	  Ay_ext[ext_index] = 3*Ay_ext[index1] 
	                    - 3*Ay_ext[index2]
	                    + 1*Ay_ext[index3];
	}
      }
    }
    
    //Upper z-Face

    for(i=0; i<Nx+1; i++)
    {
      for(j=0; j<Ny+1; j++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  k = (Nz+1)-(n_ghost)+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = (k-1)+(Nz+1)*(j+(Ny+1)*i);
	  index2 = (k-2)+(Nz+1)*(j+(Ny+1)*i);
	  index3 = (k-3)+(Nz+1)*(j+(Ny+1)*i);

	  Az_ext[ext_index] = 2*Az_ext[index1] 
	                    - 1*Az_ext[index2];

	  Ax_ext[ext_index] = 3*Ax_ext[index1] 
	                    - 3*Ax_ext[index2]
	                    + 1*Ax_ext[index3];

	  Ay_ext[ext_index] = 3*Ay_ext[index1] 
	                    - 3*Ay_ext[index2]
	                    + 1*Ay_ext[index3];
	}
      }
    }
  }

  else if (ext_order_flag == 3)
  {
    //Lower x-Face

    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	i = n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	index1 = k+(Nz+1)*(j+(Ny+1)*(i+1));
	index2 = k+(Nz+1)*(j+(Ny+1)*(i+2));	
	index3 = k+(Nz+1)*(j+(Ny+1)*(i+3));
	Ax_ext[ext_index] = 3*Ax_ext[index1] 
	                  - 3*Ax_ext[index2]
	                  + 1*Ax_ext[index3];

	for(l=0; l<n_ghost; l++)
	{
	  i = n_ghost-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*(j+(Ny+1)*(i+1));
	  index2 = k+(Nz+1)*(j+(Ny+1)*(i+2));
	  index3 = k+(Nz+1)*(j+(Ny+1)*(i+3));
	  index4 = k+(Nz+1)*(j+(Ny+1)*(i+4));

	  Ax_ext[ext_index] = 3*Ax_ext[index1] 
	                    - 3*Ax_ext[index2]
	                    + 1*Ax_ext[index3];

	  Ay_ext[ext_index] = 4*Ay_ext[index1] 
	                    - 6*Ay_ext[index2]
	                    + 4*Ay_ext[index3]
	                    - 1*Ay_ext[index4];

	  Az_ext[ext_index] = 4*Az_ext[index1] 
	                    - 6*Az_ext[index2]
	                    + 4*Az_ext[index3]
	                    - 1*Az_ext[index4];
	}
      }
    }
  
    //Upper x-Face

    for(j=0; j<Ny+1; j++)
    {
      for(k=0; k<Nz+1; k++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  i = (Nx+1)-n_ghost+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*(j+(Ny+1)*(i-1));
	  index2 = k+(Nz+1)*(j+(Ny+1)*(i-2));
	  index3 = k+(Nz+1)*(j+(Ny+1)*(i-3));
	  index4 = k+(Nz+1)*(j+(Ny+1)*(i-4));

	  Ax_ext[ext_index] = 3*Ax_ext[index1] 
	                    - 3*Ax_ext[index2]
	                    + 1*Ax_ext[index3];

	  Ay_ext[ext_index] = 4*Ay_ext[index1] 
	                    - 6*Ay_ext[index2]
	                    + 4*Ay_ext[index3]
	                    - 1*Ay_ext[index4];

	  Az_ext[ext_index] = 4*Az_ext[index1] 
	                    - 6*Az_ext[index2]
	                    + 4*Az_ext[index3]
	                    - 1*Az_ext[index4];
	}
      }
    }
  
    //Lower y-Face

    for(i=0; i<Nx+1; i++)
    {
      for(k=0; k<Nz+1; k++)
      {
	j = n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	index1 = k+(Nz+1)*((j+1)+(Ny+1)*i);
	index2 = k+(Nz+1)*((j+2)+(Ny+1)*i);
	index3 = k+(Nz+1)*((j+3)+(Ny+1)*i);
	Ay_ext[ext_index] = 3*Ay_ext[index1] 
	                  - 3*Ay_ext[index2]
	                  + 1*Ay_ext[index3];

	for(l=0; l<n_ghost; l++)
	{
	  j = n_ghost-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*((j+1)+(Ny+1)*i);
	  index2 = k+(Nz+1)*((j+2)+(Ny+1)*i);
	  index3 = k+(Nz+1)*((j+3)+(Ny+1)*i);
	  index4 = k+(Nz+1)*((j+4)+(Ny+1)*i);

	  Ay_ext[ext_index] = 3*Ay_ext[index1] 
	                    - 3*Ay_ext[index2]
	                    + 1*Ay_ext[index3];

	  Az_ext[ext_index] = 4*Az_ext[index1] 
	                    - 6*Az_ext[index2]
	                    + 4*Az_ext[index3]
	                    - 1*Az_ext[index4];

	  Ax_ext[ext_index] = 4*Ax_ext[index1] 
	                    - 6*Ax_ext[index2]
	                    + 4*Ax_ext[index3]
	                    - 1*Ax_ext[index4];
	}
      }
    }
  
    //Upper y-Face

    for(i=0; i<Nx+1; i++)
    {
      for(k=0; k<Nz+1; k++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  j = (Ny+1)-n_ghost+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = k+(Nz+1)*((j-1)+(Ny+1)*i);
	  index2 = k+(Nz+1)*((j-2)+(Ny+1)*i);
	  index3 = k+(Nz+1)*((j-3)+(Ny+1)*i);
	  index4 = k+(Nz+1)*((j-4)+(Ny+1)*i);

	  Ay_ext[ext_index] = 3*Ay_ext[index1] 
	                    - 3*Ay_ext[index2]
	                    + 1*Ay_ext[index3];

	  Az_ext[ext_index] = 4*Az_ext[index1] 
	                    - 6*Az_ext[index2]
	                    + 4*Az_ext[index3]
	                    - 1*Az_ext[index4];

	  Ax_ext[ext_index] = 4*Ax_ext[index1] 
	                    - 6*Ax_ext[index2]
	                    + 4*Ax_ext[index3]
	                    - 1*Ax_ext[index4];
	}
      }
    }
  
    //Lower z-Face

    for(i=0; i<Nx+1; i++)
    {
      for(j=0; j<Ny+1; j++)
      {
	k = n_ghost;
	ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	index1 = (k+1)+(Nz+1)*(j+(Ny+1)*i);
	index2 = (k+2)+(Nz+1)*(j+(Ny+1)*i);
	index3 = (k+3)+(Nz+1)*(j+(Ny+1)*i);
	Az_ext[ext_index] = 3*Az_ext[index1] 
	                  - 3*Az_ext[index2]
	                  + 1*Az_ext[index3];

	for(l=0; l<n_ghost; l++)
	{
	  k = n_ghost-(l+1);
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = (k+1)+(Nz+1)*(j+(Ny+1)*i);
	  index2 = (k+2)+(Nz+1)*(j+(Ny+1)*i);
	  index3 = (k+3)+(Nz+1)*(j+(Ny+1)*i);
	  index4 = (k+4)+(Nz+1)*(j+(Ny+1)*i);

	  Az_ext[ext_index] = 3*Az_ext[index1] 
	                    - 3*Az_ext[index2]
	                    + 1*Az_ext[index3];

	  Ax_ext[ext_index] = 4*Ax_ext[index1] 
	                    - 6*Ax_ext[index2]
	                    + 4*Ax_ext[index3]
	                    - 1*Ax_ext[index4];

	  Ay_ext[ext_index] = 4*Ay_ext[index1] 
	                    - 6*Ay_ext[index2]
	                    + 4*Ay_ext[index3]
	                    - 1*Ay_ext[index4];
	}
      }
    }
    
    //Upper z-Face

    for(i=0; i<Nx+1; i++)
    {
      for(j=0; j<Ny+1; j++)
      {
	for(l=0; l<n_ghost; l++)
	{
	  k = (Nz+1)-(n_ghost)+l;
	  ext_index = k+(Nz+1)*(j+(Ny+1)*i);
	  index1 = (k-1)+(Nz+1)*(j+(Ny+1)*i);
	  index2 = (k-2)+(Nz+1)*(j+(Ny+1)*i);
	  index3 = (k-3)+(Nz+1)*(j+(Ny+1)*i);
	  index4 = (k-4)+(Nz+1)*(j+(Ny+1)*i);

	  Az_ext[ext_index] = 3*Az_ext[index1] 
	                    - 3*Az_ext[index2]
	                    + 1*Az_ext[index3];

	  Ax_ext[ext_index] = 4*Ax_ext[index1] 
	                    - 6*Ax_ext[index2]
	                    + 4*Ax_ext[index3]
	                    - 1*Ax_ext[index4];

	  Ay_ext[ext_index] = 4*Ay_ext[index1] 
	                    - 6*Ay_ext[index2]
	                    + 4*Ay_ext[index3]
	                    - 1*Ay_ext[index4];
	}
      }
    }
  }

  return;
}


//----------------------------------------------------------------------
// The following routines force the B field to have zero divergence.
//----------------------------------------------------------------------

void clean6faces (double B1m,double B1p,double& B1mnew,double& B1pnew,
		  double B2m,double B2p,double& B2mnew,double& B2pnew,
		  double B3m,double B3p,double& B3mnew,double& B3pnew)
{
  //First Cell

  //i=j=k=0 : nothing fixed

  //1=x, 2=y, 3=z

  double F_net, F_net6;

  F_net = B1p + B2p + B3p - B1m - B2m - B3m;
  F_net6 = F_net/6.0;
  
  B1pnew = B1p - F_net6;
  B1mnew = B1m + F_net6;

  B2pnew = B2p - F_net6;
  B2mnew = B2m + F_net6;

  B3pnew = B3p - F_net6;
  B3mnew = B3m + F_net6;

  return;
}

void clean5faces (double B1m,double B1p,double& B1pnew,
		  double B2m,double B2p,double& B2mnew,double& B2pnew,
		  double B3m,double B3p,double& B3mnew,double& B3pnew)
{
  //Coordinate Rays

  //j=k=0 : Bxm fixed
  //i=k=0 : Bym fixed
  //i=j=0 : Bzm fixed

  //When j=k=0, 1=x, 2=y, 3=z
  //When k=i=0, 1=y, 2=z, 3=x
  //When i=j=0, 1=z, 2=x, 3=y

  double F_net, F_net5;

  F_net = B1p + B2p + B3p - B1m - B2m - B3m;
  F_net5 = F_net/5.0;

  B1pnew = B1p - F_net5;

  B2pnew = B2p - F_net5;
  B2mnew = B2m + F_net5;

  B3pnew = B3p - F_net5;
  B3mnew = B3m + F_net5;

  return;
}

void clean4faces (double B1m,double B1p,double& B1pnew,
		  double B2m,double B2p,double& B2pnew,
		  double B3m,double B3p,double& B3mnew,double& B3pnew)
{
  //Coordinate Planes

  //k=0 : Bxm, Bym fixed
  //i=0 : Bym, Bzm fixed
  //j=0 : Bzm, Bxm fixed

  //When k=0, 1=x, 2=y, 3=z
  //When i=0, 1=y, 2=z, 3=x
  //When j=0, 1=z, 2=x, 3=y

  double F_net, F_net4;

  F_net = B1p + B2p + B3p - B1m - B2m - B3m;
  F_net4 = F_net/4.0;

  B1pnew = B1p - F_net4;

  B2pnew = B2p - F_net4;

  B3pnew = B3p - F_net4;
  B3mnew = B3m + F_net4;  

  return;
}

void clean3faces (double B1m,double B1p,double& B1pnew,
		  double B2m,double B2p,double& B2pnew,
		  double B3m,double B3p,double& B3pnew)
{
  //Bulk of Grid

  //i,j,k!=0 : Bxm, Bym, Bzm fixed

  //1=x, 2=y, 3=z

  double F_net, F_net3;

  F_net = B1p + B2p + B3p - B1m - B2m - B3m;
  F_net3 = F_net/3.0;

  B1pnew = B1p - F_net3;

  B2pnew = B2p - F_net3;

  B3pnew = B3p - F_net3;

  return;
}


//----------------------------------------------------------------------
// The following routines construct the A field.
//----------------------------------------------------------------------

void calcA6faces (double& A1mm,double& A1pm,double& A1mp,double& A1pp,
		  double& A2mm,double& A2pm,double& A2mp,double& A2pp,
		  double& A3mm,double& A3pm,double& A3mp,double& A3pp,
		  double B1m,double B1p,double B2m,double B2p,double B3m,double B3p,
		  double step1,double step2,double step3)
{
  //First Cell

  //i=initi,j=initj,k=initk : nothing fixed; 1=x,2=y,3=z

  A1mm = 5.0/24.0*(-B2m*step3 + B3m*step2) + 1.0/24.0*(-B2p*step3 + B3p*step2);
  A1pm = 5.0/24.0*(-B2p*step3 - B3m*step2) + 1.0/24.0*(-B2m*step3 - B3p*step2);
  A1mp = 5.0/24.0*( B2m*step3 + B3p*step2) + 1.0/24.0*( B2p*step3 + B3m*step2);
  A1pp = 5.0/24.0*( B2p*step3 - B3p*step2) + 1.0/24.0*( B2m*step3 - B3m*step3);

  A2mm = 5.0/24.0*(-B3m*step1 + B1m*step3) + 1.0/24.0*(-B3p*step1 + B1p*step3);
  A2pm = 5.0/24.0*(-B3p*step1 - B1m*step3) + 1.0/24.0*(-B3m*step1 - B1p*step3);
  A2mp = 5.0/24.0*( B3m*step1 + B1p*step3) + 1.0/24.0*( B3p*step1 + B1m*step3);
  A2pp = 5.0/24.0*( B3p*step1 - B1p*step3) + 1.0/24.0*( B3m*step1 - B1m*step3);

  A3mm = 5.0/24.0*(-B1m*step2 + B2m*step1) + 1.0/24.0*(-B1p*step2 + B2p*step1);
  A3pm = 5.0/24.0*(-B1p*step2 - B2m*step1) + 1.0/24.0*(-B1m*step2 - B2p*step1);
  A3mp = 5.0/24.0*( B1m*step2 + B2p*step1) + 1.0/24.0*( B1p*step2 + B2m*step1);
  A3pp = 5.0/24.0*( B1p*step2 - B2p*step1) + 1.0/24.0*( B1m*step2 - B2m*step1);
  /*  
  cout << A1mm << " " << A1pm << " " << A1mp << " " << A1pp << endl;
  cout << A2mm << " " << A2pm << " " << A2mp << " " << A2pp << endl;
  cout << A3mm << " " << A3pm << " " << A3mp << " " << A3pp << endl;
  cout << endl;
  */

  return;
}

void calcA5faces (double& A1mm,double& A1pm,double& A1mp,double& A1pp,
		  double& A2mm,double& A2pm,double& A2mp,double& A2pp,
		  double& A3mm,double& A3pm,double& A3mp,double& A3pp,
		  double B1m,double B1p,double B2m,double B2p,double B3m,double B3p,
		  double A2mm_prev,double A3mm_prev,
		  double A2pm_prev,double A3mp_prev)
{
  //Coordinate Rays

  //j=initj,k=initk : Aymm,Azmm,Aypm,Azmp fixed; 1=x,2=y,3=z
  //k=initk,i=initi : Azmm,Axmm,Azpm,Axmp fixed; 1=y,2=z,3=x
  //i=initi,j=initj : Axmm,Aymm,Axpm,Aymp fixed; 1=z,2=x,3=y

  double a, b, delta1, delta2, delta3, delta4;
  double A1mm_1, A1mp_1, A1pp_1, A1pm_1, A2mp_1, A3pm_1, A2pp_1, A3pp_1;
  double A1mm_2, A1mp_2, A1pp_2, A1pm_2, A2mp_2, A3pm_2, A2pp_2, A3pp_2;
  /*
  cout << endl;
  cout << "From calcA5faces:" << endl;
  cout << i << " " << j << " " << k << endl;
  cout << setiosflags(ios::fixed) << setprecision(10);
  cout << setw(15) << A2mm_prev
       << setw(15) << A3mm_prev
       << setw(15) << A2pm_prev
       << setw(15) << A3mp_prev << endl;
  cout << endl;  
  */

  //The first option, in which differences are
  //propagated to the opposite face.

  A2mp_1 = A2mp - (A2mm - A2mm_prev);
  A3pm_1 = A3pm - (A3mm - A3mm_prev);
  A2pp_1 = A2pp - (A2pm - A2pm_prev);
  A3pp_1 = A3pp - (A3mp - A3mp_prev);

  A1mm_1 = A1mm;
  A1mp_1 = A1mp;
  A1pp_1 = A1pp;
  A1pm_1 = A1pm;


  //The second option, in which differences
  //are propagated to adjacent edges.

  a = -3.0/8.0;
  b =  1.0/8.0;
  
  //Find the corrections that must be made.

  delta1 = -(A2mm - A2mm_prev);
  delta2 =  (A3mm - A3mm_prev);
  delta3 =  (A2pm - A2pm_prev);
  delta4 = -(A3mp - A3mp_prev);

  //Make the corrections.

  A1mm_2 = A1mm + -a*delta1 + a*delta2 - b*delta3 + b*delta4;
  A1mp_2 = A1mp + -a*delta2 + a*delta3 - b*delta4 + b*delta1;
  A1pp_2 = A1pp + -a*delta3 + a*delta4 - b*delta1 + b*delta2;
  A1pm_2 = A1pm + -a*delta4 + a*delta1 - b*delta2 + b*delta3;

  A2mp_2 = A2mp;
  A3pm_2 = A3pm;
  A2pp_2 = A2pp;
  A3pp_2 = A3pp;


  //Take the linear combination of these two 
  //solutions, using the parameter set at the
  //top of the code.
  
  A1mm = (1.0-option5)*A1mm_1 + option5*A1mm_2;
  A1mp = (1.0-option5)*A1mp_1 + option5*A1mp_2;
  A1pp = (1.0-option5)*A1pp_1 + option5*A1pp_2;
  A1pm = (1.0-option5)*A1pm_1 + option5*A1pm_2;
  A2mp = (1.0-option5)*A2mp_1 + option5*A2mp_2;
  A3pm = (1.0-option5)*A3pm_1 + option5*A3pm_2;
  A2pp = (1.0-option5)*A2pp_1 + option5*A2pp_2;
  A3pp = (1.0-option5)*A3pp_1 + option5*A3pp_2;
  

  //Ensure that the sides that should remain
  //fixed do indeed remain fixed.

  A2mm = A2mm_prev;
  A3mm = A3mm_prev;
  A2pm = A2pm_prev;
  A3mp = A3mp_prev;

  return;
}

void calcA4faces (double& A1mm,double& A1pm,double& A1mp,double& A1pp,
		  double& A2mm,double& A2pm,double& A2mp,double& A2pp,
		  double& A3mm,double& A3pm,double& A3mp,double& A3pp,
		  double B1m,double B1p,double B2m,double B2p,double B3m,double B3p,
		  double A1mp_prev,double A2pm_prev,double A3mp_prev,
		  double A2mm_prev,double A1mm_prev,double A3pm_prev,double A3mm_prev,
		  double A1pm_prev,double A1pp_prev,double A2mp_prev,double A2pp_prev)
{
  //Coordinate Planes

  //k=initk : Axmp,Aypm,Axmm,Aymm,Azmp,Azpm,Azmm fixed; 1=x, 2=y, 3=z
  //i=initi : Aymp,Azpm,Aymm,Azmm,Axmp,Axpm,Axmm fixed; 1=y, 2=z, 3=x
  //j=initj : Azmp,Axpm,Azmm,Axmm,Aymp,Aypm,Aymm fixed; 1=z, 2=x, 3=y

  /*
  cout << endl;
  cout << "From calcA4faces:" << endl;
  cout << i << " " << j << " " << k << endl;
  cout << setiosflags(ios::fixed) << setprecision(10);
  cout << setw(15) << A1mp_prev
       << setw(15) << A2pm_prev
       << setw(15) << A3mp_prev
       << setw(15) << A2mm_prev
       << setw(15) << A1mm_prev
       << setw(15) << A3pm_prev
       << setw(15) << A3mm_prev << endl;
  cout << endl;
  */

  if (option4 < 0.0 || option4 > 1.0)
  {
    double a, b, delta1, delta2, delta3, delta4;

    a = -3.0/8.0;
    b =  1.0/8.0;

    //Find the corrections that must be made.

    delta1 = -(A1mp - A1mp_prev) + (A2pm - A2pm_prev);
    delta2 = -(A3mp - A3mp_prev);
    delta3 = -(A2mm - A2mm_prev) + (A1mm - A1mm_prev);
    delta4 =  (A3pm - A3pm_prev);

    //Make the corrections.

    A2pp += -a*delta4 + a*delta1 - b*delta2 + b*delta3;
    A1pp += -a*delta1 + a*delta2 - b*delta3 + b*delta4;
    A1pm += -a*delta2 + a*delta3 - b*delta4 + b*delta1;
    A2mp += -a*delta3 + a*delta4 - b*delta1 + b*delta2;

    //Ensure that the sides that should remain
    //fixed do indeed remain fixed.

    A1mp = A1mp_prev;
    A2pm = A2pm_prev;
    A1mm = A1mm_prev;
    A2mm = A2mm_prev;
    A3mp = A3mp_prev;
    A3pm = A3pm_prev;
    A3mm = A3mm_prev;
  }
  else
  {
    double A1pm_1, A1pp_1, A2mp_1, A2pp_1, A3pp_1;
    double A1pm_2, A1pp_2, A2mp_2, A2pp_2, A3pp_2;
    double Cp, Cm;

    //Ensure that the sides that should remain
    //fixed do indeed remain fixed.

    A1mp = A1mp_prev;
    A2pm = A2pm_prev;
    A1mm = A1mm_prev;
    A2mm = A2mm_prev;
    A3mp = A3mp_prev;
    A3pm = A3pm_prev;
    A3mm = A3mm_prev;


    //The first possible algorithm, in which
    //information is propagated along the two axes
    //for which a face is known. For example, if we
    //know an x face and a y face, information is
    //propagated from previous x values along the
    //x-axis and from previous y values along
    //the y-axis.

    //Set 4 of the unknown edges to previous values
    //that are adjacent, and therefore in a previous
    //cell.

    A1pm_1 = A1pm_prev;
    A1pp_1 = A1pp_prev;
    A2mp_1 = A2mp_prev;
    A2pp_1 = A2pp_prev;

    //This will make the B3p and B3m faces not work,
    //so we must correct the edges we set.

    Cp = 0.5*((A2pp_1 - A2pm) - (A1pp_1 - A1mp) - step*B3p);
    Cm = 0.5*((A2mp_1 - A2mm) - (A1pm_1 - A1mm) - step*B3m);

    A1pm_1 += Cm;
    A1pp_1 += Cp;
    A2mp_1 -= Cm;
    A2pp_1 -= Cp;

    //That leaves one edge undetermined, A3pp.
    
    A3pp_1 =  0.5*( step*B1p + A2pp_1 - A2mp_1 + A3pm)
             +0.5*(-step*B2p + A1pp_1 - A1pm_1 + A3mp);


    //The second possible algorithm, in which
    //information is also propagated along the two
    //axes for which a face is known, but in the
    //opposite sense. For example, if we know an
    //x face and a y face, information is propagated
    //from previous x values along the y-axis and 
    //from previous y values along the x-axis.

    //Set 4 of the unknown edges to previous values
    //that are opposite, and therefore in this cell.

    A1pm_2 = A1mm;
    A1pp_2 = A1mp;
    A2mp_2 = A2mm;
    A2pp_2 = A2pm;

    //This will make the B3p and B3m faces not work,
    //so we must correct these edges.

    Cp = 0.5*((A2pp_2 - A2pm) - (A1pp_2 - A1mp) - step*B3p);
    Cm = 0.5*((A2mp_2 - A2mm) - (A1pm_2 - A1mm) - step*B3m);

    A1pm_2 += Cm;
    A1pp_2 += Cp;
    A2mp_2 -= Cm;
    A2pp_2 -= Cp;

    //That leaves one edge undetermined, A3pp.
    
    A3pp_2 =  0.5*( step*B1p + A2pp_2 - A2mp_2 + A3pm)
             +0.5*(-step*B2p + A1pp_2 - A1pm_2 + A3mp);


    //Now take the linear combination of these two
    //solutions, using the parameter set at the 
    //top of the code.

    A1pm = (1.0-option4)*A1pm_1 + option4*A1pm_2;
    A1pp = (1.0-option4)*A1pp_1 + option4*A1pp_2;
    A2mp = (1.0-option4)*A2mp_1 + option4*A2mp_2;
    A2pp = (1.0-option4)*A2pp_1 + option4*A2pp_2;
    A3pp = (1.0-option4)*A3pp_1 + option4*A3pp_2;
  }

  return;
}

void calcA3faces (double& A1mm,double& A1pm,double& A1mp,double& A1pp,
		  double& A2mm,double& A2pm,double& A2mp,double& A2pp,
		  double& A3mm,double& A3pm,double& A3mp,double& A3pp,
		  double B1m,double B1p,double B2m,double B2p,double B3m,double B3p,
		  double A1mm_prev,double A1pm_prev,double A1mp_prev,
		  double A2mm_prev,double A2pm_prev,double A2mp_prev,
		  double A3mm_prev,double A3pm_prev,double A3mp_prev,
		  double A1pp_prev,double A2pp_prev,double A3pp_prev)
{
  //Bulk of Grid

  //i!=initi,j!=initj,k!=initk : Axpp,Aypp,Azpp NOT fixed; 1=x, 2=y, 3=z

  /*
  cout << endl;
  cout << "From calcA3faces:" << endl;
  cout << i << " " << j << " " << k << endl;
  cout << setiosflags(ios::fixed) << setprecision(7);
  cout << setw(12) << A1mm_prev
       << setw(12) << A1pm_prev
       << setw(12) << A1mp_prev
       << setw(12) << A2mm_prev
       << setw(12) << A2pm_prev
       << setw(12) << A2mp_prev
       << setw(12) << A3mm_prev
       << setw(12) << A3pm_prev
       << setw(12) << A3mp_prev << endl;
  cout << endl;
  */

  if (option3 == 1)
  {
    double a, delta1, delta2, delta3;
    double A1pp_bar, A2pp_bar, A3pp_bar;
    double lambda;
    
    a = -1.0/3.0;
    
    //The first option, which uses only two
    //of the other sides in the average.

    //Find the corrections that must be made.

    delta1 = -(A2pm - A2pm_prev) + (A1mp - A1mp_prev);
    delta2 = -(A3pm - A3pm_prev) + (A2mp - A2mp_prev);
    delta3 = -(A1pm - A1pm_prev) + (A3mp - A3mp_prev);

    //Make the corrections.

    A1pp += -a*delta3 + a*delta1;
    A2pp += -a*delta1 + a*delta2;
    A3pp += -a*delta2 + a*delta3;

    //Ensure that the sides that should remain
    //fixed do indeed remain fixed.

    A1mm = A1mm_prev;
    A1pm = A1pm_prev;
    A1mp = A1mp_prev;
    A2mm = A2mm_prev;
    A2pm = A2pm_prev;
    A2mp = A2mp_prev;
    A3mm = A3mm_prev;
    A3pm = A3pm_prev;
    A3mp = A3mp_prev;

    //Minimize the differences between values
    //of A within the cell.

    A1pp_bar = 0.5*(A1pm + A1mp);
    A2pp_bar = 0.5*(A2pm + A2mp);
    A3pp_bar = 0.5*(A3pm + A3mp);

    lambda = -1.0/3.0*((A1pp-A1pp_bar) + (A2pp-A2pp_bar) + (A3pp-A3pp_bar));

    A1pp += lambda;
    A2pp += lambda;
    A3pp += lambda;
  }
  else if (option3 == 2)
  {
    double a, delta1, delta2, delta3;
    double A1pp_bar, A2pp_bar, A3pp_bar;
    double lambda;
    
    a = -1.0/3.0;
    
    //The second option, which uses all three
    //of the other sides in the average.

    //Find the corrections that must be made.

    delta1 = -(A2pm - A2pm_prev) + (A1mp - A1mp_prev);
    delta2 = -(A3pm - A3pm_prev) + (A2mp - A2mp_prev);
    delta3 = -(A1pm - A1pm_prev) + (A3mp - A3mp_prev);

    //Make the corrections.

    A1pp += -a*delta3 + a*delta1;
    A2pp += -a*delta1 + a*delta2;
    A3pp += -a*delta2 + a*delta3;

    //Ensure that the sides that should remain
    //fixed do indeed remain fixed.

    A1mm = A1mm_prev;
    A1pm = A1pm_prev;
    A1mp = A1mp_prev;
    A2mm = A2mm_prev;
    A2pm = A2pm_prev;
    A2mp = A2mp_prev;
    A3mm = A3mm_prev;
    A3pm = A3pm_prev;
    A3mp = A3mp_prev;

    //Minimize the differences between values
    //of A within the cell.

    A1pp_bar = 1.0/3.0*(A1mm + A1pm + A1mp);
    A2pp_bar = 1.0/3.0*(A2mm + A2pm + A2mp);
    A3pp_bar = 1.0/3.0*(A3mm + A3pm + A3mp);

    lambda = -1.0/3.0*((A1pp-A1pp_bar) + (A2pp-A2pp_bar) + (A3pp-A3pp_bar));

    A1pp += lambda;
    A2pp += lambda;
    A3pp += lambda;
  }
  else if (option3 == 3)
  {
    double C;

    //The third option, which sets the unknown
    //side equal to the previous value along
    //that direction. For example, Axpp is set
    //equal to Axpp from the previous cell 
    //along the x-axis. Then the value is
    //corrected to agree with the values of B.

    //Ensure that the sides that should remain
    //fixed do indeed remain fixed.

    A1mm = A1mm_prev;
    A1pm = A1pm_prev;
    A1mp = A1mp_prev;
    A2mm = A2mm_prev;
    A2pm = A2pm_prev;
    A2mp = A2mp_prev;
    A3mm = A3mm_prev;
    A3pm = A3pm_prev;
    A3mp = A3mp_prev;

    //Set the new values equal to the
    //previous values.

    A1pp = A1pp_prev;
    A2pp = A2pp_prev;
    A3pp = A3pp_prev;

    //Find the correction that needs to be
    //made so that B=curl(A)

    C = 1.0/3.0*(step*B1p-(A3pp-A3pm)+(A2pp-A2mp));
      + 1.0/3.0*(step*B2p-(A1pp-A1pm)+(A3pp-A3mp))
      + 1.0/3.0*(step*B3p-(A2pp-A2pm)+(A1pp-A1mp));

    //Correct the values

    A1pp += C;
    A2pp += C;
    A3pp += C;
  }
  else if (option3 == 4)
  {
    double a, delta1, delta2, delta3;
    double A1pp_bar, A2pp_bar, A3pp_bar;
    double lambda;
    
    a = -1.0/3.0;
    
    //The fourth option, which uses the previous
    //value along that direction to correct the
    //unknown side.

    //Find the corrections that must be made.

    delta1 = -(A2pm - A2pm_prev) + (A1mp - A1mp_prev);
    delta2 = -(A3pm - A3pm_prev) + (A2mp - A2mp_prev);
    delta3 = -(A1pm - A1pm_prev) + (A3mp - A3mp_prev);

    //Make the corrections.

    A1pp += -a*delta3 + a*delta1;
    A2pp += -a*delta1 + a*delta2;
    A3pp += -a*delta2 + a*delta3;

    //Ensure that the sides that should remain
    //fixed do indeed remain fixed.

    A1mm = A1mm_prev;
    A1pm = A1pm_prev;
    A1mp = A1mp_prev;
    A2mm = A2mm_prev;
    A2pm = A2pm_prev;
    A2mp = A2mp_prev;
    A3mm = A3mm_prev;
    A3pm = A3pm_prev;
    A3mp = A3mp_prev;

    //Minimize the differences between values
    //of A within the cell.

    lambda = -1.0/3.0*((A1pp-A1pp_prev) + (A2pp-A2pp_prev) + (A3pp-A3pp_prev));

    A1pp += lambda;
    A2pp += lambda;
    A3pp += lambda;
  }
  else if (option3 == 5)
  {
    double a, delta1, delta2, delta3;
    double A1pp_bar, A2pp_bar, A3pp_bar;
    double lambda;
    
    a = -1.0/3.0;
    
    //The fifth option, which uses all three
    //of the other sides and the previous
    //value in the average.

    //Find the corrections that must be made.

    delta1 = -(A2pm - A2pm_prev) + (A1mp - A1mp_prev);
    delta2 = -(A3pm - A3pm_prev) + (A2mp - A2mp_prev);
    delta3 = -(A1pm - A1pm_prev) + (A3mp - A3mp_prev);

    //Make the corrections.

    A1pp += -a*delta3 + a*delta1;
    A2pp += -a*delta1 + a*delta2;
    A3pp += -a*delta2 + a*delta3;

    //Ensure that the sides that should remain
    //fixed do indeed remain fixed.

    A1mm = A1mm_prev;
    A1pm = A1pm_prev;
    A1mp = A1mp_prev;
    A2mm = A2mm_prev;
    A2pm = A2pm_prev;
    A2mp = A2mp_prev;
    A3mm = A3mm_prev;
    A3pm = A3pm_prev;
    A3mp = A3mp_prev;

    //Minimize the differences between values
    //of A within the cell.

    A1pp_bar = 1.0/4.0*(A1mm + A1pm + A1mp + A1pp_prev);
    A2pp_bar = 1.0/4.0*(A2mm + A2pm + A2mp + A2pp_prev);
    A3pp_bar = 1.0/4.0*(A3mm + A3pm + A3mp + A3pp_prev);

    lambda = -1.0/3.0*((A1pp-A1pp_bar) + (A2pp-A2pp_bar) + (A3pp-A3pp_bar));

    A1pp += lambda;
    A2pp += lambda;
    A3pp += lambda;
  }
  else if (option3 == 6)
  {
    double a, delta1, delta2, delta3;
    double A1pp_bar, A2pp_bar, A3pp_bar;
    double lambda;
    
    a = -1.0/3.0;
    
    //The sixth option, which uses only two
    //of the other sides and the previous
    //value in the average.

    //Find the corrections that must be made.

    delta1 = -(A2pm - A2pm_prev) + (A1mp - A1mp_prev);
    delta2 = -(A3pm - A3pm_prev) + (A2mp - A2mp_prev);
    delta3 = -(A1pm - A1pm_prev) + (A3mp - A3mp_prev);

    //Make the corrections.

    A1pp += -a*delta3 + a*delta1;
    A2pp += -a*delta1 + a*delta2;
    A3pp += -a*delta2 + a*delta3;

    //Ensure that the sides that should remain
    //fixed do indeed remain fixed.

    A1mm = A1mm_prev;
    A1pm = A1pm_prev;
    A1mp = A1mp_prev;
    A2mm = A2mm_prev;
    A2pm = A2pm_prev;
    A2mp = A2mp_prev;
    A3mm = A3mm_prev;
    A3pm = A3pm_prev;
    A3mp = A3mp_prev;

    //Minimize the differences between values
    //of A within the cell.

    A1pp_bar = 1.0/3.0*(A1pm + A1mp + A1pp_prev);
    A2pp_bar = 1.0/3.0*(A2pm + A2mp + A2pp_prev);
    A3pp_bar = 1.0/3.0*(A3pm + A3mp + A3pp_prev);

    lambda = -1.0/3.0*((A1pp-A1pp_bar) + (A2pp-A2pp_bar) + (A3pp-A3pp_bar));

    A1pp += lambda;
    A2pp += lambda;
    A3pp += lambda;
  }
  else
  {
    double a, delta1, delta2, delta3;
    double A1pp_bar_5, A2pp_bar_5, A3pp_bar_5;
    double A1pp_bar_6, A2pp_bar_6, A3pp_bar_6;
    double lambda_5, lambda_6;
    double A1pp_5, A2pp_5, A3pp_5;
    double A1pp_6, A2pp_6, A3pp_6;
    
    a = -1.0/3.0;
    
    //Linear combination of the fifth option
    //and the sixth option. 

    //Find the corrections that must be made.

    delta1 = -(A2pm - A2pm_prev) + (A1mp - A1mp_prev);
    delta2 = -(A3pm - A3pm_prev) + (A2mp - A2mp_prev);
    delta3 = -(A1pm - A1pm_prev) + (A3mp - A3mp_prev);

    //Make the corrections.

    A1pp += -a*delta3 + a*delta1;
    A2pp += -a*delta1 + a*delta2;
    A3pp += -a*delta2 + a*delta3;

    //Ensure that the sides that should remain
    //fixed do indeed remain fixed.

    A1mm = A1mm_prev;
    A1pm = A1pm_prev;
    A1mp = A1mp_prev;
    A2mm = A2mm_prev;
    A2pm = A2pm_prev;
    A2mp = A2mp_prev;
    A3mm = A3mm_prev;
    A3pm = A3pm_prev;
    A3mp = A3mp_prev;

    //Minimize the differences between values
    //of A within the cell using option 5

    A1pp_bar_5 = 1.0/4.0*(A1mm + A1pm + A1mp + A1pp_prev);
    A2pp_bar_5 = 1.0/4.0*(A2mm + A2pm + A2mp + A2pp_prev);
    A3pp_bar_5 = 1.0/4.0*(A3mm + A3pm + A3mp + A3pp_prev);

    lambda_5 = -1.0/3.0*((A1pp-A1pp_bar_5) + (A2pp-A2pp_bar_5) + (A3pp-A3pp_bar_5));

    A1pp_5 = A1pp + lambda_5;
    A2pp_5 = A2pp + lambda_5;
    A3pp_5 = A3pp + lambda_5;

    //Minimize the differences between values
    //of A within the cell using option 6

    A1pp_bar_6 = 1.0/3.0*(A1pm + A1mp + A1pp_prev);
    A2pp_bar_6 = 1.0/3.0*(A2pm + A2mp + A2pp_prev);
    A3pp_bar_6 = 1.0/3.0*(A3pm + A3mp + A3pp_prev);

    lambda_6 = -1.0/3.0*((A1pp-A1pp_bar_6) + (A2pp-A2pp_bar_6) + (A3pp-A3pp_bar_6));

    A1pp_6 = A1pp + lambda_6;
    A2pp_6 = A2pp + lambda_6;
    A3pp_6 = A3pp + lambda_6;

    //Now take the linear combination of these two
    //solutions, using the parameter set at the 
    //top of the code.

    A1pp = (1.0-comb3)*A1pp_5 + comb3*A1pp_6;
    A2pp = (1.0-comb3)*A2pp_5 + comb3*A2pp_6;
    A3pp = (1.0-comb3)*A3pp_5 + comb3*A3pp_6;
  }

  return;
}

void master (int i,int j,int k,string dir,string sign, 
	     double* Bx,double* By,double* Bz,
	     double* Ax,double* Ay,double* Az)
{
  //This function calculates the 12 values of A for a cell,
  //given the 6 values of B, the location in the grid, and
  //the stage of the calculation (first cell, coordinate
  //rays, coordinate planes, or bulk)

  int bxpindex, bxmindex, bypindex, bymindex, bzpindex, bzmindex;
  int axppindex, axpmindex, axmpindex, axmmindex;
  int ayppindex, aypmindex, aympindex, aymmindex;
  int azppindex, azpmindex, azmpindex, azmmindex;

  double Bxp, Bxm, Byp, Bym, Bzp, Bzm;

  double Axpp, Axpm, Axmp, Axmm;
  double Aypp, Aypm, Aymp, Aymm;
  double Azpp, Azpm, Azmp, Azmm;

  int axmmindex_x, axpmindex_x, axmpindex_x, axppindex_x;
  int aymmindex_x, aypmindex_x, aympindex_x, ayppindex_x;
  int azmmindex_x, azpmindex_x, azmpindex_x, azppindex_x;

  int axmmindex_y, axpmindex_y, axmpindex_y, axppindex_y;
  int aymmindex_y, aypmindex_y, aympindex_y, ayppindex_y;
  int azmmindex_y, azpmindex_y, azmpindex_y, azppindex_y;

  int axmmindex_z, axpmindex_z, axmpindex_z, axppindex_z;
  int aymmindex_z, aypmindex_z, aympindex_z, ayppindex_z;
  int azmmindex_z, azpmindex_z, azmpindex_z, azppindex_z;

  int axmmindex_prev, axpmindex_prev, axmpindex_prev, axppindex_prev;
  int aymmindex_prev, aypmindex_prev, aympindex_prev, ayppindex_prev;
  int azmmindex_prev, azpmindex_prev, azmpindex_prev, azppindex_prev;

  double Axmm_prev, Axpm_prev, Axmp_prev, Axpp_prev;
  double Aymm_prev, Aypm_prev, Aymp_prev, Aypp_prev;
  double Azmm_prev, Azpm_prev, Azmp_prev, Azpp_prev;
  
  bool BcurlA;

  /*
  if (dir.length() != sign.length())
  {
    cout << "Error: Need same number of directions and signs." << endl;
    return;
  }
  */

  bxmindex = k+(Nz)*(j+(Ny)*i);
  bxpindex = k+(Nz)*(j+(Ny)*(i+1));

  bymindex = k+(Nz)*(j+(Ny+1)*i);
  bypindex = k+(Nz)*((j+1)+(Ny+1)*i);

  bzmindex = k+(Nz+1)*(j+(Ny)*i);
  bzpindex = (k+1)+(Nz+1)*(j+(Ny)*i);

  Bxm = Bx[bxmindex];
  Bxp = Bx[bxpindex];

  Bym = By[bymindex];
  Byp = By[bypindex];

  Bzm = Bz[bzmindex];
  Bzp = Bz[bzpindex];


  axmmindex = k+(Nz+1)*(j+(Ny+1)*i);
  axpmindex = k+(Nz+1)*((j+1)+(Ny+1)*i);
  axmpindex = (k+1)+(Nz+1)*(j+(Ny+1)*i);
  axppindex = (k+1)+(Nz+1)*((j+1)+(Ny+1)*i);

  aymmindex = k+(Nz+1)*(j+(Ny)*i);
  aypmindex = (k+1)+(Nz+1)*(j+(Ny)*i);
  aympindex = k+(Nz+1)*(j+(Ny)*(i+1));
  ayppindex = (k+1)+(Nz+1)*(j+(Ny)*(i+1));

  azmmindex = k+(Nz)*(j+(Ny+1)*i);
  azpmindex = k+(Nz)*(j+(Ny+1)*(i+1));
  azmpindex = k+(Nz)*((j+1)+(Ny+1)*i);
  azppindex = k+(Nz)*((j+1)+(Ny+1)*(i+1));


  if (dir == "x")
  {
    if (sign == "-")
    {
      //Moving in the negative x-direction. Need to 
      //rotate the cell 180 degrees around the z-axis
      //in order to use calcA5faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      aymmindex_x = k+(Nz+1)*(j+(Ny)*(i+1));
      azmpindex_x = k+(Nz)*((j+1)+(Ny+1)*(i+1));
      aypmindex_x = (k+1)+(Nz+1)*(j+(Ny)*(i+1));
      azmmindex_x = k+(Nz)*(j+(Ny+1)*(i+1));

      //Use these indices for the indices of the
      //previous values.

      aympindex_prev = aymmindex_x;
      azppindex_prev = azmpindex_x;
      ayppindex_prev = aypmindex_x;
      azpmindex_prev = azmmindex_x;

      //These are the previous values, the values
      //that must remain fixed.

      Aymp_prev = Ay[aympindex_prev];
      Azpp_prev = Az[azppindex_prev];
      Aypp_prev = Ay[ayppindex_prev];
      Azpm_prev = Az[azpmindex_prev];
      
      //Calculate A for this cell.

      calcA6faces(Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Bxm,Bxp,Bym,Byp,Bzm,Bzp,
		  stepx,stepy,stepz);

      Axpm = -Ax[axpmindex];
      Axmm = -Ax[axmmindex];
      Axpp = -Ax[axppindex];
      Axmp = -Ax[axmpindex];
      Aymp = -Ay[aympindex];
      Aypp = -Ay[ayppindex];
      Aymm = -Ay[aymmindex];
      Aypm = -Ay[aypmindex];
      Azpp =  Az[azppindex];
      Azmp =  Az[azmpindex];
      Azpm =  Az[azpmindex];
      Azmm =  Az[azmmindex];

      calcA5faces(Axpm,Axmm,Axpp,Axmp,
		  Aymp,Aypp,Aymm,Aypm,
		  Azpp,Azmp,Azpm,Azmm,
		  -Bxp,-Bxm,-Byp,-Bym,Bzm,Bzp,
		  -Aymp_prev,Azpp_prev,-Aypp_prev,Azpm_prev);

      Ax[axpmindex] = -Axpm;
      Ax[axmmindex] = -Axmm;
      Ax[axppindex] = -Axpp;
      Ax[axmpindex] = -Axmp;
      Ay[aympindex] = -Aymp;
      Ay[ayppindex] = -Aypp;
      Ay[aymmindex] = -Aymm;
      Ay[aypmindex] = -Aypm;
      Az[azppindex] =  Azpp;
      Az[azmpindex] =  Azmp;
      Az[azpmindex] =  Azpm;
      Az[azmmindex] =  Azmm;
    }
    else if (sign == "+")
    {
      //Moving in the positive x-direction. Do not need
      //to do anything special to use calcA5faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      aympindex_x = k+(Nz+1)*(j+(Ny)*(i+1-1));
      azpmindex_x = k+(Nz)*(j+(Ny+1)*(i+1-1));
      ayppindex_x = (k+1)+(Nz+1)*(j+(Ny)*(i+1-1));
      azppindex_x = k+(Nz)*((j+1)+(Ny+1)*(i+1-1));

      //Use these indices for the indices of the
      //previous values.

      aymmindex_prev = aympindex_x;
      azmmindex_prev = azpmindex_x;
      aypmindex_prev = ayppindex_x;
      azmpindex_prev = azppindex_x;

      //These are the previous values, the values
      //that must remain fixed.

      Aymm_prev = Ay[aymmindex_prev];
      Azmm_prev = Az[azmmindex_prev];
      Aypm_prev = Ay[aypmindex_prev];
      Azmp_prev = Az[azmpindex_prev];
      
      //Calculate A for this cell.

      calcA6faces(Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Bxm,Bxp,Bym,Byp,Bzm,Bzp,
		  stepx,stepy,stepz);
      
      calcA5faces(Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Bxm,Bxp,Bym,Byp,Bzm,Bzp,
		  Aymm_prev,Azmm_prev,Aypm_prev,Azmp_prev);
    }
  }
  else if (dir == "y")
  {
    if (sign == "-")
    {
      //Moving in the negative y-direction. Need to 
      //rotate the cell 180 degrees around the x-axis
      //in order to use calcA5faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      azmmindex_y = k+(Nz)*((j+1)+(Ny+1)*i);
      axmpindex_y = (k+1)+(Nz+1)*((j+1)+(Ny+1)*i);
      azpmindex_y = k+(Nz)*((j+1)+(Ny+1)*(i+1));
      axmmindex_y = k+(Nz+1)*((j+1)+(Ny+1)*i);

      //Use these indices for the indices of the
      //previous values.

      azmpindex_prev = azmmindex_y;
      axppindex_prev = axmpindex_y;
      azppindex_prev = azpmindex_y;
      axpmindex_prev = axmmindex_y;

      //These are the previous values, the values
      //that must remain fixed.

      Azmp_prev = Az[azmpindex_prev];
      Axpp_prev = Ax[axppindex_prev];
      Azpp_prev = Az[azppindex_prev];
      Axpm_prev = Ax[axpmindex_prev];
 
      //Calculate A for this cell.

      calcA6faces(Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Bym,Byp,Bzm,Bzp,Bxm,Bxp,
		  stepy,stepz,stepx);

      Aypm = -Ay[aypmindex];
      Aymm = -Ay[aymmindex];
      Aypp = -Ay[ayppindex];
      Aymp = -Ay[aympindex];
      Azmp = -Az[azmpindex];
      Azpp = -Az[azppindex];
      Azmm = -Az[azmmindex];
      Azpm = -Az[azpmindex];
      Axpp =  Ax[axppindex];
      Axmp =  Ax[axmpindex];
      Axpm =  Ax[axpmindex];
      Axmm =  Ax[axmmindex];

      calcA5faces(Aypm,Aymm,Aypp,Aymp,
		  Azmp,Azpp,Azmm,Azpm,
		  Axpp,Axmp,Axpm,Axmm,
		  -Byp,-Bym,-Bzp,-Bzm,Bxm,Bxp,
		  -Azmp_prev,Axpp_prev,-Azpp_prev,Axpm_prev);

      Ay[aypmindex] = -Aypm;
      Ay[aymmindex] = -Aymm;
      Ay[ayppindex] = -Aypp;
      Ay[aympindex] = -Aymp;
      Az[azmpindex] = -Azmp;
      Az[azppindex] = -Azpp;
      Az[azmmindex] = -Azmm;
      Az[azpmindex] = -Azpm;
      Ax[axppindex] =  Axpp;
      Ax[axmpindex] =  Axmp;
      Ax[axpmindex] =  Axpm;
      Ax[axmmindex] =  Axmm;
    }
    else if (sign == "+")
    {
      //Moving in the positive y-direction. Do not need
      //to do anything special to use calcA5faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      azmpindex_y = k+(Nz)*((j+1-1)+(Ny+1)*i);
      axpmindex_y = k+(Nz+1)*((j+1-1)+(Ny+1)*i);
      azppindex_y = k+(Nz)*((j+1-1)+(Ny+1)*(i+1));
      axppindex_y = (k+1)+(Nz+1)*((j+1-1)+(Ny+1)*i);

      //Use these indices for the indices of the
      //previous values.

      azmmindex_prev = azmpindex_y;
      axmmindex_prev = axpmindex_y;
      azpmindex_prev = azppindex_y;
      axmpindex_prev = axppindex_y;

      //These are the previous values, the values
      //that must remain fixed.

      Azmm_prev = Az[azmmindex_prev];
      Axmm_prev = Ax[axmmindex_prev];
      Azpm_prev = Az[azpmindex_prev];
      Axmp_prev = Ax[axmpindex_prev];

      //Calculate A for this cell.

      calcA6faces(Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Bym,Byp,Bzm,Bzp,Bxm,Bxp,
		  stepy,stepz,stepx);

      calcA5faces(Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Bym,Byp,Bzm,Bzp,Bxm,Bxp,
		  Azmm_prev,Axmm_prev,Azpm_prev,Axmp_prev);
    }
  }
  else if (dir == "z")
  {
    if (sign == "-")
    {
      //Moving in the negative z-direction. Need to 
      //rotate the cell 180 degrees around the y-axis
      //in order to use calcA5faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      axmmindex_z = (k+1)+(Nz+1)*(j+(Ny+1)*i);
      aympindex_z = (k+1)+(Nz+1)*(j+(Ny)*(i+1));
      axpmindex_z = (k+1)+(Nz+1)*((j+1)+(Ny+1)*i);
      aymmindex_z = (k+1)+(Nz+1)*(j+(Ny)*i);

      //Use these indices for the indices of the
      //previous values.

      axmpindex_prev = axmmindex_z;
      ayppindex_prev = aympindex_z;
      axppindex_prev = axpmindex_z;
      aypmindex_prev = aymmindex_z;

      //These are the previous values, the values
      //that must remain fixed.

      Axmp_prev = Ax[axmpindex_prev];
      Aypp_prev = Ay[ayppindex_prev];
      Axpp_prev = Ax[axppindex_prev];
      Aypm_prev = Ay[aypmindex_prev];

      //Calculate A for this cell.

      calcA6faces(Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Bzm,Bzp,Bxm,Bxp,Bym,Byp,
		  stepz,stepx,stepy);

      Azpm = -Az[azpmindex];
      Azmm = -Az[azmmindex];
      Azpp = -Az[azppindex];
      Azmp = -Az[azmpindex];
      Axmp = -Ax[axmpindex];
      Axpp = -Ax[axppindex];
      Axmm = -Ax[axmmindex];
      Axpm = -Ax[axpmindex];
      Aypp =  Ay[ayppindex];
      Aymp =  Ay[aympindex];
      Aypm =  Ay[aypmindex];
      Aymm =  Ay[aymmindex];

      calcA5faces(Azpm,Azmm,Azpp,Azmp,
		  Axmp,Axpp,Axmm,Axpm,
		  Aypp,Aymp,Aypm,Aymm,
		  -Bzp,-Bzm,-Bxp,-Bxm,Bym,Byp,
		  -Axmp_prev,Aypp_prev,-Axpp_prev,Aypm_prev);

      Az[azpmindex] = -Azpm;
      Az[azmmindex] = -Azmm;
      Az[azppindex] = -Azpp;
      Az[azmpindex] = -Azmp;
      Ax[axmpindex] = -Axmp;
      Ax[axppindex] = -Axpp;
      Ax[axmmindex] = -Axmm;
      Ax[axpmindex] = -Axpm;
      Ay[ayppindex] =  Aypp;
      Ay[aympindex] =  Aymp;
      Ay[aypmindex] =  Aypm;
      Ay[aymmindex] =  Aymm;
    }
    else if (sign == "+")
    {
      //Moving in the positive z-direction. Do not need
      //to do anything special to use calcA5faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      axmpindex_z = (k+1-1)+(Nz+1)*(j+(Ny+1)*i);
      aypmindex_z = (k+1-1)+(Nz+1)*(j+(Ny)*i);
      axppindex_z = (k+1-1)+(Nz+1)*((j+1)+(Ny+1)*i);
      ayppindex_z = (k+1-1)+(Nz+1)*(j+(Ny)*(i+1));

      //Use these indices for the indices of the
      //previous values.

      axmmindex_prev = axmpindex_z;
      aymmindex_prev = aypmindex_z;
      axpmindex_prev = axppindex_z;
      aympindex_prev = ayppindex_z;

      //These are the previous values, the values
      //that must remain fixed.

      Axmm_prev = Ax[axmmindex_prev];
      Aymm_prev = Ay[aymmindex_prev];
      Axpm_prev = Ax[axpmindex_prev];
      Aymp_prev = Ay[aympindex_prev];

      //Calculate A for this cell.

      calcA6faces(Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Bzm,Bzp,Bxm,Bxp,Bym,Byp,
		  stepz,stepx,stepy);

      calcA5faces(Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Bzm,Bzp,Bxm,Bxp,Bym,Byp,
		  Axmm_prev,Aymm_prev,Axpm_prev,Aymp_prev);
    }
  }


  else if (dir == "xy")
  {
    if (sign == "--")
    {
      //Moving in the negative x-direction and the
      //negative y-direction. Need to rotate the cell
      //180 degrees around the z-axis in order to use
      //calcA4faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      axmpindex_y = (k+1)+(Nz+1)*((j+1)+(Ny+1)*i);
      aypmindex_x = (k+1)+(Nz+1)*(j+(Ny)*(i+1));
      azmmindex_x = k+(Nz)*(j+(Ny+1)*(i+1));
      aymmindex_x = k+(Nz+1)*(j+(Ny)*(i+1));
      axmmindex_y = k+(Nz+1)*((j+1)+(Ny+1)*i);
      azmmindex_y = k+(Nz)*((j+1)+(Ny+1)*i);
      azpmindex_y = k+(Nz)*((j+1)+(Ny+1)*(i+1));

      //Use these indices for the indices of the
      //previous values.

      axppindex_prev = axmpindex_y;
      ayppindex_prev = aypmindex_x;
      azpmindex_prev = azmmindex_x;
      aympindex_prev = aymmindex_x;
      axpmindex_prev = axmmindex_y;
      azmpindex_prev = azmmindex_y;
      azppindex_prev = azpmindex_y;

      //These are the previous values, the values
      //that must remain fixed.

      Axpp_prev = Ax[axppindex_prev];
      Aypp_prev = Ay[ayppindex_prev];
      Azpm_prev = Az[azpmindex_prev];
      Aymp_prev = Ay[aympindex_prev];
      Axpm_prev = Ax[axpmindex_prev];
      Azmp_prev = Az[azmpindex_prev];
      Azpp_prev = Az[azppindex_prev];

      //Additional information is needed to
      //help minimize changes. This comes in
      //the form of more previously set A
      //values.

      axmmindex_x = k+(Nz+1)*(j+(Ny+1)*(i+1));
      axmpindex_x = (k+1)+(Nz+1)*(j+(Ny+1)*(i+1));
      aymmindex_y = k+(Nz+1)*((j+1)+(Ny)*i);
      aypmindex_y = (k+1)+(Nz+1)*((j+1)+(Ny)*i);

      Axmm_prev = Ax[axmmindex_x];
      Axmp_prev = Ax[axmpindex_x];
      Aymm_prev = Ay[aymmindex_y];
      Aypm_prev = Ay[aypmindex_y];

      //Calculate A for this cell.

      calcA6faces(Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Bxm,Bxp,Bym,Byp,Bzm,Bzp,
		  stepx,stepy,stepz);

      Axmm = -Ax[axmmindex];
      Axpm = -Ax[axpmindex];
      Axmp = -Ax[axmpindex];
      Axpp = -Ax[axppindex];
      Aymm = -Ay[aymmindex];
      Aypm = -Ay[aypmindex];
      Aymp = -Ay[aympindex];
      Aypp = -Ay[ayppindex];
      Azmm =  Az[azmmindex];
      Azpm =  Az[azpmindex];
      Azmp =  Az[azmpindex];
      Azpp =  Az[azppindex];

      calcA4faces(Axpm,Axmm,Axpp,Axmp,
		  Aymp,Aypp,Aymm,Aypm,
		  Azpp,Azmp,Azpm,Azmm,
		  -Bxp,-Bxm,-Byp,-Bym,Bzm,Bzp,
		  -Axpp_prev,-Aypp_prev, Azpm_prev,
		  -Aymp_prev,-Axpm_prev, Azmp_prev, Azpp_prev,
		  -Axmm_prev,-Axmp_prev,-Aymm_prev,-Aypm_prev);

      Ax[axmmindex] = -Axmm;
      Ax[axpmindex] = -Axpm;
      Ax[axmpindex] = -Axmp;
      Ax[axppindex] = -Axpp;
      Ay[aymmindex] = -Aymm;
      Ay[aypmindex] = -Aypm;
      Ay[aympindex] = -Aymp;
      Ay[ayppindex] = -Aypp;
      Az[azmmindex] =  Azmm;
      Az[azpmindex] =  Azpm;
      Az[azmpindex] =  Azmp;
      Az[azppindex] =  Azpp;
    }
    else if (sign == "+-")
    {
      //Moving in the positive x-direction and the
      //negative y-direction. Need to rotate the cell
      //180 degrees around the x-axis in order to use
      //calcA4faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      axmmindex_y = k+(Nz+1)*((j+1)+(Ny+1)*i);
      aympindex_x = k+(Nz+1)*(j+(Ny)*(i+1-1));
      azpmindex_x = k+(Nz)*(j+(Ny+1)*(i+1-1));
      ayppindex_x = (k+1)+(Nz+1)*(j+(Ny)*(i+1-1));
      axmpindex_y = (k+1)+(Nz+1)*((j+1)+(Ny+1)*i);
      azpmindex_y = k+(Nz)*((j+1)+(Ny+1)*(i+1));
      azmmindex_y = k+(Nz)*((j+1)+(Ny+1)*i);

      //Use these indices for the indices of the
      //previous values.

      axpmindex_prev = axmmindex_y;
      aymmindex_prev = aympindex_x;
      azmmindex_prev = azpmindex_x;
      aypmindex_prev = ayppindex_x;
      axppindex_prev = axmpindex_y;
      azppindex_prev = azpmindex_y;
      azmpindex_prev = azmmindex_y;

      //These are the previous values, the values
      //that must remain fixed.

      Axpm_prev = Ax[axpmindex_prev];
      Aymm_prev = Ay[aymmindex_prev];
      Azmm_prev = Az[azmmindex_prev];
      Aypm_prev = Ay[aypmindex_prev];
      Axpp_prev = Ax[axppindex_prev];
      Azpp_prev = Az[azppindex_prev];
      Azmp_prev = Az[azmpindex_prev];

      //Additional information is needed to
      //help minimize changes. This comes in
      //the form of more previously set A
      //values.

      axmpindex_x = (k+1)+(Nz+1)*(j+(Ny+1)*(i-1));
      axmmindex_x = k+(Nz+1)*(j+(Ny+1)*(i-1));
      ayppindex_y = (k+1)+(Nz+1)*((j+1)+(Ny)*(i+1));
      aympindex_y = k+(Nz+1)*((j+1)+(Ny)*(i+1));

      Axmp_prev = Ax[axmpindex_x];
      Axmm_prev = Ax[axmmindex_x];
      Aypp_prev = Ay[ayppindex_y];
      Aymp_prev = Ay[aympindex_y];

      //Calculate A for this cell.

      calcA6faces(Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Bxm,Bxp,Bym,Byp,Bzm,Bzp,
		  stepx,stepy,stepz);

      Axmm =  Ax[axmmindex];
      Axpm =  Ax[axpmindex];
      Axmp =  Ax[axmpindex];
      Axpp =  Ax[axppindex];
      Aymm = -Ay[aymmindex];
      Aypm = -Ay[aypmindex];
      Aymp = -Ay[aympindex];
      Aypp = -Ay[ayppindex];
      Azmm = -Az[azmmindex];
      Azpm = -Az[azpmindex];
      Azmp = -Az[azmpindex];
      Azpp = -Az[azppindex];

      calcA4faces(Axpp,Axmp,Axpm,Axmm,
		  Aypm,Aymm,Aypp,Aymp,
		  Azmp,Azpp,Azmm,Azpm,
		  Bxm,Bxp,-Byp,-Bym,-Bzp,-Bzm,
		   Axpm_prev,-Aymm_prev,-Azmm_prev,
		  -Aypm_prev, Axpp_prev,-Azpp_prev,-Azmp_prev,
		   Axmp_prev, Axmm_prev,-Aypp_prev,-Aymp_prev);

      Ax[axmmindex] =  Axmm;
      Ax[axpmindex] =  Axpm;
      Ax[axmpindex] =  Axmp;
      Ax[axppindex] =  Axpp;
      Ay[aymmindex] = -Aymm;
      Ay[aypmindex] = -Aypm;
      Ay[aympindex] = -Aymp;
      Ay[ayppindex] = -Aypp;
      Az[azmmindex] = -Azmm;
      Az[azpmindex] = -Azpm;
      Az[azmpindex] = -Azmp;
      Az[azppindex] = -Azpp;
    }
    else if (sign == "-+")
    {
      //Moving in the negative x-direction and the
      //positive y-direction. Need to rotate the cell
      //180 degrees around the y-axis in order to use
      //calcA4faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      axpmindex_y = k+(Nz+1)*((j+1-1)+(Ny+1)*i);
      aymmindex_x = k+(Nz+1)*(j+(Ny)*(i+1));
      azmpindex_x = k+(Nz)*((j+1)+(Ny+1)*(i+1));
      aypmindex_x = (k+1)+(Nz+1)*(j+(Ny)*(i+1));
      axppindex_y = (k+1)+(Nz+1)*((j+1-1)+(Ny+1)*i);
      azmpindex_y = k+(Nz)*((j+1-1)+(Ny+1)*i);
      azppindex_y = k+(Nz)*((j+1-1)+(Ny+1)*(i+1));

      //Use these indices for the indices of the
      //previous values.

      axmmindex_prev = axpmindex_y;
      aympindex_prev = aymmindex_x;
      azppindex_prev = azmpindex_x;
      ayppindex_prev = aypmindex_x;
      axmpindex_prev = axppindex_y;
      azmmindex_prev = azmpindex_y;
      azpmindex_prev = azppindex_y;

      //These are the previous values, the values
      //that must remain fixed.

      Axmm_prev = Ax[axmmindex_prev];
      Aymp_prev = Ay[aympindex_prev];
      Azpp_prev = Az[azppindex_prev];
      Aypp_prev = Ay[ayppindex_prev];
      Axmp_prev = Ax[axmpindex_prev];
      Azmm_prev = Az[azmmindex_prev];
      Azpm_prev = Az[azpmindex_prev];

      //Additional information is needed to
      //help minimize changes. This comes in
      //the form of more previously set A
      //values.

      axppindex_x = (k+1)+(Nz+1)*((j+1)+(Ny+1)*(i+1));
      axpmindex_x = k+(Nz+1)*((j+1)+(Ny+1)*(i+1));
      aypmindex_y = (k+1)+(Nz+1)*((j-1)+(Ny)*i);
      aymmindex_y = k+(Nz+1)*((j-1)+(Ny)*i);

      Axpp_prev = Ax[axppindex_x];
      Axpm_prev = Ax[axpmindex_x];
      Aypm_prev = Ay[aypmindex_y];
      Aymm_prev = Ay[aymmindex_y];

      //Calculate A for this cell.

      calcA6faces(Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Bxm,Bxp,Bym,Byp,Bzm,Bzp,
		  stepx,stepy,stepz);

      Axmm = -Ax[axmmindex];
      Axpm = -Ax[axpmindex];
      Axmp = -Ax[axmpindex];
      Axpp = -Ax[axppindex];
      Aymm =  Ay[aymmindex];
      Aypm =  Ay[aypmindex];
      Aymp =  Ay[aympindex];
      Aypp =  Ay[ayppindex];
      Azmm = -Az[azmmindex];
      Azpm = -Az[azpmindex];
      Azmp = -Az[azmpindex];
      Azpp = -Az[azppindex];

      calcA4faces(Axmp,Axpp,Axmm,Axpm,
		  Aypp,Aymp,Aypm,Aymm,
		  Azpm,Azmm,Azpp,Azmp,
		  -Bxp,-Bxm,Bym,Byp,-Bzp,-Bzm,
		  -Axmm_prev, Aymp_prev,-Azpp_prev,
		   Aypp_prev,-Axmp_prev,-Azmm_prev,-Azpm_prev,
		  -Axpp_prev,-Axpm_prev, Aypm_prev, Aymm_prev);

      Ax[axmmindex] = -Axmm;
      Ax[axpmindex] = -Axpm;
      Ax[axmpindex] = -Axmp;
      Ax[axppindex] = -Axpp;
      Ay[aymmindex] =  Aymm;
      Ay[aypmindex] =  Aypm;
      Ay[aympindex] =  Aymp;
      Ay[ayppindex] =  Aypp;
      Az[azmmindex] = -Azmm;
      Az[azpmindex] = -Azpm;
      Az[azmpindex] = -Azmp;
      Az[azppindex] = -Azpp;
    }
    else if (sign == "++")
    {
      //Moving in the positive x-direction and the
      //positive y-direction. Do not need to do
      //anything special in order to use calcA4faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      axppindex_y = (k+1)+(Nz+1)*((j+1-1)+(Ny+1)*i);
      ayppindex_x = (k+1)+(Nz+1)*(j+(Ny)*(i+1-1));
      azppindex_x = k+(Nz)*((j+1)+(Ny+1)*(i+1-1));
      aympindex_x = k+(Nz+1)*(j+(Ny)*(i+1-1));
      axpmindex_y = k+(Nz+1)*((j+1-1)+(Ny+1)*i);
      azppindex_y = k+(Nz)*((j+1-1)+(Ny+1)*(i+1));
      azmpindex_y = k+(Nz)*((j+1-1)+(Ny+1)*i);

      //Use these indices for the indices of the
      //previous values.

      axmpindex_prev = axppindex_y;
      aypmindex_prev = ayppindex_x;
      azmpindex_prev = azppindex_x;
      aymmindex_prev = aympindex_x;
      axmmindex_prev = axpmindex_y;
      azpmindex_prev = azppindex_y;
      azmmindex_prev = azmpindex_y;

      //These are the previous values, the values
      //that must remain fixed.

      Axmp_prev = Ax[axmpindex_prev];
      Aypm_prev = Ay[aypmindex_prev];
      Azmp_prev = Az[azmpindex_prev];
      Aymm_prev = Ay[aymmindex_prev];
      Axmm_prev = Ax[axmmindex_prev];
      Azpm_prev = Az[azpmindex_prev];
      Azmm_prev = Az[azmmindex_prev];

      //Additional information is needed to
      //help minimize changes. This comes in
      //the form of more previously set A
      //values.

      axpmindex_x = k+(Nz+1)*((j+1)+(Ny+1)*(i-1));
      axppindex_x = (k+1)+(Nz+1)*((j+1)+(Ny+1)*(i-1));
      aympindex_y = k+(Nz+1)*((j-1)+(Ny)*(i+1));
      ayppindex_y = (k+1)+(Nz+1)*((j-1)+(Ny)*(i+1));

      Axpm_prev = Ax[axpmindex_x];
      Axpp_prev = Ax[axppindex_x];
      Aymp_prev = Ay[aympindex_y];
      Aypp_prev = Ay[ayppindex_y];

      //Calculate A for this cell.

      calcA6faces(Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Bxm,Bxp,Bym,Byp,Bzm,Bzp,
		  stepx,stepy,stepz);

      calcA4faces(Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Bxm,Bxp,Bym,Byp,Bzm,Bzp,
		  Axmp_prev,Aypm_prev,Azmp_prev,
		  Aymm_prev,Axmm_prev,Azpm_prev,Azmm_prev,
		  Axpm_prev,Axpp_prev,Aymp_prev,Aypp_prev);
    }
  }
  else if (dir == "yz")
  {
    if (sign == "--")
    {
      //Moving in the negative y-direction and the
      //negative z-direction. Need to rotate the cell
      //180 degrees around the x-axis in order to use
      //calcA4faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      aympindex_z = (k+1)+(Nz+1)*(j+(Ny)*(i+1));
      azpmindex_y = k+(Nz)*((j+1)+(Ny+1)*(i+1));
      axmmindex_y = k+(Nz+1)*((j+1)+(Ny+1)*i);
      azmmindex_y = k+(Nz)*((j+1)+(Ny+1)*i);
      aymmindex_z = (k+1)+(Nz+1)*(j+(Ny)*i);
      axmmindex_z = (k+1)+(Nz+1)*(j+(Ny+1)*i);
      axpmindex_z = (k+1)+(Nz+1)*((j+1)+(Ny+1)*i);

      //Use these indices for the indices of the
      //previous values.

      ayppindex_prev = aympindex_z;
      azppindex_prev = azpmindex_y;
      axpmindex_prev = axmmindex_y;
      azmpindex_prev = azmmindex_y;
      aypmindex_prev = aymmindex_z;
      axmpindex_prev = axmmindex_z;
      axppindex_prev = axpmindex_z;

      //These are the previous values, the values
      //that must remain fixed.

      Aypp_prev = Ay[ayppindex_prev];
      Azpp_prev = Az[azppindex_prev];
      Axpm_prev = Ax[axpmindex_prev];
      Azmp_prev = Az[azmpindex_prev];
      Aypm_prev = Ay[aypmindex_prev];
      Axmp_prev = Ax[axmpindex_prev];
      Axpp_prev = Ax[axppindex_prev];

      //Additional information is needed to
      //help minimize changes. This comes in
      //the form of more previously set A
      //values.

      aymmindex_y = k+(Nz+1)*((j+1)+(Ny)*i);
      aympindex_y = k+(Nz+1)*((j+1)+(Ny)*(i+1));
      azmmindex_z = (k+1)+(Nz)*(j+(Ny+1)*i);
      azpmindex_z = (k+1)+(Nz)*(j+(Ny+1)*(i+1));

      Aymm_prev = Ay[aymmindex_y];
      Aymp_prev = Ay[aympindex_y];
      Azmm_prev = Az[azmmindex_z];
      Azpm_prev = Az[azpmindex_z];

      //Calculate A for this cell.

      calcA6faces(Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Bym,Byp,Bzm,Bzp,Bxm,Bxp,
		  stepy,stepz,stepx);

      Aymm = -Ay[aymmindex];
      Aypm = -Ay[aypmindex];
      Aymp = -Ay[aympindex];
      Aypp = -Ay[ayppindex];
      Azmm = -Az[azmmindex];
      Azpm = -Az[azpmindex];
      Azmp = -Az[azmpindex];
      Azpp = -Az[azppindex];
      Axmm =  Ax[axmmindex];
      Axpm =  Ax[axpmindex];
      Axmp =  Ax[axmpindex];
      Axpp =  Ax[axppindex];

      calcA4faces(Aypm,Aymm,Aypp,Aymp,
		  Azmp,Azpp,Azmm,Azpm,
		  Axpp,Axmp,Axpm,Axmm,
		  -Byp,-Bym,-Bzp,-Bzm,Bxm,Bxp,
		  -Aypp_prev,-Azpp_prev, Axpm_prev,
		  -Azmp_prev,-Aypm_prev, Axmp_prev, Axpp_prev,
		  -Aymm_prev,-Aymp_prev,-Azmm_prev,-Azpm_prev);

      Ay[aymmindex] = -Aymm;
      Ay[aypmindex] = -Aypm;
      Ay[aympindex] = -Aymp;
      Ay[ayppindex] = -Aypp;
      Az[azmmindex] = -Azmm;
      Az[azpmindex] = -Azpm;
      Az[azmpindex] = -Azmp;
      Az[azppindex] = -Azpp;
      Ax[axmmindex] =  Axmm;
      Ax[axpmindex] =  Axpm;
      Ax[axmpindex] =  Axmp;
      Ax[axppindex] =  Axpp;
    }
    else if (sign == "+-")
    {
      //Moving in the positive y-direction and the
      //negative z-direction. Need to rotate the cell
      //180 degrees around the y-axis in order to use
      //calcA4faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      aymmindex_z = (k+1)+(Nz+1)*(j+(Ny)*i);
      azmpindex_y = k+(Nz)*((j+1-1)+(Ny+1)*i);
      axpmindex_y = k+(Nz+1)*((j+1-1)+(Ny+1)*i);
      azppindex_y = k+(Nz)*((j+1-1)+(Ny+1)*(i+1));
      aympindex_z = (k+1)+(Nz+1)*(j+(Ny)*(i+1));
      axpmindex_z = (k+1)+(Nz+1)*((j+1)+(Ny+1)*i);
      axmmindex_z = (k+1)+(Nz+1)*(j+(Ny+1)*i);

      //Use these indices for the indices of the
      //previous values.

      aypmindex_prev = aymmindex_z;
      azmmindex_prev = azmpindex_y;
      axmmindex_prev = axpmindex_y;
      azpmindex_prev = azppindex_y;
      ayppindex_prev = aympindex_z;
      axppindex_prev = axpmindex_z;
      axmpindex_prev = axmmindex_z;

      //These are the previous values, the values
      //that must remain fixed.

      Aypm_prev = Ay[aypmindex_prev];
      Azmm_prev = Az[azmmindex_prev];
      Axmm_prev = Ax[axmmindex_prev];
      Azpm_prev = Az[azpmindex_prev];
      Aypp_prev = Ay[ayppindex_prev];
      Axpp_prev = Ax[axppindex_prev];
      Axmp_prev = Ax[axmpindex_prev];

      //Additional information is needed to
      //help minimize changes. This comes in
      //the form of more previously set A
      //values.

      aympindex_y = k+(Nz+1)*((j-1)+(Ny)*(i+1));
      aymmindex_y = k+(Nz+1)*((j-1)+(Ny)*i);
      azppindex_z = (k+1)+(Nz)*((j+1)+(Ny+1)*(i+1));
      azmpindex_z = (k+1)+(Nz)*((j+1)+(Ny+1)*i);

      Aymp_prev = Ay[aympindex_y];
      Aymm_prev = Ay[aymmindex_y];
      Azpp_prev = Az[azppindex_z];
      Azmp_prev = Az[azmpindex_z];

      //Calculate A for this cell.

      calcA6faces(Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Bym,Byp,Bzm,Bzp,Bxm,Bxp,
		  stepy,stepz,stepx);

      Aymm =  Ay[aymmindex];
      Aypm =  Ay[aypmindex];
      Aymp =  Ay[aympindex];
      Aypp =  Ay[ayppindex];
      Azmm = -Az[azmmindex];
      Azpm = -Az[azpmindex];
      Azmp = -Az[azmpindex];
      Azpp = -Az[azppindex];
      Axmm = -Ax[axmmindex];
      Axpm = -Ax[axpmindex];
      Axmp = -Ax[axmpindex];
      Axpp = -Ax[axppindex];

      calcA4faces(Aypp,Aymp,Aypm,Aymm,
		  Azpm,Azmm,Azpp,Azmp,
		  Axmp,Axpp,Axmm,Axpm,
		  Bym,Byp,-Bzp,-Bzm,-Bxp,-Bxm,
		   Aypm_prev,-Azmm_prev,-Axmm_prev,
		  -Azpm_prev, Aypp_prev,-Axpp_prev,-Axmp_prev,
		   Aymp_prev, Aymm_prev,-Azpp_prev,-Azmp_prev);

      Ay[aymmindex] =  Aymm;
      Ay[aypmindex] =  Aypm;
      Ay[aympindex] =  Aymp;
      Ay[ayppindex] =  Aypp;
      Az[azmmindex] = -Azmm;
      Az[azpmindex] = -Azpm;
      Az[azmpindex] = -Azmp;
      Az[azppindex] = -Azpp;
      Ax[axmmindex] = -Axmm;
      Ax[axpmindex] = -Axpm;
      Ax[axmpindex] = -Axmp;
      Ax[axppindex] = -Axpp;
    }
    else if (sign == "-+")
    {
      //Moving in the negative y-direction and the
      //positive z-direction. Need to rotate the cell
      //180 degrees around the z-axis in order to use
      //calcA4faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      aypmindex_z = (k+1-1)+(Nz+1)*(j+(Ny)*i);
      azmmindex_y = k+(Nz)*((j+1)+(Ny+1)*i);
      axmpindex_y = (k+1)+(Nz+1)*((j+1)+(Ny+1)*i);
      azpmindex_y = k+(Nz)*((j+1)+(Ny+1)*(i+1));
      ayppindex_z = (k+1-1)+(Nz+1)*(j+(Ny)*(i+1));
      axmpindex_z = (k+1-1)+(Nz+1)*(j+(Ny+1)*i);
      axppindex_z = (k+1-1)+(Nz+1)*((j+1)+(Ny+1)*i);

      //Use these indices for the indices of the
      //previous values.

      aymmindex_prev = aypmindex_z;
      azmpindex_prev = azmmindex_y;
      axppindex_prev = axmpindex_y;
      azppindex_prev = azpmindex_y;
      aympindex_prev = ayppindex_z;
      axmmindex_prev = axmpindex_z;
      axpmindex_prev = axppindex_z;

      //These are the previous values, the values
      //that must remain fixed.

      Aymm_prev = Ay[aymmindex_prev];
      Azmp_prev = Az[azmpindex_prev];
      Axpp_prev = Ax[axppindex_prev];
      Azpp_prev = Az[azppindex_prev];
      Aymp_prev = Ay[aympindex_prev];
      Axmm_prev = Ax[axmmindex_prev];
      Axpm_prev = Ax[axpmindex_prev];

      //Additional information is needed to
      //help minimize changes. This comes in
      //the form of more previously set A
      //values.

      ayppindex_y = (k+1)+(Nz+1)*((j+1)+(Ny)*(i+1));
      aypmindex_y = (k+1)+(Nz+1)*((j+1)+(Ny)*i);
      azpmindex_z = (k-1)+(Nz)*(j+(Ny+1)*(i+1));
      azmmindex_z = (k-1)+(Nz)*(j+(Ny+1)*i);

      Aypp_prev = Ay[ayppindex_y];
      Aypm_prev = Ay[aypmindex_y];
      Azpm_prev = Az[azpmindex_z];
      Azmm_prev = Az[azmmindex_z];

      //Calculate A for this cell.

      calcA6faces(Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Bym,Byp,Bzm,Bzp,Bxm,Bxp,
		  stepy,stepz,stepx);

      Aymm = -Ay[aymmindex];
      Aypm = -Ay[aypmindex];
      Aymp = -Ay[aympindex];
      Aypp = -Ay[ayppindex];
      Azmm =  Az[azmmindex];
      Azpm =  Az[azpmindex];
      Azmp =  Az[azmpindex];
      Azpp =  Az[azppindex];
      Axmm = -Ax[axmmindex];
      Axpm = -Ax[axpmindex];
      Axmp = -Ax[axmpindex];
      Axpp = -Ax[axppindex];

      calcA4faces(Aymp,Aypp,Aymm,Aypm,
		  Azpp,Azmp,Azpm,Azmm,
		  Axpm,Axmm,Axpp,Axmp,
		  -Byp,-Bym,Bzm,Bzp,-Bxp,-Bxm,
		  -Aymm_prev, Azmp_prev,-Axpp_prev,
		   Azpp_prev,-Aymp_prev,-Axmm_prev,-Axpm_prev,
		  -Aypp_prev,-Aypm_prev, Azpm_prev, Azmm_prev);

      Ay[aymmindex] = -Aymm;
      Ay[aypmindex] = -Aypm;
      Ay[aympindex] = -Aymp;
      Ay[ayppindex] = -Aypp;
      Az[azmmindex] =  Azmm;
      Az[azpmindex] =  Azpm;
      Az[azmpindex] =  Azmp;
      Az[azppindex] =  Azpp;
      Ax[axmmindex] = -Axmm;
      Ax[axpmindex] = -Axpm;
      Ax[axmpindex] = -Axmp;
      Ax[axppindex] = -Axpp;
    }
    else if (sign == "++")
    {
      //Moving in the positive y-direction and the
      //positive z-direction. Do not need to do
      //anything special in order to use calcA4faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      ayppindex_z = (k+1-1)+(Nz+1)*(j+(Ny)*(i+1));
      azppindex_y = k+(Nz)*((j+1-1)+(Ny+1)*(i+1));
      axppindex_y = (k+1)+(Nz+1)*((j+1-1)+(Ny+1)*i);
      azmpindex_y = k+(Nz)*((j+1-1)+(Ny+1)*i);
      aypmindex_z = (k+1-1)+(Nz+1)*(j+(Ny)*i);
      axppindex_z = (k+1-1)+(Nz+1)*((j+1)+(Ny+1)*i);
      axmpindex_z = (k+1-1)+(Nz+1)*(j+(Ny+1)*i);

      //Use these indices for the indices of the
      //previous values.

      aympindex_prev = ayppindex_z;
      azpmindex_prev = azppindex_y;
      axmpindex_prev = axppindex_y;
      azmmindex_prev = azmpindex_y;
      aymmindex_prev = aypmindex_z;
      axpmindex_prev = axppindex_z;
      axmmindex_prev = axmpindex_z;

      //These are the previous values, the values
      //that must remain fixed.

      Aymp_prev = Ay[aympindex_prev];
      Azpm_prev = Az[azpmindex_prev];
      Axmp_prev = Ax[axmpindex_prev];
      Azmm_prev = Az[azmmindex_prev];
      Aymm_prev = Ay[aymmindex_prev];
      Axpm_prev = Ax[axpmindex_prev];
      Axmm_prev = Ax[axmmindex_prev];

      //Additional information is needed to
      //help minimize changes. This comes in
      //the form of more previously set A
      //values.

      aypmindex_y = (k+1)+(Nz+1)*((j-1)+(Ny)*i);
      ayppindex_y = (k+1)+(Nz+1)*((j-1)+(Ny)*(i+1));
      azmpindex_z = (k-1)+(Nz)*((j+1)+(Ny+1)*i);
      azppindex_z = (k-1)+(Nz)*((j+1)+(Ny+1)*(i+1));

      Aypm_prev = Ay[aypmindex_y];
      Aypp_prev = Ay[ayppindex_y];
      Azmp_prev = Az[azmpindex_z];
      Azpp_prev = Az[azppindex_z];

      //Calculate A for this cell.

      calcA6faces(Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Bym,Byp,Bzm,Bzp,Bxm,Bxp,
		  stepy,stepz,stepx);

      calcA4faces(Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Bym,Byp,Bzm,Bzp,Bxm,Bxp,
		  Aymp_prev,Azpm_prev,Axmp_prev,
		  Azmm_prev,Aymm_prev,Axpm_prev,Axmm_prev,
		  Aypm_prev,Aypp_prev,Azmp_prev,Azpp_prev);
    }
  }
  else if (dir == "zx")
  {
    if (sign == "--")
    {
      //Moving in the negative z-direction and the
      //negative x-direction. Need to rotate the cell
      //180 degrees around the y-axis in order to use
      //calcA4faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      azmpindex_x = k+(Nz)*((j+1)+(Ny+1)*(i+1));
      axpmindex_z = (k+1)+(Nz+1)*((j+1)+(Ny+1)*i);
      aymmindex_z = (k+1)+(Nz+1)*(j+(Ny)*i);
      axmmindex_z = (k+1)+(Nz+1)*(j+(Ny+1)*i);
      azmmindex_x = k+(Nz)*(j+(Ny+1)*(i+1));
      aymmindex_x = k+(Nz+1)*(j+(Ny)*(i+1));
      aypmindex_x = (k+1)+(Nz+1)*(j+(Ny)*(i+1));

      //Use these indices for the indices of the
      //previous values.

      azppindex_prev = azmpindex_x;
      axppindex_prev = axpmindex_z;
      aypmindex_prev = aymmindex_z;
      axmpindex_prev = axmmindex_z;
      azpmindex_prev = azmmindex_x;
      aympindex_prev = aymmindex_x;
      ayppindex_prev = aypmindex_x;

      //These are the previous values, the values
      //that must remain fixed.

      Azpp_prev = Az[azppindex_prev];
      Axpp_prev = Ax[axppindex_prev];
      Aypm_prev = Ay[aypmindex_prev];
      Axmp_prev = Ax[axmpindex_prev];
      Azpm_prev = Az[azpmindex_prev];
      Aymp_prev = Ay[aympindex_prev];
      Aypp_prev = Ay[ayppindex_prev];

      //Additional information is needed to
      //help minimize changes. This comes in
      //the form of more previously set A
      //values.

      azmmindex_z = (k+1)+(Nz)*(j+(Ny+1)*i);
      azmpindex_z = (k+1)+(Nz)*((j+1)+(Ny+1)*i);
      axmmindex_x = k+(Nz+1)*(j+(Ny+1)*(i+1));
      axpmindex_x = k+(Nz+1)*((j+1)+(Ny+1)*(i+1));

      Azmm_prev = Az[azmmindex_z];
      Azmp_prev = Az[azmpindex_z];
      Axmm_prev = Ax[axmmindex_x];
      Axpm_prev = Ax[axpmindex_x];

      //Calculate A for this cell.

      calcA6faces(Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Bzm,Bzp,Bxm,Bxp,Bym,Byp,
		  stepz,stepx,stepy);

      Azmm = -Az[azmmindex];
      Azpm = -Az[azpmindex];
      Azmp = -Az[azmpindex];
      Azpp = -Az[azppindex];
      Axmm = -Ax[axmmindex];
      Axpm = -Ax[axpmindex];
      Axmp = -Ax[axmpindex];
      Axpp = -Ax[axppindex];
      Aymm =  Ay[aymmindex];
      Aypm =  Ay[aypmindex];
      Aymp =  Ay[aympindex];
      Aypp =  Ay[ayppindex];

      calcA4faces(Azpm,Azmm,Azpp,Azmp,
		  Axmp,Axpp,Axmm,Axpm,
		  Aypp,Aymp,Aypm,Aymm,
		  -Bzp,-Bzm,-Bxp,-Bxm,Bym,Byp,
		  -Azpp_prev,-Axpp_prev, Aypm_prev,
		  -Axmp_prev,-Azpm_prev, Aymp_prev, Aypp_prev,
		  -Azmm_prev,-Azmp_prev,-Axmm_prev,-Axpm_prev);

      Az[azmmindex] = -Azmm;
      Az[azpmindex] = -Azpm;
      Az[azmpindex] = -Azmp;
      Az[azppindex] = -Azpp;
      Ax[axmmindex] = -Axmm;
      Ax[axpmindex] = -Axpm;
      Ax[axmpindex] = -Axmp;
      Ax[axppindex] = -Axpp;
      Ay[aymmindex] =  Aymm;
      Ay[aypmindex] =  Aypm;
      Ay[aympindex] =  Aymp;
      Ay[ayppindex] =  Aypp;
    }
    else if (sign == "+-")
    {
      //Moving in the positive z-direction and the
      //negative x-direction. Need to rotate the cell
      //180 degrees around the z-axis in order to use
      //calcA4faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      azmmindex_x = k+(Nz)*(j+(Ny+1)*(i+1));
      axmpindex_z = (k+1-1)+(Nz+1)*(j+(Ny+1)*i);
      aypmindex_z = (k+1-1)+(Nz+1)*(j+(Ny)*i);
      axppindex_z = (k+1-1)+(Nz+1)*((j+1)+(Ny+1)*i);
      azmpindex_x = k+(Nz)*((j+1)+(Ny+1)*(i+1));
      aypmindex_x = (k+1)+(Nz+1)*(j+(Ny)*(i+1));
      aymmindex_x = k+(Nz+1)*(j+(Ny)*(i+1));

      //Use these indices for the indices of the
      //previous values.

      azpmindex_prev = azmmindex_x;
      axmmindex_prev = axmpindex_z;
      aymmindex_prev = aypmindex_z;
      axpmindex_prev = axppindex_z;
      azppindex_prev = azmpindex_x;
      ayppindex_prev = aypmindex_x;
      aympindex_prev = aymmindex_x;

      //These are the previous values, the values
      //that must remain fixed.

      Azpm_prev = Az[azpmindex_prev];
      Axmm_prev = Ax[axmmindex_prev];
      Aymm_prev = Ay[aymmindex_prev];
      Axpm_prev = Ax[axpmindex_prev];
      Azpp_prev = Az[azppindex_prev];
      Aypp_prev = Ay[ayppindex_prev];
      Aymp_prev = Ay[aympindex_prev];

      //Additional information is needed to
      //help minimize changes. This comes in
      //the form of more previously set A
      //values.

      azmpindex_z = (k-1)+(Nz)*((j+1)+(Ny+1)*i);
      azmmindex_z = (k-1)+(Nz)*(j+(Ny+1)*i);
      axppindex_x = (k+1)+(Nz+1)*((j+1)+(Ny+1)*(i+1));
      axmpindex_x = (k+1)+(Nz+1)*(j+(Ny+1)*(i+1));

      Azmp_prev = Az[azmpindex_z];
      Azmm_prev = Az[azmmindex_z];
      Axpp_prev = Ax[axppindex_x];
      Axmp_prev = Ax[axmpindex_x];

      //Calculate A for this cell.

      calcA6faces(Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Bzm,Bzp,Bxm,Bxp,Bym,Byp,
		  stepz,stepx,stepy);

      Azmm =  Az[azmmindex];
      Azpm =  Az[azpmindex];
      Azmp =  Az[azmpindex];
      Azpp =  Az[azppindex];
      Axmm = -Ax[axmmindex];
      Axpm = -Ax[axpmindex];
      Axmp = -Ax[axmpindex];
      Axpp = -Ax[axppindex];
      Aymm = -Ay[aymmindex];
      Aypm = -Ay[aypmindex];
      Aymp = -Ay[aympindex];
      Aypp = -Ay[ayppindex];

      calcA4faces(Azpp,Azmp,Azpm,Azmm,
		  Axpm,Axmm,Axpp,Axmp,
		  Aymp,Aypp,Aymm,Aypm,
		  Bzm,Bzp,-Bxp,-Bxm,-Byp,-Bym,
		   Azpm_prev,-Axmm_prev,-Aymm_prev,
		  -Axpm_prev, Azpp_prev,-Aypp_prev,-Aymp_prev,
		   Azmp_prev, Azmm_prev,-Axpp_prev,-Axmp_prev);

      Az[azmmindex] =  Azmm;
      Az[azpmindex] =  Azpm;
      Az[azmpindex] =  Azmp;
      Az[azppindex] =  Azpp;
      Ax[axmmindex] = -Axmm;
      Ax[axpmindex] = -Axpm;
      Ax[axmpindex] = -Axmp;
      Ax[axppindex] = -Axpp;
      Ay[aymmindex] = -Aymm;
      Ay[aypmindex] = -Aypm;
      Ay[aympindex] = -Aymp;
      Ay[ayppindex] = -Aypp;
    }
    else if (sign == "-+")
    {
      //Moving in the negative z-direction and the
      //positive x-direction. Need to rotate the cell
      //180 degrees around the x-axis in order to use
      //calcA4faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      azpmindex_x = k+(Nz)*(j+(Ny+1)*(i+1-1));
      axmmindex_z = (k+1)+(Nz+1)*(j+(Ny+1)*i);
      aympindex_z = (k+1)+(Nz+1)*(j+(Ny)*(i+1));
      axpmindex_z = (k+1)+(Nz+1)*((j+1)+(Ny+1)*i);
      azppindex_x = k+(Nz)*((j+1)+(Ny+1)*(i+1-1));
      aympindex_x = k+(Nz+1)*(j+(Ny)*(i+1-1));
      ayppindex_x = (k+1)+(Nz+1)*(j+(Ny)*(i+1-1));

      //Use these indices for the indices of the
      //previous values.

      azmmindex_prev = azpmindex_x;
      axmpindex_prev = axmmindex_z;
      ayppindex_prev = aympindex_z;
      axppindex_prev = axpmindex_z;
      azmpindex_prev = azppindex_x;
      aymmindex_prev = aympindex_x;
      aypmindex_prev = ayppindex_x;

      //These are the previous values, the values
      //that must remain fixed.

      Azmm_prev = Az[azmmindex_prev];
      Axmp_prev = Ax[axmpindex_prev];
      Aypp_prev = Ay[ayppindex_prev];
      Axpp_prev = Ax[axppindex_prev];
      Azmp_prev = Az[azmpindex_prev];
      Aymm_prev = Ay[aymmindex_prev];
      Aypm_prev = Ay[aypmindex_prev];

      //Additional information is needed to
      //help minimize changes. This comes in
      //the form of more previously set A
      //values.

      azppindex_z = (k+1)+(Nz)*((j+1)+(Ny+1)*(i+1));
      azpmindex_z = (k+1)+(Nz)*(j+(Ny+1)*(i+1));
      axpmindex_x = k+(Nz+1)*((j+1)+(Ny+1)*(i-1));
      axmmindex_x = k+(Nz+1)*(j+(Ny+1)*(i-1));

      Azpp_prev = Az[azppindex_z];
      Azpm_prev = Az[azpmindex_z];
      Axpm_prev = Ax[axpmindex_x];
      Axmm_prev = Ax[axmmindex_x];

      //Calculate A for this cell.

      calcA6faces(Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Bzm,Bzp,Bxm,Bxp,Bym,Byp,
		  stepz,stepx,stepy);

      Azmm = -Az[azmmindex];
      Azpm = -Az[azpmindex];
      Azmp = -Az[azmpindex];
      Azpp = -Az[azppindex];
      Axmm =  Ax[axmmindex];
      Axpm =  Ax[axpmindex];
      Axmp =  Ax[axmpindex];
      Axpp =  Ax[axppindex];
      Aymm = -Ay[aymmindex];
      Aypm = -Ay[aypmindex];
      Aymp = -Ay[aympindex];
      Aypp = -Ay[ayppindex];

      calcA4faces(Azmp,Azpp,Azmm,Azpm,
		  Axpp,Axmp,Axpm,Axmm,
		  Aypm,Aymm,Aypp,Aymp,
		  -Bzp,-Bzm,Bxm,Bxp,-Byp,-Bym,
		  -Azmm_prev, Axmp_prev,-Aypp_prev,
		   Axpp_prev,-Azmp_prev,-Aymm_prev,-Aypm_prev,
		  -Azpp_prev,-Azpm_prev, Axpm_prev, Axmm_prev);

      Az[azmmindex] = -Azmm;
      Az[azpmindex] = -Azpm;
      Az[azmpindex] = -Azmp;
      Az[azppindex] = -Azpp;
      Ax[axmmindex] =  Axmm;
      Ax[axpmindex] =  Axpm;
      Ax[axmpindex] =  Axmp;
      Ax[axppindex] =  Axpp;
      Ay[aymmindex] = -Aymm;
      Ay[aypmindex] = -Aypm;
      Ay[aympindex] = -Aymp;
      Ay[ayppindex] = -Aypp;
    }
    else if (sign == "++")
    {
      //Moving in the positive z-direction and the
      //positive x-direction. Do not need to do
      //anything special in order to use calcA4faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      azppindex_x = k+(Nz)*((j+1)+(Ny+1)*(i+1-1));
      axppindex_z = (k+1-1)+(Nz+1)*((j+1)+(Ny+1)*i);
      ayppindex_z = (k+1-1)+(Nz+1)*(j+(Ny)*(i+1));
      axmpindex_z = (k+1-1)+(Nz+1)*(j+(Ny+1)*i);
      azpmindex_x = k+(Nz)*(j+(Ny+1)*(i+1-1));
      ayppindex_x = (k+1)+(Nz+1)*(j+(Ny)*(i+1-1));
      aympindex_x = k+(Nz+1)*(j+(Ny)*(i+1-1));

      //Use these indices for the indices of the
      //previous values.

      azmpindex_prev = azppindex_x;
      axpmindex_prev = axppindex_z;
      aympindex_prev = ayppindex_z;
      axmmindex_prev = axmpindex_z;
      azmmindex_prev = azpmindex_x;
      aypmindex_prev = ayppindex_x;
      aymmindex_prev = aympindex_x;

      //These are the previous values, the values
      //that must remain fixed.

      Azmp_prev = Az[azmpindex_prev];
      Axpm_prev = Ax[axpmindex_prev];
      Aymp_prev = Ay[aympindex_prev];
      Axmm_prev = Ax[axmmindex_prev];
      Azmm_prev = Az[azmmindex_prev];
      Aypm_prev = Ay[aypmindex_prev];
      Aymm_prev = Ay[aymmindex_prev];

      //Additional information is needed to
      //help minimize changes. This comes in
      //the form of more previously set A
      //values.

      azpmindex_z = (k-1)+(Nz)*(j+(Ny+1)*(i+1));
      azppindex_z = (k-1)+(Nz)*((j+1)+(Ny+1)*(i+1));
      axmpindex_x = (k+1)+(Nz+1)*(j+(Ny+1)*(i-1));
      axppindex_x = (k+1)+(Nz+1)*((j+1)+(Ny+1)*(i-1));

      Azpm_prev = Az[azpmindex_z];
      Azpp_prev = Az[azppindex_z];
      Axmp_prev = Ax[axmpindex_x];
      Axpp_prev = Ax[axppindex_x];

      //Calculate A for this cell.

      calcA6faces(Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Bzm,Bzp,Bxm,Bxp,Bym,Byp,
		  stepz,stepx,stepy);

      calcA4faces(Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Bzm,Bzp,Bxm,Bxp,Bym,Byp,
		  Azmp_prev,Axpm_prev,Aymp_prev,
		  Axmm_prev,Azmm_prev,Aypm_prev,Aymm_prev,
		  Azpm_prev,Azpp_prev,Axmp_prev,Axpp_prev);
    }
  }


  else if (dir == "xyz")
  {
    if (sign == "---")
    {
      //Moving in the negative x-direction, the
      //negative y-direction, and the negative
      //z-direction. Need to rotate the cell 180
      //degrees around the z-axis and then -90
      //degrees around the new y-axis in order 
      //to use calcA3faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      azmpindex_x = k+(Nz)*((j+1)+(Ny+1)*(i+1));
      azmmindex_x = k+(Nz)*(j+(Ny+1)*(i+1));
      azmmindex_y = k+(Nz)*((j+1)+(Ny+1)*i);
      aympindex_z = (k+1)+(Nz+1)*(j+(Ny)*(i+1));
      aymmindex_z = (k+1)+(Nz+1)*(j+(Ny)*i);
      aymmindex_x = k+(Nz+1)*(j+(Ny)*(i+1));
      axmpindex_y = (k+1)+(Nz+1)*((j+1)+(Ny+1)*i);
      axmmindex_y = k+(Nz+1)*((j+1)+(Ny+1)*i);
      axmmindex_z = (k+1)+(Nz+1)*(j+(Ny+1)*i);

      //Use these indices for the indices of the
      //previous values.

      azppindex_prev = azmpindex_x;
      azpmindex_prev = azmmindex_x;
      azmpindex_prev = azmmindex_y;
      ayppindex_prev = aympindex_z;
      aypmindex_prev = aymmindex_z;
      aympindex_prev = aymmindex_x;
      axppindex_prev = axmpindex_y;
      axpmindex_prev = axmmindex_y;
      axmpindex_prev = axmmindex_z;

      //These are the previous values, the values
      //that must remain fixed.

      Azpp_prev = Az[azppindex_prev];
      Azpm_prev = Az[azpmindex_prev];
      Azmp_prev = Az[azmpindex_prev];
      Aypp_prev = Ay[ayppindex_prev];
      Aypm_prev = Ay[aypmindex_prev];
      Aymp_prev = Ay[aympindex_prev];
      Axpp_prev = Ax[axppindex_prev];
      Axpm_prev = Ax[axpmindex_prev];
      Axmp_prev = Ax[axmpindex_prev];

      //Additional information is needed to
      //help minimize changes. This comes in
      //the form of more previously set A
      //values.

      azmmindex_z = (k+1)+(Nz)*(j+(Ny+1)*i);
      aymmindex_y = k+(Nz+1)*((j+1)+(Ny)*i);
      axmmindex_x = k+(Nz+1)*(j+(Ny+1)*(i+1));

      Azmm_prev = Az[azmmindex_z];
      Aymm_prev = Ay[aymmindex_y];
      Axmm_prev = Ax[axmmindex_x];

      //Calculate A for this cell.

      calcA6faces(Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Bxm,Bxp,Bym,Byp,Bzm,Bzp,
		  stepx,stepy,stepz);

      Axmm = -Ax[axmmindex];
      Axpm = -Ax[axpmindex];
      Axmp = -Ax[axmpindex];
      Axpp = -Ax[axppindex];
      Aymm = -Ay[aymmindex];
      Aypm = -Ay[aypmindex];
      Aymp = -Ay[aympindex];
      Aypp = -Ay[ayppindex];
      Azmm = -Az[azmmindex];
      Azpm = -Az[azpmindex];
      Azmp = -Az[azmpindex];
      Azpp = -Az[azppindex];

      calcA3faces(Azpp,Azpm,Azmp,Azmm,
		  Aypp,Aypm,Aymp,Aymm,
		  Axpp,Axpm,Axmp,Axmm,
		  -Bzp,-Bzm,-Byp,-Bym,-Bxp,-Bxm,
		  -Azpp_prev,-Azpm_prev,-Azmp_prev,
		  -Aypp_prev,-Aypm_prev,-Aymp_prev,
		  -Axpp_prev,-Axpm_prev,-Axmp_prev,
		  -Azmm_prev,-Aymm_prev,-Axmm_prev);

      Ax[axmmindex] = -Axmm;
      Ax[axpmindex] = -Axpm;
      Ax[axmpindex] = -Axmp;
      Ax[axppindex] = -Axpp;
      Ay[aymmindex] = -Aymm;
      Ay[aypmindex] = -Aypm;
      Ay[aympindex] = -Aymp;
      Ay[ayppindex] = -Aypp;
      Az[azmmindex] = -Azmm;
      Az[azpmindex] = -Azpm;
      Az[azmpindex] = -Azmp;
      Az[azppindex] = -Azpp;
    }
    else if (sign == "-+-")
    {
      //Moving in the negative x-direction, the
      //positive y-direction, and the negative
      //z-direction. Need to rotate the cell 180
      //degrees around the y-axis in order to use 
      //calcA3faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      axmmindex_z = (k+1)+(Nz+1)*(j+(Ny+1)*i);
      axpmindex_z = (k+1)+(Nz+1)*((j+1)+(Ny+1)*i);
      axpmindex_y = k+(Nz+1)*((j+1-1)+(Ny+1)*i);
      aypmindex_x = (k+1)+(Nz+1)*(j+(Ny)*(i+1));
      aymmindex_x = k+(Nz+1)*(j+(Ny)*(i+1));
      aymmindex_z = (k+1)+(Nz+1)*(j+(Ny)*i);
      azppindex_y = k+(Nz)*((j+1-1)+(Ny+1)*(i+1));
      azmpindex_y = k+(Nz)*((j+1-1)+(Ny+1)*i);
      azmpindex_x = k+(Nz)*((j+1)+(Ny+1)*(i+1));

      //Use these indices for the indices of the
      //previous values.

      axmpindex_prev = axmmindex_z;
      axppindex_prev = axpmindex_z;
      axmmindex_prev = axpmindex_y;
      ayppindex_prev = aypmindex_x;
      aympindex_prev = aymmindex_x;
      aypmindex_prev = aymmindex_z;
      azpmindex_prev = azppindex_y;
      azmmindex_prev = azmpindex_y;
      azppindex_prev = azmpindex_x;

      //These are the previous values, the values
      //that must remain fixed.

      Axmp_prev = Ax[axmpindex_prev];
      Axpp_prev = Ax[axppindex_prev];
      Axmm_prev = Ax[axmmindex_prev];
      Aypp_prev = Ay[ayppindex_prev];
      Aymp_prev = Ay[aympindex_prev];
      Aypm_prev = Ay[aypmindex_prev];
      Azpm_prev = Az[azpmindex_prev];
      Azmm_prev = Az[azmmindex_prev];
      Azpp_prev = Az[azppindex_prev];

      //Additional information is needed to
      //help minimize changes. This comes in
      //the form of more previously set A
      //values.

      axpmindex_x = k+(Nz+1)*((j+1)+(Ny+1)*(i+1));
      aymmindex_y = k+(Nz+1)*((j-1)+(Ny)*i);
      azmpindex_z = (k+1)+(Nz)*((j+1)+(Ny+1)*i);

      Axpm_prev = Ax[axpmindex_x];
      Aymm_prev = Ay[aymmindex_y];
      Azmp_prev = Az[azmpindex_z];

      //Calculate A for this cell.

      calcA6faces(Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Bxm,Bxp,Bym,Byp,Bzm,Bzp,
		  stepx,stepy,stepz);

      Azmm = -Az[azmmindex];
      Azpm = -Az[azpmindex];
      Azmp = -Az[azmpindex];
      Azpp = -Az[azppindex];
      Axmm = -Ax[axmmindex];
      Axpm = -Ax[axpmindex];
      Axmp = -Ax[axmpindex];
      Axpp = -Ax[axppindex];
      Aymm =  Ay[aymmindex];
      Aypm =  Ay[aypmindex];
      Aymp =  Ay[aympindex];
      Aypp =  Ay[ayppindex];

      calcA3faces(Azpm,Azmm,Azpp,Azmp,
		  Axmp,Axpp,Axmm,Axpm,
		  Aypp,Aymp,Aypm,Aymm,
		  -Bzp,-Bzm,-Bxp,-Bxm,Bym,Byp,
		  -Azpm_prev,-Azmm_prev,-Azpp_prev,
		  -Axmp_prev,-Axpp_prev,-Axmm_prev,
		   Aypp_prev, Aymp_prev, Aypm_prev,
		  -Axpm_prev, Aymm_prev,-Azmp_prev);

      Az[azmmindex] = -Azmm;
      Az[azpmindex] = -Azpm;
      Az[azmpindex] = -Azmp;
      Az[azppindex] = -Azpp;
      Ax[axmmindex] = -Axmm;
      Ax[axpmindex] = -Axpm;
      Ax[axmpindex] = -Axmp;
      Ax[axppindex] = -Axpp;
      Ay[aymmindex] =  Aymm;
      Ay[aypmindex] =  Aypm;
      Ay[aympindex] =  Aymp;
      Ay[ayppindex] =  Aypp;
    }
    else if (sign == "--+")
    {
      //Moving in the negative x-direction, the
      //negative y-direction, and the positive
      //z-direction. Need to rotate the cell 180
      //degrees around the z-axis in order to use
      //calcA3faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      axppindex_z = (k+1-1)+(Nz+1)*((j+1)+(Ny+1)*i);
      axmpindex_z = (k+1-1)+(Nz+1)*(j+(Ny+1)*i);
      axmpindex_y = (k+1)+(Nz+1)*((j+1)+(Ny+1)*i);
      aymmindex_x = k+(Nz+1)*(j+(Ny)*(i+1));
      aypmindex_x = (k+1)+(Nz+1)*(j+(Ny)*(i+1));
      aypmindex_z = (k+1-1)+(Nz+1)*(j+(Ny)*i);
      azpmindex_y = k+(Nz)*((j+1)+(Ny+1)*(i+1));
      azmmindex_y = k+(Nz)*((j+1)+(Ny+1)*i);
      azmmindex_x = k+(Nz)*(j+(Ny+1)*(i+1));

      //Use these indices for the indices of the
      //previous values.

      axpmindex_prev = axppindex_z;
      axmmindex_prev = axmpindex_z;
      axppindex_prev = axmpindex_y;
      aympindex_prev = aymmindex_x;
      ayppindex_prev = aypmindex_x;
      aymmindex_prev = aypmindex_z;
      azppindex_prev = azpmindex_y;
      azmpindex_prev = azmmindex_y;
      azpmindex_prev = azmmindex_x;

      //These are the previous values, the values
      //that must remain fixed.

      Axpm_prev = Ax[axpmindex_prev];
      Axmm_prev = Ax[axmmindex_prev];
      Axpp_prev = Ax[axppindex_prev];
      Aymp_prev = Ay[aympindex_prev];
      Aypp_prev = Ay[ayppindex_prev];
      Aymm_prev = Ay[aymmindex_prev];
      Azpp_prev = Az[azppindex_prev];
      Azmp_prev = Az[azmpindex_prev];
      Azpm_prev = Az[azpmindex_prev];

      //Additional information is needed to
      //help minimize changes. This comes in
      //the form of more previously set A
      //values.

      axmpindex_x = (k+1)+(Nz+1)*(j+(Ny+1)*(i+1));
      aypmindex_y = (k+1)+(Nz+1)*((j+1)+(Ny)*i);
      azmmindex_z = (k-1)+(Nz)*(j+(Ny+1)*i);

      Axmp_prev = Ax[axmpindex_x];
      Aypm_prev = Ay[aypmindex_y];
      Azmm_prev = Az[azmmindex_z];

      //Calculate A for this cell.

      calcA6faces(Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Bxm,Bxp,Bym,Byp,Bzm,Bzp,
		  stepx,stepy,stepz);

      Axmm = -Ax[axmmindex];
      Axpm = -Ax[axpmindex];
      Axmp = -Ax[axmpindex];
      Axpp = -Ax[axppindex];
      Aymm = -Ay[aymmindex];
      Aypm = -Ay[aypmindex];
      Aymp = -Ay[aympindex];
      Aypp = -Ay[ayppindex];
      Azmm =  Az[azmmindex];
      Azpm =  Az[azpmindex];
      Azmp =  Az[azmpindex];
      Azpp =  Az[azppindex];

      calcA3faces(Axpm,Axmm,Axpp,Axmp,
		  Aymp,Aypp,Aymm,Aypm,
		  Azpp,Azmp,Azpm,Azmm,
		  -Bxp,-Bxm,-Byp,-Bym,Bzm,Bzp,
		  -Axpm_prev,-Axmm_prev,-Axpp_prev,
		  -Aymp_prev,-Aypp_prev,-Aymm_prev,
		   Azpp_prev, Azmp_prev, Azpm_prev,
		  -Axmp_prev,-Aypm_prev, Azmm_prev);

      Ax[axmmindex] = -Axmm;
      Ax[axpmindex] = -Axpm;
      Ax[axmpindex] = -Axmp;
      Ax[axppindex] = -Axpp;
      Ay[aymmindex] = -Aymm;
      Ay[aypmindex] = -Aypm;
      Ay[aympindex] = -Aymp;
      Ay[ayppindex] = -Aypp;
      Az[azmmindex] =  Azmm;
      Az[azpmindex] =  Azpm;
      Az[azmpindex] =  Azmp;
      Az[azppindex] =  Azpp;
    }
    else if (sign == "-++")
    {
      //Moving in the negative x-direction, the
      //positive y-direction, and the positive
      //z-direction. Need to rotate the cell -90
      //degrees around the z-axis in order to use
      //calcA3faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      ayppindex_z = (k+1-1)+(Nz+1)*(j+(Ny)*(i+1));
      aypmindex_z = (k+1-1)+(Nz+1)*(j+(Ny)*i);
      aypmindex_x = (k+1)+(Nz+1)*(j+(Ny)*(i+1));
      axpmindex_y = k+(Nz+1)*((j+1-1)+(Ny+1)*i);
      axppindex_y = (k+1)+(Nz+1)*((j+1-1)+(Ny+1)*i);
      axppindex_z = (k+1-1)+(Nz+1)*((j+1)+(Ny+1)*i);
      azmmindex_x = k+(Nz)*(j+(Ny+1)*(i+1));
      azmpindex_x = k+(Nz)*((j+1)+(Ny+1)*(i+1));
      azmpindex_y = k+(Nz)*((j+1-1)+(Ny+1)*i);

      //Use these indices for the indices of the
      //previous values.

      aympindex_prev = ayppindex_z;
      aymmindex_prev = aypmindex_z;
      ayppindex_prev = aypmindex_x;
      axmmindex_prev = axpmindex_y;
      axmpindex_prev = axppindex_y;
      axpmindex_prev = axppindex_z;
      azpmindex_prev = azmmindex_x;
      azppindex_prev = azmpindex_x;
      azmmindex_prev = azmpindex_y;

      //These are the previous values, the values
      //that must remain fixed.

      Aymp_prev = Ay[aympindex_prev];
      Aymm_prev = Ay[aymmindex_prev];
      Aypp_prev = Ay[ayppindex_prev];
      Axmm_prev = Ax[axmmindex_prev];
      Axmp_prev = Ax[axmpindex_prev];
      Axpm_prev = Ax[axpmindex_prev];
      Azpm_prev = Az[azpmindex_prev];
      Azpp_prev = Az[azppindex_prev];
      Azmm_prev = Az[azmmindex_prev];

      //Additional information is needed to
      //help minimize changes. This comes in
      //the form of more previously set A
      //values.

      aypmindex_y = (k+1)+(Nz+1)*((j-1)+(Ny)*i);
      axppindex_x = (k+1)+(Nz+1)*((j+1)+(Ny+1)*(i+1));
      azmpindex_z = (k-1)+(Nz)*((j+1)+(Ny+1)*i);

      Aypm_prev = Ay[aypmindex_y];
      Axpp_prev = Ax[axppindex_x];
      Azmp_prev = Az[azmpindex_z];

      //Calculate A for this cell.

      calcA6faces(Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Bxm,Bxp,Bym,Byp,Bzm,Bzp,
		  stepx,stepy,stepz);

      Aymm =  Ay[aymmindex];
      Aypm =  Ay[aypmindex];
      Aymp =  Ay[aympindex];
      Aypp =  Ay[ayppindex];
      Axmm = -Ax[axmmindex];
      Axpm = -Ax[axpmindex];
      Axmp = -Ax[axmpindex];
      Axpp = -Ax[axppindex];
      Azmm =  Az[azmmindex];
      Azpm =  Az[azpmindex];
      Azmp =  Az[azmpindex];
      Azpp =  Az[azppindex];

      calcA3faces(Aymp,Aymm,Aypp,Aypm,
		  Axmm,Axmp,Axpm,Axpp,
		  Azpm,Azpp,Azmm,Azmp,
		  Bym,Byp,-Bxp,-Bxm,Bzm,Bzp,
		   Aymp_prev, Aymm_prev, Aypp_prev,
		  -Axmm_prev,-Axmp_prev,-Axpm_prev,
		   Azpm_prev, Azpp_prev, Azmm_prev,
		   Aypm_prev,-Axpp_prev, Azmp_prev);

      Ay[aymmindex] =  Aymm;
      Ay[aypmindex] =  Aypm;
      Ay[aympindex] =  Aymp;
      Ay[ayppindex] =  Aypp;
      Ax[axmmindex] = -Axmm;
      Ax[axpmindex] = -Axpm;
      Ax[axmpindex] = -Axmp;
      Ax[axppindex] = -Axpp;
      Az[azmmindex] =  Azmm;
      Az[azpmindex] =  Azpm;
      Az[azmpindex] =  Azmp;
      Az[azppindex] =  Azpp;
    }
    else if (sign == "+--")
    {
      //Moving in the positive x-direction, the
      //negative y-direction, and the negative
      //z-direction. Need to rotate the cell  180
      //degrees around the x-axis in order to use
      //calcA3faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      axpmindex_z = (k+1)+(Nz+1)*((j+1)+(Ny+1)*i);
      axmmindex_z = (k+1)+(Nz+1)*(j+(Ny+1)*i);
      axmmindex_y = k+(Nz+1)*((j+1)+(Ny+1)*i);
      ayppindex_x = (k+1)+(Nz+1)*(j+(Ny)*(i+1-1));
      aympindex_x = k+(Nz+1)*(j+(Ny)*(i+1-1));
      aympindex_z = (k+1)+(Nz+1)*(j+(Ny)*(i+1));
      azmmindex_y = k+(Nz)*((j+1)+(Ny+1)*i);
      azpmindex_y = k+(Nz)*((j+1)+(Ny+1)*(i+1));
      azpmindex_x = k+(Nz)*(j+(Ny+1)*(i+1-1));

      //Use these indices for the indices of the
      //previous values.

      axppindex_prev = axpmindex_z;
      axmpindex_prev = axmmindex_z;
      axpmindex_prev = axmmindex_y;
      aypmindex_prev = ayppindex_x;
      aymmindex_prev = aympindex_x;
      ayppindex_prev = aympindex_z;
      azmpindex_prev = azmmindex_y;
      azppindex_prev = azpmindex_y;
      azmmindex_prev = azpmindex_x;

      //These are the previous values, the values
      //that must remain fixed.

      Axpp_prev = Ax[axppindex_prev];
      Axmp_prev = Ax[axmpindex_prev];
      Axpm_prev = Ax[axpmindex_prev];
      Aypm_prev = Ay[aypmindex_prev];
      Aymm_prev = Ay[aymmindex_prev];
      Aypp_prev = Ay[ayppindex_prev];
      Azmp_prev = Az[azmpindex_prev];
      Azpp_prev = Az[azppindex_prev];
      Azmm_prev = Az[azmmindex_prev];

      //Additional information is needed to
      //help minimize changes. This comes in
      //the form of more previously set A
      //values.

      axmmindex_x = k+(Nz+1)*(j+(Ny+1)*(i-1));
      aympindex_y = k+(Nz+1)*((j+1)+(Ny)*(i+1));
      azpmindex_z = (k+1)+(Nz)*(j+(Ny+1)*(i+1));

      Axmm_prev = Ax[axmmindex_x];
      Aymp_prev = Ay[aympindex_y];
      Azpm_prev = Az[azpmindex_z];

      //Calculate A for this cell.

      calcA6faces(Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Bxm,Bxp,Bym,Byp,Bzm,Bzp,
		  stepx,stepy,stepz);

      Aymm = -Ay[aymmindex];
      Aypm = -Ay[aypmindex];
      Aymp = -Ay[aympindex];
      Aypp = -Ay[ayppindex];
      Azmm = -Az[azmmindex];
      Azpm = -Az[azpmindex];
      Azmp = -Az[azmpindex];
      Azpp = -Az[azppindex];
      Axmm =  Ax[axmmindex];
      Axpm =  Ax[axpmindex];
      Axmp =  Ax[axmpindex];
      Axpp =  Ax[axppindex];

      calcA3faces(Aypm,Aymm,Aypp,Aymp,
		  Azmp,Azpp,Azmm,Azpm,
		  Axpp,Axmp,Axpm,Axmm,
		  -Byp,-Bym,-Bzp,Bzm,Bxm,Bxp,
		  -Aypm_prev,-Aymm_prev,-Aypp_prev,
		  -Azmp_prev,-Azpp_prev,-Azmm_prev,
		   Axpp_prev, Axmp_prev, Axpm_prev,
		   Axmm_prev,-Aymp_prev,-Azpm_prev);

      Ay[aymmindex] = -Aymm;
      Ay[aypmindex] = -Aypm;
      Ay[aympindex] = -Aymp;
      Ay[ayppindex] = -Aypp;
      Az[azmmindex] = -Azmm;
      Az[azpmindex] = -Azpm;
      Az[azmpindex] = -Azmp;
      Az[azppindex] = -Azpp;
      Ax[axmmindex] =  Axmm;
      Ax[axpmindex] =  Axpm;
      Ax[axmpindex] =  Axmp;
      Ax[axppindex] =  Axpp;
    }
    else if (sign == "++-")
    {
      //Moving in the positive x-direction, the
      //positive y-direction, and the negative
      //z-direction. Need to rotate the cell -90
      //degrees around the y-axis in order to use
      //calcA3faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      axppindex_y = (k+1)+(Nz+1)*((j+1-1)+(Ny+1)*i);
      axpmindex_y = k+(Nz+1)*((j+1-1)+(Ny+1)*i);
      axpmindex_z = (k+1)+(Nz+1)*((j+1)+(Ny+1)*i);
      azpmindex_x = k+(Nz)*(j+(Ny+1)*(i+1-1));
      azppindex_x = k+(Nz)*((j+1)+(Ny+1)*(i+1-1));
      azppindex_y = k+(Nz)*((j+1-1)+(Ny+1)*(i+1));
      aymmindex_z = (k+1)+(Nz+1)*(j+(Ny)*i);
      aympindex_z = (k+1)+(Nz+1)*(j+(Ny)*(i+1));
      aympindex_x = k+(Nz+1)*(j+(Ny)*(i+1-1));

      //Use these indices for the indices of the
      //previous values.

      axmpindex_prev = axppindex_y;
      axmmindex_prev = axpmindex_y;
      axppindex_prev = axpmindex_z;
      azmmindex_prev = azpmindex_x;
      azmpindex_prev = azppindex_x;
      azpmindex_prev = azppindex_y;
      aypmindex_prev = aymmindex_z;
      ayppindex_prev = aympindex_z;
      aymmindex_prev = aympindex_x;

      //These are the previous values, the values
      //that must remain fixed.

      Axmp_prev = Ax[axmpindex_prev];
      Axmm_prev = Ax[axmmindex_prev];
      Axpp_prev = Ax[axppindex_prev];
      Azmm_prev = Az[azmmindex_prev];
      Azmp_prev = Az[azmpindex_prev];
      Azpm_prev = Az[azpmindex_prev];
      Aypm_prev = Ay[aypmindex_prev];
      Aypp_prev = Ay[ayppindex_prev];
      Aymm_prev = Ay[aymmindex_prev];

      //Additional information is needed to
      //help minimize changes. This comes in
      //the form of more previously set A
      //values.

      azppindex_z = (k+1)+(Nz)*((j+1)+(Ny+1)*(i+1));
      aympindex_y = k+(Nz+1)*((j-1)+(Ny)*(i+1));
      axpmindex_x = k+(Nz+1)*((j+1)+(Ny+1)*(i-1));

      Azpp_prev = Az[azppindex_z];
      Aymp_prev = Ay[aympindex_y];
      Axpm_prev = Ax[axpmindex_x];

      //Calculate A for this cell.

      calcA6faces(Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Bxm,Bxp,Bym,Byp,Bzm,Bzp,
		  stepx,stepy,stepz);

      Axmm =  Ax[axmmindex];
      Axpm =  Ax[axpmindex];
      Axmp =  Ax[axmpindex];
      Axpp =  Ax[axppindex];
      Azmm = -Az[azmmindex];
      Azpm = -Az[azpmindex];
      Azmp = -Az[azmpindex];
      Azpp = -Az[azppindex];
      Aymm =  Ay[aymmindex];
      Aypm =  Ay[aypmindex];
      Aymp =  Ay[aympindex];
      Aypp =  Ay[ayppindex];

      calcA3faces(Axmp,Axmm,Axpp,Axpm,
		  Azmm,Azmp,Azpm,Azpp,
		  Aypm,Aypp,Aymm,Aymp,
		  Bxm,Bxp,-Bzp,-Bzm,Bym,Byp,
		   Axmp_prev, Axmm_prev, Axpp_prev,
		  -Azmm_prev,-Azmp_prev,-Azpm_prev,
		   Aypm_prev, Aypp_prev, Aymm_prev,
		  -Azpp_prev, Aymp_prev, Axpm_prev);

      Ax[axmmindex] =  Axmm;
      Ax[axpmindex] =  Axpm;
      Ax[axmpindex] =  Axmp;
      Ax[axppindex] =  Axpp;
      Az[azmmindex] = -Azmm;
      Az[azpmindex] = -Azpm;
      Az[azmpindex] = -Azmp;
      Az[azppindex] = -Azpp;
      Ay[aymmindex] =  Aymm;
      Ay[aypmindex] =  Aypm;
      Ay[aympindex] =  Aymp;
      Ay[ayppindex] =  Aypp;
    }
    else if (sign == "+-+")
    {
      //Moving in the positive x-direction, the
      //negative y-direction, and the positive
      //z-direction. Need to rotate the cell -90
      //degrees around the x-axis in order to use
      //calcA3faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      azppindex_x = k+(Nz)*((j+1)+(Ny+1)*(i+1-1));
      azpmindex_x = k+(Nz)*(j+(Ny+1)*(i+1-1));
      azpmindex_y = k+(Nz)*((j+1)+(Ny+1)*(i+1));
      aypmindex_z = (k+1-1)+(Nz+1)*(j+(Ny)*i);
      ayppindex_z = (k+1-1)+(Nz+1)*(j+(Ny)*(i+1));
      ayppindex_x = (k+1)+(Nz+1)*(j+(Ny)*(i+1-1));
      axmmindex_y = k+(Nz+1)*((j+1)+(Ny+1)*i);
      axmpindex_y = (k+1)+(Nz+1)*((j+1)+(Ny+1)*i);
      axmpindex_z = (k+1-1)+(Nz+1)*(j+(Ny+1)*i);

      //Use these indices for the indices of the
      //previous values.

      azmpindex_prev = azppindex_x;
      azmmindex_prev = azpmindex_x;
      azppindex_prev = azpmindex_y;
      aymmindex_prev = aypmindex_z;
      aympindex_prev = ayppindex_z;
      aypmindex_prev = ayppindex_x;
      axpmindex_prev = axmmindex_y;
      axppindex_prev = axmpindex_y;
      axmmindex_prev = axmpindex_z;

      //These are the previous values, the values
      //that must remain fixed.

      Azmp_prev = Az[azmpindex_prev];
      Azmm_prev = Az[azmmindex_prev];
      Azpp_prev = Az[azppindex_prev];
      Aymm_prev = Ay[aymmindex_prev];
      Aymp_prev = Ay[aympindex_prev];
      Aypm_prev = Ay[aypmindex_prev];
      Axpm_prev = Ax[axpmindex_prev];
      Axpp_prev = Ax[axppindex_prev];
      Axmm_prev = Ax[axmmindex_prev];

      //Additional information is needed to
      //help minimize changes. This comes in
      //the form of more previously set A
      //values.

      axmpindex_x = (k+1)+(Nz+1)*(j+(Ny+1)*(i-1));
      azpmindex_z = (k-1)+(Nz)*(j+(Ny+1)*(i+1));
      ayppindex_y = (k+1)+(Nz+1)*((j+1)+(Ny)*(i+1));

      Axmp_prev = Ax[axmpindex_x];
      Azpm_prev = Az[azpmindex_z];
      Aypp_prev = Ay[ayppindex_y];

      //Calculate A for this cell.

      calcA6faces(Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Bxm,Bxp,Bym,Byp,Bzm,Bzp,
		  stepx,stepy,stepz);

      Azmm =  Az[azmmindex];
      Azpm =  Az[azpmindex];
      Azmp =  Az[azmpindex];
      Azpp =  Az[azppindex];
      Aymm = -Ay[aymmindex];
      Aypm = -Ay[aypmindex];
      Aymp = -Ay[aympindex];
      Aypp = -Ay[ayppindex];
      Axmm =  Ax[axmmindex];
      Axpm =  Ax[axpmindex];
      Axmp =  Ax[axmpindex];
      Axpp =  Ax[axppindex];

      calcA3faces(Azmp,Azmm,Azpp,Azpm,
		  Aymm,Aymp,Aypm,Aypp,
		  Axpm,Axpp,Axmm,Axmp,
		  Bzm,Bzp,-Byp,-Bym,Bxm,Bxp,
		   Azmp_prev, Azmm_prev, Azpp_prev,
		  -Aymm_prev,-Aymp_prev,-Aypm_prev,
		   Axpm_prev, Axpp_prev, Axmm_prev,
		   Axmp_prev, Azpm_prev,-Aypp_prev);

      Az[azmmindex] =  Azmm;
      Az[azpmindex] =  Azpm;
      Az[azmpindex] =  Azmp;
      Az[azppindex] =  Azpp;
      Ay[aymmindex] = -Aymm;
      Ay[aypmindex] = -Aypm;
      Ay[aympindex] = -Aymp;
      Ay[ayppindex] = -Aypp;
      Ax[axmmindex] =  Axmm;
      Ax[axpmindex] =  Axpm;
      Ax[axmpindex] =  Axmp;
      Ax[axppindex] =  Axpp;
    }
    else if (sign == "+++")
    {
      //Moving in the positive x-direction, the
      //positive y-direction, and the positive
      //z-direction. Do not need to do anything
      //special in order to use calcA4faces.

      //Take the indices for the fixed sides from the
      //cell in which the side was fixed. In practice,
      //this involves "going back a cell" from the
      //current cell.

      axmpindex_z = (k+1-1)+(Nz+1)*(j+(Ny+1)*i);
      axppindex_z = (k+1-1)+(Nz+1)*((j+1)+(Ny+1)*i);
      axppindex_y = (k+1)+(Nz+1)*((j+1-1)+(Ny+1)*i);
      aympindex_x = k+(Nz+1)*(j+(Ny)*(i+1-1));
      ayppindex_x = (k+1)+(Nz+1)*(j+(Ny)*(i+1-1));
      ayppindex_z = (k+1-1)+(Nz+1)*(j+(Ny)*(i+1));
      azmpindex_y = k+(Nz)*((j+1-1)+(Ny+1)*i);
      azppindex_y = k+(Nz)*((j+1-1)+(Ny+1)*(i+1));
      azppindex_x = k+(Nz)*((j+1)+(Ny+1)*(i+1-1));

      //Use these indices for the indices of the
      //previous values.

      axmmindex_prev = axmpindex_z;
      axpmindex_prev = axppindex_z;
      axmpindex_prev = axppindex_y;
      aymmindex_prev = aympindex_x;
      aypmindex_prev = ayppindex_x;
      aympindex_prev = ayppindex_z;
      azmmindex_prev = azmpindex_y;
      azpmindex_prev = azppindex_y;
      azmpindex_prev = azppindex_x;

      //These are the previous values, the values
      //that must remain fixed.

      Axmm_prev = Ax[axmmindex_prev];
      Axpm_prev = Ax[axpmindex_prev];
      Axmp_prev = Ax[axmpindex_prev];
      Aymm_prev = Ay[aymmindex_prev];
      Aypm_prev = Ay[aypmindex_prev];
      Aymp_prev = Ay[aympindex_prev];
      Azmm_prev = Az[azmmindex_prev];
      Azpm_prev = Az[azpmindex_prev];
      Azmp_prev = Az[azmpindex_prev];

      //Additional information is needed to
      //help minimize changes. This comes in
      //the form of more previously set A
      //values.

      axppindex_x = (k+1)+(Nz+1)*((j+1)+(Ny+1)*(i-1));
      ayppindex_y = (k+1)+(Nz+1)*((j-1)+(Ny)*(i+1));
      azppindex_z = (k-1)+(Nz)*((j+1)+(Ny+1)*(i+1));

      Axpp_prev = Ax[axppindex_x];
      Aypp_prev = Ay[ayppindex_y];
      Azpp_prev = Az[azppindex_z];

      //Calculate A for this cell.

      calcA6faces(Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Bxm,Bxp,Bym,Byp,Bzm,Bzp,
		  stepx,stepy,stepz);

      calcA3faces(Ax[axmmindex],Ax[axpmindex],
		  Ax[axmpindex],Ax[axppindex],
		  Ay[aymmindex],Ay[aypmindex],
		  Ay[aympindex],Ay[ayppindex],
		  Az[azmmindex],Az[azpmindex],
		  Az[azmpindex],Az[azppindex],
		  Bxm,Bxp,Bym,Byp,Bzm,Bzp,
		  Axmm_prev,Axpm_prev,Axmp_prev,
		  Aymm_prev,Aypm_prev,Aymp_prev,
		  Azmm_prev,Azpm_prev,Azmp_prev,
		  Axpp_prev,Aypp_prev,Azpp_prev);
    }
  }
  else
  {
    //If dir and sign are anything besides a
    //combination of x, y, z and +, - respectively, 
    //it is the first cell, so do not do anything else.
  
    calcA6faces(Ax[axmmindex],Ax[axpmindex],
		Ax[axmpindex],Ax[axppindex],
		Ay[aymmindex],Ay[aypmindex],
		Ay[aympindex],Ay[ayppindex],
		Az[azmmindex],Az[azpmindex],
		Az[azmpindex],Az[azppindex],
		Bxm,Bxp,Bym,Byp,Bzm,Bzp,
		stepx,stepy,stepz);
  }
  
  return;
}
