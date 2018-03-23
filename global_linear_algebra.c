//----------------------------------------------------------------------
// B to A Solver Global Linear Algebra Method
// Copyright (C) 2018 Joshua A. Faber and Zachary J. Silberman
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

#include <stdio.h>
#include <string.h>
#include "mpi.h"
//#include <fftw3.h>
#include "dmumps_c.h"
#define JOB_INIT -1
#define JOB_END -2
#define USE_COMM_WORLD -987654

void xface(int *count, int row, int i, int j, int k, int nx, int ny, int nz, double dx, double dy, double dz, double* bval, int* irn, int* jcn, double* a, double* rhs);
void yface(int *count, int row, int i, int j, int k, int nx, int ny, int nz, double dx, double dy, double dz, double* bval, int* irn, int* jcn, double* a, double* rhs);
void zface(int *count, int row, int i, int j, int k, int nx, int ny, int nz, double dx, double dy, double dz, double* bval, int* irn, int* jcn, double* a, double* rhs);
void Coulomb(int *count, int row, int i, int j, int k, int nx, int ny, int nz, int* irn, int* jcn, double* a);

double xface_error(int i, int j, int k, int nx, int ny, int nz, double dx, double dy, double dz, double* bval, double* rhs);
double yface_error(int i, int j, int k, int nx, int ny, int nz, double dx, double dy, double dz, double* bval, double* rhs);
double zface_error(int i, int j, int k, int nx, int ny, int nz, double dx, double dy, double dz, double* bval, double* rhs);
double Coulomb_error(int i, int j, int k, int nx, int ny, int nz, double* rhs);
double Relative_xface_error(int i, int j, int k, int nx, int ny, int nz, double dx, double dy, double dz, 
			    double* bval, double* rhs);
double Relative_yface_error(int i, int j, int k, int nx, int ny, int nz, double dx, double dy, double dz, 
			    double* bval, double* rhs);
double Relative_zface_error(int i, int j, int k, int nx, int ny, int nz, double dx, double dy, double dz, 
			    double* bval, double* rhs);
double Relative_Coulomb_error(int i, int j, int k, int nx, int ny, int nz, double* rhs);

// A_x is of size n, n+1, n+1
int xi(int i, int j, int k, int nx, int ny, int nz) {return k+(nz+1)*(j+(ny+1)*i);};

// A_y is of size n+1, n, n+1, but goes second in the list
int yi(int i, int j, int k, int nx, int ny, int nz) {return k+(nz+1)*(j+ny*i)+nx*(ny+1)*(nz+1);};

// A_z is of size n+1, n+1, n, but goes third in the list
int zi(int i, int j, int k, int nx, int ny, int nz) {return k+nz*(j+(ny+1)*i)+nx*(ny+1)*(nz+1)+(nx+1)*ny*(nz+1);};


// B_x is of size n+1, n, n
int Bxi(int i, int j, int k, int nx, int ny, int nz) {return k+nz*(j+ny*i);};
// B_y is of size n, n+1, n
int Byi(int i, int j, int k, int nx, int ny, int nz) {return k+nz*(j+(ny+1)*i);};
// B_z is of size n, n, n+1
int Bzi(int i, int j, int k, int nx, int ny, int nz) {return k+(nz+1)*(j+ny*i);};

// A_x, A_y, A_z are of size n+1, n+1, n+1 (for output)
int Ai(int i, int j, int k, int nx, int ny, int nz) {return k+(nz+1)*(j+(ny+1)*i);};

#if defined(MAIN_COMP)
/*
 * Some Fortran compilers (COMPAQ fort) define main inside
 * their runtime library while a Fortran program translates
 * to MAIN_ or MAIN__ which is then called from "main". This
 * is annoying because MAIN__ has no arguments and we must
 * define argc/argv arbitrarily !!
 */
int MAIN__();
int MAIN_()
  {
    return MAIN__();
  }

int MAIN__()
{
  int argc=1;
  char * name = "c_example";
  char ** argv ;
#else
  int main(int argc, char ** argv)
    {
#endif
      
      int nx = atoi(argv[1]);
      int ny = atoi(argv[2]);
      int nz = atoi(argv[3]);

      int n_ghost = 3;  //number of ghost cells at the edge of the grid

      int test_flag = 1;  //option for the test being run, used to set up the grid
                          // 1: Rotor Test
                          // 2: TOV stars Test
      int dim = 2;        //dimension of data
                          // 2: Code copies data along the z-axis after reading
                          // 3: Do nothing special after reading

      double step;                 //numerical size of cell
      double stepx, stepy, stepz;  //numerical size of cell along each direction
      double dx, dy, dz;           //physical size of cell along each direction
      double xinit, yinit, zinit;  //beginning of physical grid along each direction

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
	xinit = -2.0-(n_ghost-1)*dx;
	yinit = -2.0-(n_ghost-1)*dx;
	zinit = -2.0-(n_ghost-1)*dx;
      }      

      DMUMPS_STRUC_C id;
      
      // 3*n*(n+1)^2 equations to solve
      MUMPS_INT nrows = nx*(ny+1)*(nz+1)+(nx+1)*ny*(nz+1)+(nx+1)*(ny+1)*nz;
      
      
      // [2nxnynz +(nxny+nxnz+nynz)]*4 consistency conditions = 8nxnynz + 4(nxny+nxnz+nynz)
      //
      // Coulomb: 7 corners (*3) = 21
      // 4([nx+ny+nz]-3) edges (*4) = 16[nx+ny+nz] - 48
      // 2(nxny+nxnz+nynz)-4(nx+ny+nz)+6   faces (*5) = 10[nxny+nxnz+nynz]-20[nx+ny+nz]+30
      // nxnynz -[nxny+nxnz+nynz] + [nx+ny+nz] - 1 interior points (*6) = 6nxnynz - 6[nxny+nxnz+nynz]+6[nx+ny+nz]-6
      //      total 14[nxnynz] + 8[nxny+nxnz+nynz] +2[nx+ny+nz] - 3

      MUMPS_INT nvals = 14*nx*ny*nz + 8*(nx*ny+nx*nz+ny*nz)+2*(nx+ny+nz)-3;


  printf("nrows and nvals: %d %d\n",nrows,nvals);

  MUMPS_INT *irn;
  MUMPS_INT *jcn;
  double *a;

  irn = (MUMPS_INT *) malloc(nvals*sizeof(MUMPS_INT));
  jcn = (MUMPS_INT *) malloc(nvals*sizeof(MUMPS_INT));
  a = (double *) malloc(nvals*sizeof(double));


  int i,j,k,l;
  double *rhs;
  rhs = (double *) malloc(nrows*sizeof(double));
  //This one should be initialized
  for(i=0; i<nrows; i++) rhs[i]=0.0;

  //This should be a read in of B-field values!!!
  double *Bx;
  double *By;
  double *Bz;

  Bx = (double *) malloc((nx+1)*ny*nz*sizeof(double));
  By = (double *) malloc(nx*(ny+1)*nz*sizeof(double));
  Bz = (double *) malloc(nx*ny*(nz+1)*sizeof(double));
  
  int bindex, b0index;
  int idum,jdum,kdum;
  float xdum,ydum,zdum;
  float col1,col2,col3,col4,col5,col9;
  double bvalue;
  FILE *infile_Bx, *infile_By, *infile_Bz;

  infile_Bx = fopen("Bx_stagger2.xy_0.2_sched.asc", "r");
  infile_By = fopen("By_stagger2.xy_0.2_sched.asc", "r");
  infile_Bz = fopen("Bz_stagger2.xy_0.2_sched.asc", "r");

  
  printf("Reading Bx\n");

  fscanf(infile_Bx, "%*[^\n]\n", NULL);
  fscanf(infile_Bx, "%*[^\n]\n", NULL);
  fscanf(infile_Bx, "%*[^\n]\n", NULL);
  fscanf(infile_Bx, "%*[^\n]\n", NULL);
  while(!feof(infile_Bx)) {
    //printf("I'm here.\n");
    fscanf(infile_Bx, "%f %f %f %f %f %i %i %i %f %f %f %f %lf\n", 
	   &col1, &col2, &col3, &col4, &col5, &idum, &jdum, &kdum, 
	   &col9, &xdum, &ydum, &zdum, &bvalue);
    if(j != 0 && k != 0) {
      bindex = Bxi(idum,(jdum-1),(kdum-1),nx,ny,nz);
      Bx[bindex] = bvalue;
    }

    fgetc(infile_Bx);
  }
    
  printf("Reading By\n");

  fscanf(infile_By, "%*[^\n]\n", NULL);
  fscanf(infile_By, "%*[^\n]\n", NULL);
  fscanf(infile_By, "%*[^\n]\n", NULL);
  fscanf(infile_By, "%*[^\n]\n", NULL);
  while(!feof(infile_By)) {
    fscanf(infile_By, "%f %f %f %f %f %i %i %i %f %f %f %f %lf\n", 
	   &col1, &col2, &col3, &col4, &col5, &idum, &jdum, &kdum, 
	   &col9, &xdum, &ydum, &zdum, &bvalue);
    if(i != 0 && k != 0) {
      bindex = Byi((idum-1),jdum,(kdum-1),nx,ny,nz);
      By[bindex] = bvalue;
    }
    fgetc(infile_By);
  }

  printf("Reading Bz\n");

  fscanf(infile_Bz, "%*[^\n]\n", NULL);
  fscanf(infile_Bz, "%*[^\n]\n", NULL);
  fscanf(infile_Bz, "%*[^\n]\n", NULL);
  fscanf(infile_Bz, "%*[^\n]\n", NULL);
  while(!feof(infile_Bz)) {
    fscanf(infile_Bz, "%f %f %f %f %f %i %i %i %f %f %f %f %lf\n", 
	   &col1, &col2, &col3, &col4, &col5, &idum, &jdum, &kdum, 
	   &col9, &xdum, &ydum, &zdum, &bvalue);
    if(i != 0 && j != 0) {
      bindex = Bzi((idum-1),(jdum-1),kdum,nx,ny,nz);
      Bz[bindex] = bvalue;    
    }
    fgetc(infile_Bz);
  }
  
  fclose(infile_Bx);
  fclose(infile_By);
  fclose(infile_Bz);

  if (dim == 2)
  {
    //If this was 2D data, copy that 
    //data along the z-direction.

    for (i=0; i<nx+1; i++) {
      for (j=0; j<ny; j++) {
	for (k=0; k<nz; k++) {
	  if(k != (kdum-1)) {
	    bindex = Bxi(i,j,k,nx,ny,nz);
	    b0index = Bxi(i,j,(kdum-1),nx,ny,nz);
	    Bx[bindex] = Bx[b0index];
	  }
	}
      }
    }
    for (i=0; i<nx; i++) {
      for (j=0; j<ny+1; j++) {
	for (k=0; k<nz; k++) {
	  if(k != (kdum-1)) {
	    bindex = Byi(i,j,k,nx,ny,nz);
	    b0index = Byi(i,j,(kdum-1),nx,ny,nz);
	    By[bindex] = By[b0index];
	  }
	}
      }
    }
    for (i=0; i<nx; i++) {
      for (j=0; j<ny; j++) {
	for (k=0; k<nz+1; k++) {
	  if(k != kdum) {
	    bindex = Bzi(i,j,k,nx,ny,nz);
	    b0index = Bzi(i,j,kdum,nx,ny,nz);
	    Bz[bindex] = Bz[b0index];
	  }
	}
      }
    }
  }

 //This is where we calculate the coefficients!
  int countval=0;
  int* count = &countval;
  int row;

  //A_x : 
  //The only step -- Coulomb conditions in the x+ direction
  for (i=0; i<nx; i++) {
    for(j=0; j<=ny; j++) {
      for (k=0; k<=nz; k++) {
	row=xi(i,j,k,nx,ny,nz);
        Coulomb(count,row,i+1,j,k,nx,ny,nz,irn,jcn,a);
      }
    }
  }
  
  //A_y :
  
  //Step 1: For x=0, we get n(n+1) Coulomb conditions in the y+ direction
  for (j=0; j<ny; j++) {
    for(k=0; k<=nz; k++) {
      row=yi(0,j,k,nx,ny,nz);
      Coulomb(count,row,0,j+1,k,nx,ny,nz,irn,jcn,a);
    }
  }

  //Step 2: Fill out the remaining z faces in the x- direction
  for (i=1; i<=nx; i++) {
    for (j=0; j<ny; j++) {
      for (k=0; k<=nz; k++) {
        row=yi(i,j,k,nx,ny,nz);
        zface(count,row,i-1,j,k,nx,ny,nz,dx,dy,dz,Bz,irn,jcn,a,rhs);
      }
    }
  }

  //A_z:

  //Step 1: we will have n colulomb conditions in the z+ direction
  for (k=0; k<nz; k++) {
    row = zi(0,0,k,nx,ny,nz);
    Coulomb(count,row,0,0,k+1,nx,ny,nz, irn, jcn, a);
  }

 //Step 2: we will have n^2 x-face values set at x=0 in the y- direction
  for (j=1; j<=ny; j++) {
    for (k=0; k<nz; k++) {
      row=zi(0,j,k,nx,ny,nz);
      xface(count,row,0,j-1,k,nx,ny,nz,dx,dy,dz,Bx,irn,jcn,a,rhs);
    }
  }

  //Step 3: we will have n*n*(n+1) y-faces to set in the x- direction
  for (i=1; i<=nx; i++) {
    for (j=0; j<=ny; j++) {
      for (k=0; k<nz; k++) {
	row=zi(i,j,k,nx,ny,nz);
	yface(count,row,i-1,j,k,nx,ny,nz,dx,dy,dz,By,irn,jcn,a,rhs);
      }
    }
  }

  // Switch counts to Fortran-based 1-initialized indexing!
  for (i=0; i<nvals; i++) {
    irn[i]+=1;
    jcn[i]+=1;
  }


  MUMPS_INT myid, ierr;
#if defined(MAIN_COMP)
  argv = &name;
#endif
  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  /* Initialize a MUMPS instance. Use MPI_COMM_WORLD */
  id.job=JOB_INIT; id.par=1; id.sym=0;id.comm_fortran=USE_COMM_WORLD;
  dmumps_c(&id);
  /* Define the problem on the host */
  if (myid == 0) {
    id.n = nrows; id.nz =nvals; id.irn=irn; id.jcn=jcn;
    id.a = a; id.rhs = rhs;
  }
#define ICNTL(I) icntl[(I)-1] /* macro s.t. indices match documentation */
/* No outputs */
  id.ICNTL(1)=6; id.ICNTL(2)=6; id.ICNTL(3)=6; id.ICNTL(4)=3;

  //This command helps locally to fix a bug in ScaLapack/MPICH1 compatibility
  id.ICNTL(13)=1;

  //This command might help when there is not enough memory in the workarrays
  id.ICNTL(14)=40;

/* Call the MUMPS package. */
  id.job=6;
  dmumps_c(&id);

  if (myid == 0) {
    for(i=0; i<10; i++) {
      printf("Solution is : (%i %f)\n", i,rhs[i]);
    }
  }

/*   if (myid == 0) { */
    
/*     FILE* outfile = fopen("afields.dat", "w"); */
/*     for (i=0; i<n/3; i++) { */
/*       fprintf(outfile,"%d %.15g %.15g %.15g\n",i,rhs[3*i],rhs[3*i+1],rhs[3*i+2]); */
/*     } */
/*     fclose(outfile); */
/*   } */

  id.job=JOB_END; dmumps_c(&id); /* Terminate instance */

  ierr = MPI_Finalize();

  if (myid == 0) {
    
    FILE* outfile = fopen("afields.dat", "w");
    FILE* outfilex = fopen("Ax_stagger.xyz_xy400_sched.dat", "w");
    FILE* outfiley = fopen("Ay_stagger.xyz_xy400_sched.dat", "w");
    FILE* outfilez = fopen("Az_stagger.xyz_xy400_sched.dat", "w");

    for (i=0; i<nrows/3; i++) {
      fprintf(outfile,"%d %.15g %.15g %.15g\n",i,rhs[i],rhs[i+nrows/3],rhs[i+2*nrows/3]);
    }

    int index=0;
    int aindex, edge_index, prev_index;
    double outx, outy, outz;

    //We have to extrapolate the data so all
    //3 fields are the same size

    double *Ax;
    double *Ay;
    double *Az;

    Ax = (double *) malloc((nx+1)*(ny+1)*(nz+1)*sizeof(double));
    Ay = (double *) malloc((nx+1)*(ny+1)*(nz+1)*sizeof(double));
    Az = (double *) malloc((nx+1)*(ny+1)*(nz+1)*sizeof(double));
    
    int ext_order_flag = 3;
    int ext_index, index1, index2, index3, index4;

    //First populate the bigger arrays in
    //preparation for the necessary
    //extrapolations.

    for (i=0; i<nx; i++) {
      for (j=0; j<ny+1; j++) {
	for (k=0; k<nz+1; k++) {
	  index = xi(i,j,k,nx,ny,nz);
	  aindex = Ai((i+1),j,k,nx,ny,nz);
	  Ax[aindex] = rhs[index];
	}
      }
    }
    for (i=0; i<nx+1; i++) {
      for (j=0; j<ny; j++) {
	for (k=0; k<nz+1; k++) {
	  index = yi(i,j,k,nx,ny,nz);
	  aindex = Ai(i,(j+1),k,nx,ny,nz);
	  Ay[aindex] = rhs[index];
	}
      }
    }
    for (i=0; i<nx+1; i++) {
      for (j=0; j<ny+1; j++) {
	for (k=0; k<nz; k++) {
	  index = zi(i,j,k,nx,ny,nz);
	  aindex = Ai(i,j,(k+1),nx,ny,nz);
	  Az[aindex] = rhs[index];
	}
      }
    }

    //Now extrapolate the fields to make
    //them all the same size.

    for (j=0; j<ny+1; j++) {
      for (k=0; k<nz+1; k++) {
	i = 0;
	aindex = Ai(i,j,k,nx,ny,nz);
	edge_index = Ai((i+1),j,k,nx,ny,nz);
	prev_index = Ai((i+2),j,k,nx,ny,nz);
	Ax[aindex] = 2*Ax[edge_index] - Ax[prev_index];
      }
    }
    for (i=0; i<nx+1; i++) {
      for (k=0; k<nz+1; k++) {
	j = 0;
	aindex = Ai(i,j,k,nx,ny,nz);
	edge_index = Ai(i,(j+1),k,nx,ny,nz);
	prev_index = Ai(i,(j+2),k,nx,ny,nz);
	Ay[aindex] = 2*Ay[edge_index] - Ay[prev_index];
      }
    }
    for (i=0; i<nx+1; i++) {
      for (j=0; j<ny+1; j++) {
	k = 0;
	aindex = Ai(i,j,k,nx,ny,nz);
	edge_index = Ai(i,j,(k+1),nx,ny,nz);
	prev_index = Ai(i,j,(k+2),nx,ny,nz);
	Az[aindex] = 2*Az[edge_index] - Az[prev_index];
      }
    }

    //Now anticipate the boundary conditions
    //of IllinoisGRMHD by extrapolating in
    //the ghost zones

    if (ext_order_flag == 0)
    {
      //Lower x-Face

      for(j=0; j<ny+1; j++)
      {
	for(k=0; k<nz+1; k++)
	{
	  i = n_ghost;
	  ext_index = k+(nz+1)*(j+(ny+1)*i);
	  Ax[ext_index] = 0.0;

	  for(l=0; l<n_ghost; l++)
	  {
	    i = n_ghost-(l+1);
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = k+(nz+1)*(j+(ny+1)*(i+1));

	    Ax[ext_index] = 0.0;
	  
	    Ay[ext_index] = Ay[index1];

	    Az[ext_index] = Az[index1];
	  }
	}
      }
  
      //Upper x-Face

      for(j=0; j<ny+1; j++)
      {
	for(k=0; k<nz+1; k++)
	{
	  for(l=0; l<n_ghost; l++)
	  {
	    i = (nx+1)-n_ghost+l;
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = k+(nz+1)*(j+(ny+1)*(i-1));

	    Ax[ext_index] = 0.0;
	  
	    Ay[ext_index] = Ay[index1];

	    Az[ext_index] = Az[index1];
	  }
	}
      }
  
      //Lower y-Face

      for(i=0; i<nx+1; i++)
      {
	for(k=0; k<nz+1; k++)
	{
	  j = n_ghost;
	  ext_index = k+(nz+1)*(j+(ny+1)*i);
	  Ay[ext_index] = 0.0;

	  for(l=0; l<n_ghost; l++)
	  {
	    j = n_ghost-(l+1);
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = k+(nz+1)*((j+1)+(ny+1)*i);

	    Ay[ext_index] = 0.0;
	  
	    Az[ext_index] = Az[index1];

	    Ax[ext_index] = Ax[index1];
	  }
	}
      }
  
      //Upper y-Face

      for(i=0; i<nx+1; i++)
      {
	for(k=0; k<nz+1; k++)
	{
	  for(l=0; l<n_ghost; l++)
	  {
	    j = (ny+1)-n_ghost+l;
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = k+(nz+1)*((j-1)+(ny+1)*i);

	    Ay[ext_index] = 0.0;
	  
	    Az[ext_index] = Az[index1];

	    Ax[ext_index] = Ax[index1];
	  }
	}
      }
  
      //Lower z-Face

      for(i=0; i<nx+1; i++)
      {
	for(j=0; j<ny+1; j++)
	{
	  k = n_ghost;
	  ext_index = k+(nz+1)*(j+(ny+1)*i);
	  Az[ext_index] = 0.0;

	  for(l=0; l<n_ghost; l++)
	    {
	      k = n_ghost-(l+1);
	      ext_index = k+(nz+1)*(j+(ny+1)*i);
	      index1 = (k+1)+(nz+1)*(j+(ny+1)*i);

	      Az[ext_index] = 0.0;
	  
	      Ax[ext_index] = Ax[index1];

	      Ay[ext_index] = Ay[index1];
	    }
	}
      }
    
      //Upper z-Face

      for(i=0; i<nx+1; i++)
      {
	for(j=0; j<ny+1; j++)
	{
	  for(l=0; l<n_ghost; l++)
	  {
	    k = (nz+1)-(n_ghost)+l;
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = (k-1)+(nz+1)*(j+(ny+1)*i);

	    Az[ext_index] = 0.0;
	  
	    Ax[ext_index] = Ax[index1];

	    Ay[ext_index] = Ay[index1];
	  }
	}
      }
    }

    else if (ext_order_flag == 1)
    {
      //Lower x-Face

      for(j=0; j<ny+1; j++)
      {
	for(k=0; k<nz+1; k++)
	{
	  i = n_ghost;
	  ext_index = k+(nz+1)*(j+(ny+1)*i);
	  index1 = k+(nz+1)*(j+(ny+1)*(i+1));
	  Ax[ext_index] = Ax[index1];

	  for(l=0; l<n_ghost; l++)
	  {
	    i = n_ghost-(l+1);
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = k+(nz+1)*(j+(ny+1)*(i+1));
	    index2 = k+(nz+1)*(j+(ny+1)*(i+2));

	    Ax[ext_index] = Ax[index1];
	  
	    Ay[ext_index] = 2*Ay[index1] 
	                  - 1*Ay[index2];

	    Az[ext_index] = 2*Az[index1] 
	                  - 1*Az[index2];
	  }
	}
      }
  
      //Upper x-Face

      for(j=0; j<ny+1; j++)
      {
	for(k=0; k<nz+1; k++)
	{
	  for(l=0; l<n_ghost; l++)
	  {
	    i = (nx+1)-n_ghost+l;
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = k+(nz+1)*(j+(ny+1)*(i-1));
	    index2 = k+(nz+1)*(j+(ny+1)*(i-2));

	    Ax[ext_index] = Ax[index1];
	  
	    Ay[ext_index] = 2*Ay[index1] 
	                  - 1*Ay[index2];

	    Az[ext_index] = 2*Az[index1] 
	                  - 1*Az[index2];
	  }
	}
      }
  
      //Lower y-Face

      for(i=0; i<nx+1; i++)
      {
	for(k=0; k<nz+1; k++)
	{
	  j = n_ghost;
	  ext_index = k+(nz+1)*(j+(ny+1)*i);
	  index1 = k+(nz+1)*((j+1)+(ny+1)*i);
	  Ay[ext_index] = Ay[index1];

	  for(l=0; l<n_ghost; l++)
	  {
	    j = n_ghost-(l+1);
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = k+(nz+1)*((j+1)+(ny+1)*i);
	    index2 = k+(nz+1)*((j+2)+(ny+1)*i);

	    Ay[ext_index] = Ay[index1];
	  
	    Az[ext_index] = 2*Az[index1] 
	                  - 1*Az[index2];

	    Ax[ext_index] = 2*Ax[index1] 
	                  - 1*Ax[index2];
	  }
	}
      }
  
      //Upper y-Face

      for(i=0; i<nx+1; i++)
      {
	for(k=0; k<nz+1; k++)
	{
	  for(l=0; l<n_ghost; l++)
	  {
	    j = (ny+1)-n_ghost+l;
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = k+(nz+1)*((j-1)+(ny+1)*i);
	    index2 = k+(nz+1)*((j-2)+(ny+1)*i);

	    Ay[ext_index] = Ay[index1];
	  
	    Az[ext_index] = 2*Az[index1] 
	                  - 1*Az[index2];

	    Ax[ext_index] = 2*Ax[index1] 
	                  - 1*Ax[index2];
	  }
	}
      }
  
      //Lower z-Face

      for(i=0; i<nx+1; i++)
      {
	for(j=0; j<ny+1; j++)
	{
	  k = n_ghost;
	  ext_index = k+(nz+1)*(j+(ny+1)*i);
	  index1 = (k+1)+(nz+1)*(j+(ny+1)*i);
	  Az[ext_index] = Az[index1];

	  for(l=0; l<n_ghost; l++)
	  {
	    k = n_ghost-(l+1);
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = (k+1)+(nz+1)*(j+(ny+1)*i);
	    index2 = (k+2)+(nz+1)*(j+(ny+1)*i);

	    Az[ext_index] = Az[index1];
	  
	    Ax[ext_index] = 2*Ax[index1] 
	                  - 1*Ax[index2];

	    Ay[ext_index] = 2*Ay[index1] 
	                  - 1*Ay[index2];
	  }
	}
      }
    
      //Upper z-Face

      for(i=0; i<nx+1; i++)
      {
	for(j=0; j<ny+1; j++)
	{
	  for(l=0; l<n_ghost; l++)
	  {
	    k = (nz+1)-(n_ghost)+l;
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = (k-1)+(nz+1)*(j+(ny+1)*i);
	    index2 = (k-2)+(nz+1)*(j+(ny+1)*i);

	    Az[ext_index] = Az[index1];
	  
	    Ax[ext_index] = 2*Ax[index1] 
	                  - 1*Ax[index2];

	    Ay[ext_index] = 2*Ay[index1] 
	                  - 1*Ay[index2];
	  }
	}
      }
    }

    else if (ext_order_flag == 2)
    {
      //Lower x-Face

      for(j=0; j<ny+1; j++)
      {
	for(k=0; k<nz+1; k++)
	{
	  i = n_ghost;
	  ext_index = k+(nz+1)*(j+(ny+1)*i);
	  index1 = k+(nz+1)*(j+(ny+1)*(i+1));
	  index2 = k+(nz+1)*(j+(ny+1)*(i+2));
	  Ax[ext_index] = 2*Ax[index1] 
	                - 1*Ax[index2];

	  for(l=0; l<n_ghost; l++)
	  {
	    i = n_ghost-(l+1);
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = k+(nz+1)*(j+(ny+1)*(i+1));
	    index2 = k+(nz+1)*(j+(ny+1)*(i+2));
	    index3 = k+(nz+1)*(j+(ny+1)*(i+3));

	    Ax[ext_index] = 2*Ax[index1] 
	                  - 1*Ax[index2];

	    Ay[ext_index] = 3*Ay[index1] 
	                  - 3*Ay[index2]
	                  + 1*Ay[index3];

	    Az[ext_index] = 3*Az[index1] 
	                  - 3*Az[index2]
	                  + 1*Az[index3];
	  }
	}
      }
  
      //Upper x-Face

      for(j=0; j<ny+1; j++)
      {
	for(k=0; k<nz+1; k++)
	{
	  for(l=0; l<n_ghost; l++)
	  {
	    i = (nx+1)-n_ghost+l;
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = k+(nz+1)*(j+(ny+1)*(i-1));
	    index2 = k+(nz+1)*(j+(ny+1)*(i-2));
	    index3 = k+(nz+1)*(j+(ny+1)*(i-3));

	    Ax[ext_index] = 2*Ax[index1] 
	                  - 1*Ax[index2];

	    Ay[ext_index] = 3*Ay[index1] 
	                  - 3*Ay[index2]
	                  + 1*Ay[index3];

	    Az[ext_index] = 3*Az[index1] 
	                  - 3*Az[index2]
	                  + 1*Az[index3];
	  }
	}
      }
  
      //Lower y-Face

      for(i=0; i<nx+1; i++)
      {
	for(k=0; k<nz+1; k++)
	{
	  j = n_ghost;
	  ext_index = k+(nz+1)*(j+(ny+1)*i);
	  index1 = k+(nz+1)*((j+1)+(ny+1)*i);
	  index2 = k+(nz+1)*((j+2)+(ny+1)*i);
	  Ay[ext_index] = 2*Ay[index1] 
	                - 1*Ay[index2];

	  for(l=0; l<n_ghost; l++)
	  {
	    j = n_ghost-(l+1);
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = k+(nz+1)*((j+1)+(ny+1)*i);
	    index2 = k+(nz+1)*((j+2)+(ny+1)*i);
	    index3 = k+(nz+1)*((j+3)+(ny+1)*i);

	    Ay[ext_index] = 2*Ay[index1] 
	                  - 1*Ay[index2];

	    Az[ext_index] = 3*Az[index1] 
	                  - 3*Az[index2]
	                  + 1*Az[index3];

	    Ax[ext_index] = 3*Ax[index1] 
	                  - 3*Ax[index2]
	                  + 1*Ax[index3];
	  }
	}
      }
  
      //Upper y-Face

      for(i=0; i<nx+1; i++)
      {
	for(k=0; k<nz+1; k++)
	{
	  for(l=0; l<n_ghost; l++)
	  {
	    j = (ny+1)-n_ghost+l;
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = k+(nz+1)*((j-1)+(ny+1)*i);
	    index2 = k+(nz+1)*((j-2)+(ny+1)*i);
	    index3 = k+(nz+1)*((j-3)+(ny+1)*i);

	    Ay[ext_index] = 2*Ay[index1] 
	                  - 1*Ay[index2];

	    Az[ext_index] = 3*Az[index1] 
	                  - 3*Az[index2]
	                  + 1*Az[index3];

	    Ax[ext_index] = 3*Ax[index1] 
	                  - 3*Ax[index2]
	                  + 1*Ax[index3];
	  }
	}
      }
  
      //Lower z-Face

      for(i=0; i<nx+1; i++)
      {
	for(j=0; j<ny+1; j++)
	{
	  k = n_ghost;
	  ext_index = k+(nz+1)*(j+(ny+1)*i);
	  index1 = (k+1)+(nz+1)*(j+(ny+1)*i);
	  index2 = (k+2)+(nz+1)*(j+(ny+1)*i);
	  Az[ext_index] = 2*Az[index1] 
	                - 1*Az[index2];

	  for(l=0; l<n_ghost; l++)
	  {
	    k = n_ghost-(l+1);
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = (k+1)+(nz+1)*(j+(ny+1)*i);
	    index2 = (k+2)+(nz+1)*(j+(ny+1)*i);
	    index3 = (k+3)+(nz+1)*(j+(ny+1)*i);

	    Az[ext_index] = 2*Az[index1] 
	                  - 1*Az[index2];

	    Ax[ext_index] = 3*Ax[index1] 
	                  - 3*Ax[index2]
	                  + 1*Ax[index3];

	    Ay[ext_index] = 3*Ay[index1] 
	                  - 3*Ay[index2]
	                  + 1*Ay[index3];
	  }
	}
      }
    
      //Upper z-Face

      for(i=0; i<nx+1; i++)
      {
	for(j=0; j<ny+1; j++)
	{
	  for(l=0; l<n_ghost; l++)
	  {
	    k = (nz+1)-(n_ghost)+l;
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = (k-1)+(nz+1)*(j+(ny+1)*i);
	    index2 = (k-2)+(nz+1)*(j+(ny+1)*i);
	    index3 = (k-3)+(nz+1)*(j+(ny+1)*i);

	    Az[ext_index] = 2*Az[index1] 
	                  - 1*Az[index2];

	    Ax[ext_index] = 3*Ax[index1] 
	                  - 3*Ax[index2]
	                  + 1*Ax[index3];

	    Ay[ext_index] = 3*Ay[index1] 
	                  - 3*Ay[index2]
	                  + 1*Ay[index3];
	  }
	}
      }
    }

    else if (ext_order_flag == 3)
    {
      //Lower x-Face

      for(j=0; j<ny+1; j++)
      {
	for(k=0; k<nz+1; k++)
	{
	  i = n_ghost;
	  ext_index = k+(nz+1)*(j+(ny+1)*i);
	  index1 = k+(nz+1)*(j+(ny+1)*(i+1));
	  index2 = k+(nz+1)*(j+(ny+1)*(i+2));	
	  index3 = k+(nz+1)*(j+(ny+1)*(i+3));
	  Ax[ext_index] = 3*Ax[index1] 
	                - 3*Ax[index2]
	                + 1*Ax[index3];

	  for(l=0; l<n_ghost; l++)
	  {
	    i = n_ghost-(l+1);
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = k+(nz+1)*(j+(ny+1)*(i+1));
	    index2 = k+(nz+1)*(j+(ny+1)*(i+2));
	    index3 = k+(nz+1)*(j+(ny+1)*(i+3));
	    index4 = k+(nz+1)*(j+(ny+1)*(i+4));

	    Ax[ext_index] = 3*Ax[index1] 
	                  - 3*Ax[index2]
	                  + 1*Ax[index3];

	    Ay[ext_index] = 4*Ay[index1] 
	                  - 6*Ay[index2]
	                  + 4*Ay[index3]
	                  - 1*Ay[index4];

	    Az[ext_index] = 4*Az[index1] 
	                  - 6*Az[index2]
	                  + 4*Az[index3]
	                  - 1*Az[index4];
	  }
	}
      }
  
      //Upper x-Face

      for(j=0; j<ny+1; j++)
      {
	for(k=0; k<nz+1; k++)
	{
	  for(l=0; l<n_ghost; l++)
	  {
	    i = (nx+1)-n_ghost+l;
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = k+(nz+1)*(j+(ny+1)*(i-1));
	    index2 = k+(nz+1)*(j+(ny+1)*(i-2));
	    index3 = k+(nz+1)*(j+(ny+1)*(i-3));
	    index4 = k+(nz+1)*(j+(ny+1)*(i-4));

	    Ax[ext_index] = 3*Ax[index1] 
	                  - 3*Ax[index2]
	                  + 1*Ax[index3];

	    Ay[ext_index] = 4*Ay[index1] 
	                  - 6*Ay[index2]
	                  + 4*Ay[index3]
	                  - 1*Ay[index4];

	    Az[ext_index] = 4*Az[index1] 
	                  - 6*Az[index2]
	                  + 4*Az[index3]
	                  - 1*Az[index4];
	  }
	}
      }
  
      //Lower y-Face

      for(i=0; i<nx+1; i++)
      {
	for(k=0; k<nz+1; k++)
	{
	  j = n_ghost;
	  ext_index = k+(nz+1)*(j+(ny+1)*i);
	  index1 = k+(nz+1)*((j+1)+(ny+1)*i);
	  index2 = k+(nz+1)*((j+2)+(ny+1)*i);
	  index3 = k+(nz+1)*((j+3)+(ny+1)*i);
	  Ay[ext_index] = 3*Ay[index1] 
	                - 3*Ay[index2]
	                + 1*Ay[index3];

	  for(l=0; l<n_ghost; l++)
	  {
	    j = n_ghost-(l+1);
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = k+(nz+1)*((j+1)+(ny+1)*i);
	    index2 = k+(nz+1)*((j+2)+(ny+1)*i);
	    index3 = k+(nz+1)*((j+3)+(ny+1)*i);
	    index4 = k+(nz+1)*((j+4)+(ny+1)*i);

	    Ay[ext_index] = 3*Ay[index1] 
	                  - 3*Ay[index2]
	                  + 1*Ay[index3];

	    Az[ext_index] = 4*Az[index1] 
	                  - 6*Az[index2]
	                  + 4*Az[index3]
	                  - 1*Az[index4];

	    Ax[ext_index] = 4*Ax[index1] 
	                  - 6*Ax[index2]
	                  + 4*Ax[index3]
	                  - 1*Ax[index4];
	  }
	}
      }
  
      //Upper y-Face

      for(i=0; i<nx+1; i++)
      {
	for(k=0; k<nz+1; k++)
	{
	  for(l=0; l<n_ghost; l++)
	  {
	    j = (ny+1)-n_ghost+l;
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = k+(nz+1)*((j-1)+(ny+1)*i);
	    index2 = k+(nz+1)*((j-2)+(ny+1)*i);
	    index3 = k+(nz+1)*((j-3)+(ny+1)*i);
	    index4 = k+(nz+1)*((j-4)+(ny+1)*i);

	    Ay[ext_index] = 3*Ay[index1] 
	                  - 3*Ay[index2]
	                  + 1*Ay[index3];

	    Az[ext_index] = 4*Az[index1] 
	                  - 6*Az[index2]
	                  + 4*Az[index3]
	                  - 1*Az[index4];

	    Ax[ext_index] = 4*Ax[index1] 
	                  - 6*Ax[index2]
	                  + 4*Ax[index3]
	                  - 1*Ax[index4];
	  }
	}
      }
  
      //Lower z-Face

      for(i=0; i<nx+1; i++)
      {
	for(j=0; j<ny+1; j++)
	{
	  k = n_ghost;
	  ext_index = k+(nz+1)*(j+(ny+1)*i);
	  index1 = (k+1)+(nz+1)*(j+(ny+1)*i);
	  index2 = (k+2)+(nz+1)*(j+(ny+1)*i);
	  index3 = (k+3)+(nz+1)*(j+(ny+1)*i);
	  Az[ext_index] = 3*Az[index1] 
	                - 3*Az[index2]
	                + 1*Az[index3];

	  for(l=0; l<n_ghost; l++)
	  {
	    k = n_ghost-(l+1);
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = (k+1)+(nz+1)*(j+(ny+1)*i);
	    index2 = (k+2)+(nz+1)*(j+(ny+1)*i);
	    index3 = (k+3)+(nz+1)*(j+(ny+1)*i);
	    index4 = (k+4)+(nz+1)*(j+(ny+1)*i);

	    Az[ext_index] = 3*Az[index1] 
	                  - 3*Az[index2]
	                  + 1*Az[index3];

	    Ax[ext_index] = 4*Ax[index1] 
	                  - 6*Ax[index2]
	                  + 4*Ax[index3]
	                      - 1*Ax[index4];

	    Ay[ext_index] = 4*Ay[index1] 
	                  - 6*Ay[index2]
	                  + 4*Ay[index3]
	                  - 1*Ay[index4];
	  }
	}
      }
    
      //Upper z-Face

      for(i=0; i<nx+1; i++)
      {
	for(j=0; j<ny+1; j++)
	{
	  for(l=0; l<n_ghost; l++)
	  {
	    k = (nz+1)-(n_ghost)+l;
	    ext_index = k+(nz+1)*(j+(ny+1)*i);
	    index1 = (k-1)+(nz+1)*(j+(ny+1)*i);
	    index2 = (k-2)+(nz+1)*(j+(ny+1)*i);
	    index3 = (k-3)+(nz+1)*(j+(ny+1)*i);
	    index4 = (k-4)+(nz+1)*(j+(ny+1)*i);

	    Az[ext_index] = 3*Az[index1] 
	                  - 3*Az[index2]
	                  + 1*Az[index3];

	    Ax[ext_index] = 4*Ax[index1] 
	                  - 6*Ax[index2]
	                  + 4*Ax[index3]
	                  - 1*Ax[index4];

	    Ay[ext_index] = 4*Ay[index1] 
	                  - 6*Ay[index2]
	                  + 4*Ay[index3]
	                  - 1*Ay[index4];
	  }
	}
      }
    }

    //Output the extrapolated fields for
    //use in IllinoisGRMHD

    fprintf(outfilex,"# time 2.8\n");
    fprintf(outfilex,"# Nx: %i   Ny: %i   Nz: %i\n",(nx+1),(ny+1),(nz+1));
    fprintf(outfilex,"# column format: 1:ix   2:iy   3:iz   4:x   5:y   6:z   7:data\n");
    for (k=0; k<nz+1; k++) {
      for (j=0; j<ny+1; j++) {
	for (i=0; i<nx+1; i++) {
	  outx=xinit+i*dx;
	  outy=yinit+j*dy;
	  outz=zinit+k*dz;
	  aindex=Ai(i,j,k,nx,ny,nz);
	  fprintf(outfilex,"%i %i %i %.16e %.16e %.16e %.16e\n",i,j,k,outx,outy,outz,Ax[aindex]);
	}
      }
    }
    fprintf(outfiley,"# time 2.8\n");
    fprintf(outfiley,"# Nx: %i   Ny: %i   Nz: %i\n",(nx+1),(ny+1),(nz+1));
    fprintf(outfiley,"# column format: 1:ix   2:iy   3:iz   4:x   5:y   6:z   7:data\n");
    for (k=0; k<nz+1; k++) {
      for (j=0; j<ny+1; j++) {
	for (i=0; i<nx+1; i++) {
	  outx=xinit+i*dx;
	  outy=yinit+j*dy;
	  outz=zinit+k*dz;
	  aindex=Ai(i,j,k,nx,ny,nz);
	  fprintf(outfiley,"%i %i %i %.16e %.16e %.16e %.16e\n",i,j,k,outx,outy,outz,Ay[aindex]);
	}
      }
    }
    fprintf(outfilez,"# time 2.8\n");
    fprintf(outfilez,"# Nx: %i   Ny: %i   Nz: %i\n",(nx+1),(ny+1),(nz+1));
    fprintf(outfilez,"# column format: 1:ix   2:iy   3:iz   4:x   5:y   6:z   7:data\n");
    for (k=0; k<nz+1; k++) {
      for (j=0; j<ny+1; j++) {
	for (i=0; i<nx+1; i++) {
	  outx=xinit+i*dx;
	  outy=yinit+j*dy;
	  outz=zinit+k*dz;
	  aindex=Ai(i,j,k,nx,ny,nz);
	  fprintf(outfilez,"%i %i %i %.16e %.16e %.16e %.16e\n",i,j,k,outx,outy,outz,Az[aindex]);
	}
      }
    }

    fclose(outfile);
    fclose(outfilex);
    fclose(outfiley);
    fclose(outfilez);

    //Now we should do some error checking!

    int ncoulomb = 0; 
    int nbfield = 0;
    double errcoulomb = 0.0;
    double errbfield = 0.0;

    double relerrbfield=0.0;
    double relerrcoulomb=0.0;


    //A_x :
    //The only step -- Coulomb conditions in the x+ direction
    for (i=0; i<nx; i++) {
      for(j=0; j<=ny; j++) {
	for (k=0; k<=nz; k++) {
	  ncoulomb++;
	  errcoulomb += pow(Coulomb_error(i+1,j,k,nx,ny,nz,rhs),2);
	  relerrcoulomb += pow(Relative_Coulomb_error(i+1,j,k,nx,ny,nz,rhs),2);
	}
      }
    }
    
    //A_y :
    //Step 1: For x=0, we get n(n+1) Coulomb conditions in the y+ direction
    for (j=0; j<ny; j++) {
      for (k=0; k<=nz; k++) {
	ncoulomb++;
	errcoulomb += pow(Coulomb_error(0,j+1,k,nx,ny,nz,rhs),2);
	relerrcoulomb += pow(Relative_Coulomb_error(0,j+1,k,nx,ny,nz,rhs),2);
      }
    }
    
    //Step 2: Fill out the remaining z faces in the x- direction
    for (i=1; i<=nx; i++) {
      for (j=0; j<ny; j++) {
	for (k=0; k<=nz; k++) {
	  nbfield++;
	  errbfield += pow(zface_error(i-1,j,k,nx,ny,nz,dx,dy,dz,Bz,rhs),2);
	  relerrbfield += pow(Relative_zface_error(i-1,j,k,nx,ny,nz,dx,dy,dz,Bz,rhs),2);
	}
      }
    }
  
    //A_z: 

    //Step 1
    for (k=0; k<nz; k++) {
      ncoulomb++;
      errcoulomb += pow(Coulomb_error(0,0,k+1,nx,ny,nz, rhs),2);
      relerrcoulomb += pow(Relative_Coulomb_error(0,0,k+1,nx,ny,nz, rhs),2);
    }
    
    //Step 2: we will have n^2 x-face values set at x=0 in the y- direction
    for (j=1; j<=ny; j++) {
      for (k=0; k<nz; k++) {
	nbfield++;
	errbfield += pow(xface_error(0,j-1,k,nx,ny,nz,dx,dy,dz,Bx,rhs),2);
	relerrbfield += pow(Relative_xface_error(0,j-1,k,nx,ny,nz,dx,dy,dz,Bx,rhs),2);
      }
    }
    
    //Step 3: we will have n*n*n+1 y-faces to set in the x- direction
    for (i=1; i<=nx; i++) {
      for (j=0; j<=ny; j++) {
	for (k=0; k<nz; k++) {
	  nbfield++;
	  errbfield += pow(yface_error(i-1,j,k,nx,ny,nz,dx,dy,dz,By,rhs),2);
	  relerrbfield += pow(Relative_yface_error(i-1,j,k,nx,ny,nz,dx,dy,dz,By,rhs),2);
	}
      }
    }
       
    printf("Coulomb errors: %d %g\n", ncoulomb,sqrt(errcoulomb/(1.0*ncoulomb)));
    printf("B-field errors: %d %g\n", nbfield,sqrt(errbfield/(1.0*nbfield)));
    printf("Relative Coulomb errors: %d %g\n", ncoulomb,sqrt(relerrcoulomb/(1.0*ncoulomb)));
    printf("Relative B-field errors: %d %g\n", nbfield,sqrt(relerrbfield/(1.0*nbfield)));
    
  } 
  
  return 0;
}    
  
 void xface(int *count, int row, int i, int j, int k, int nx, int ny, int nz, double dx, double dy, double dz, 
	    double* bval, int* irn, int* jcn, double* a, double* rhs) 
{

    irn[*count] = row;
    jcn[*count] = zi(i,j+1,k,nx,ny,nz);
    a[*count] = 1.0;
    (*count)++;

    irn[*count] = row;
    jcn[*count] = zi(i,j,k,nx,ny,nz);
    a[*count] = -1.0;
    (*count)++;

    irn[*count] = row;
    jcn[*count] = yi(i,j,k+1,nx,ny,nz);
    a[*count] = -1.0;
    (*count)++;

    irn[*count] = row;
    jcn[*count] = yi(i,j,k,nx,ny,nz);
    a[*count] = 1.0;
    (*count)++;
  
    rhs[row]=bval[Bxi(i,j,k,nx,ny,nz)]*dx;
}

void yface(int *count, int row, int i, int j, int k, int nx, int ny, int nz, double dx, double dy, double dz, 
	   double* bval, int* irn, int* jcn, double* a, double* rhs) 
{

  irn[*count] = row;
  jcn[*count] = xi(i,j,k+1,nx,ny,nz);
  a[*count] = 1.0;
  (*count)++;

  irn[*count] = row;
  jcn[*count] = xi(i,j,k,nx,ny,nz);
  a[*count] = -1.0;
  (*count)++;

  irn[*count] = row;
  jcn[*count] = zi(i+1,j,k,nx,ny,nz);
  a[*count] = -1.0;
  (*count)++;

  irn[*count] = row;
  jcn[*count] = zi(i,j,k,nx,ny,nz);
  a[*count] = 1.0;
  (*count)++;

  rhs[row]=bval[Byi(i,j,k,nx,ny,nz)]*dx;
}

void zface(int *count, int row, int i, int j, int k, int nx, int ny, int nz, double dx, double dy, double dz, 
	   double* bval, int* irn, int* jcn, double* a, double* rhs) 
{

  irn[*count] = row;
  jcn[*count] = yi(i+1,j,k,nx,ny,nz);
  a[*count] = 1.0;
  (*count)++;


  irn[*count] = row;
  jcn[*count] = yi(i,j,k,nx,ny,nz);
  a[*count] = -1.0;
  (*count)++;

  irn[*count] = row;
  jcn[*count] = xi(i,j+1,k,nx,ny,nz);
  a[*count] = -1.0;
  (*count)++;

  irn[*count] = row;
  jcn[*count] = xi(i,j,k,nx,ny,nz);
  a[*count] = 1.0;
  (*count)++;

  rhs[row]=bval[Bzi(i,j,k,nx,ny,nz)]*dx;
}

void Coulomb(int *count, int row, int i, int j, int k, int nx, int ny, int nz, int* irn, int* jcn, double* a)
{
  if(i!=0){
    irn[*count] = row;
    jcn[*count] = xi(i-1,j,k,nx,ny,nz);
    a[*count] = -1.0;
    (*count)++;
  }    

  if(i!=nx){
    irn[*count] = row;
    jcn[*count] = xi(i,j,k,nx,ny,nz);
    a[*count] = 1.0;
    (*count)++;
  }

  if(j!=0){
    irn[*count] = row;
    jcn[*count] = yi(i,j-1,k,nx,ny,nz);
    a[*count] = -1.0;
    (*count)++;
  }

  if(j!=ny){
    irn[*count] = row;
    jcn[*count] = yi(i,j,k,nx,ny,nz);
    a[*count] = 1.0;
    (*count)++;
  }

  if(k!=0){
    irn[*count] = row;
    jcn[*count] = zi(i,j,k-1,nx,ny,nz);
    a[*count] = -1.0;
    (*count)++;
  }

  if(k!=nz){
    irn[*count] = row;
    jcn[*count] = zi(i,j,k,nx,ny,nz);
    a[*count] = 1.0;
    (*count)++;
  }
}

double xface_error(int i, int j, int k, int nx, int ny, int nz, double dx, double dy, double dz, double* bval, double* rhs) 
{
  double error = rhs[zi(i,j+1,k,nx,ny,nz)] - rhs[zi(i,j,k,nx,ny,nz)] - rhs[yi(i,j,k+1,nx,ny,nz)] + rhs[yi(i,j,k,nx,ny,nz)] - bval[Bxi(i,j,k,nx,ny,nz)]*dx;
  return error;
}

double yface_error(int i, int j, int k, int nx, int ny, int nz, double dx, double dy, double dz, double* bval, double* rhs) 
{
  double error = rhs[xi(i,j,k+1,nx,ny,nz)] - rhs[xi(i,j,k,nx,ny,nz)] - rhs[zi(i+1,j,k,nx,ny,nz)] + rhs[zi(i,j,k,nx,ny,nz)] - bval[Byi(i,j,k,nx,ny,nz)]*dx;
  return error;
}

double zface_error(int i, int j, int k, int nx, int ny, int nz, double dx, double dy, double dz, double* bval, double* rhs) 
{
  double error = rhs[yi(i+1,j,k,nx,ny,nz)] - rhs[yi(i,j,k,nx,ny,nz)] - rhs[xi(i,j+1,k,nx,ny,nz)] + rhs[xi(i,j,k,nx,ny,nz)] - bval[Bzi(i,j,k,nx,ny,nz)]*dx;
  return error;
}

double Coulomb_error(int i, int j, int k, int nx, int ny, int nz, double* rhs)
{
  
  double error = 0.0;
  if(i!=0)error -= rhs[xi(i-1,j,k,nx,ny,nz)];
  if(i!=nx)error += rhs[xi(i,j,k,nx,ny,nz)];
  if(j!=0)error -= rhs[yi(i,j-1,k,nx,ny,nz)];
  if(j!=ny)error += rhs[yi(i,j,k,nx,ny,nz)];
  if(k!=0)error -= rhs[zi(i,j,k-1,nx,ny,nz)];
  if(k!=nz)error += rhs[zi(i,j,k,nx,ny,nz)];

  return error;

}

double Relative_xface_error(int i, int j, int k, int nx, int ny, int nz, double dx, double dy, double dz, 
			    double* bval, double* rhs) 
{
  double error = fabs((rhs[zi(i,j+1,k,nx,ny,nz)] - rhs[zi(i,j,k,nx,ny,nz)] - rhs[yi(i,j,k+1,nx,ny,nz)] + rhs[yi(i,j,k,nx,ny,nz)] - bval[Bxi(i,j,k,nx,ny,nz)]*dx)
		      /(fabs(bval[Bxi(i,j,k,nx,ny,nz)]*dx)+1.0d-12));
  return error;
}

double Relative_yface_error(int i, int j, int k, int nx, int ny, int nz, double dx, double dy, double dz, 
			    double* bval, double* rhs) 
{
  double error = fabs((rhs[xi(i,j,k+1,nx,ny,nz)] - rhs[xi(i,j,k,nx,ny,nz)] - rhs[zi(i+1,j,k,nx,ny,nz)] + rhs[zi(i,j,k,nx,ny,nz)] - bval[Byi(i,j,k,nx,ny,nz)]*dx)
		      /(fabs(bval[Byi(i,j,k,nx,ny,nz)]*dx)+1.0d-12));
  return error;
}

double Relative_zface_error(int i, int j, int k, int nx, int ny, int nz, double dx, double dy, double dz, 
			    double* bval, double* rhs) 
{
  double error = fabs((rhs[yi(i+1,j,k,nx,ny,nz)] - rhs[yi(i,j,k,nx,ny,nz)] - rhs[xi(i,j+1,k,nx,ny,nz)] + rhs[xi(i,j,k,nx,ny,nz)] - bval[Bzi(i,j,k,nx,ny,nz)]*dx)
		      /(fabs(bval[Bzi(i,j,k,nx,ny,nz)]*dx)+1.0d-12));
  return error;
}

double Relative_Coulomb_error(int i, int j, int k, int nx, int ny, int nz, double* rhs)
{
  
  double error = 0.0;
  double denom = 1.0e-10;
  if(i!=0){
    error -= rhs[xi(i-1,j,k,nx,ny,nz)];
    denom += fabs(rhs[xi(i-1,j,k,nx,ny,nz)]);
  }
  if(i!=nx){
    error += rhs[xi(i,j,k,nx,ny,nz)];
    denom += fabs(rhs[xi(i,j,k,nx,ny,nz)]);
  }
  if(j!=0){
    error -= rhs[yi(i,j-1,k,nx,ny,nz)];
    denom += fabs(rhs[yi(i,j-1,k,nx,ny,nz)]);
  }
  if(j!=ny){
    error += rhs[yi(i,j,k,nx,ny,nz)];
    denom += fabs(rhs[yi(i,j,k,nx,ny,nz)]);
  }
  if(k!=0){
    error -= rhs[zi(i,j,k-1,nx,ny,nz)];
    denom += fabs(rhs[zi(i,j,k-1,nx,ny,nz)]);
  }
  if(k!=nz){
    error += rhs[zi(i,j,k,nx,ny,nz)];
    denom += fabs(rhs[zi(i,j,k,nx,ny,nz)]);
  }
  return error/denom;

}

