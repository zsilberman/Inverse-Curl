//----------------------------------------------------------------------
// Kernel Generator
// Copyright (C) 2018 Joshua A. Faber
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

#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int nx, int ny, int nz);
void saveAsBitmap(const Eigen::VectorXd& x, int n, const char* filename);

int main(int argc, char** argv)
{
  //  int nx = 17;  // size of the image
  //  int ny = 17;  // size of the image
  //  int nz = 17;  // size of the image

  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int nz = atoi(argv[3]);

  int m = nx*ny*nz;  // number of unknows (=number of pixels)
  // Assembly:
  std::vector<T> coefficients;            // list of non-zeros coefficients
  Eigen::VectorXd b(m);                   // the right hand side-vector resulting from the constraints
  buildProblem(coefficients, b, nx, ny,nz);
  SpMat A(m,m);
  A.setFromTriplets(coefficients.begin(), coefficients.end());
  // Solving:


  Eigen::BiCGSTAB<Eigen::SparseMatrix<double> > BCGST;

  //Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double> > BCGST;
  //BCGST.preconditioner().setDroptol(0.1);


  BCGST.compute(A);
  Eigen::VectorXd x = BCGST.solve(b);         // use the factorization to solve for the given right hand side
  // Export the result to a file:
 // saveAsBitmap(x, n, argv[1]);

  int i;
  //  for (i=0; i<nx; i++)std::cout<<"ival:"<<i<<" "<<x[i]<<" "<<1.0/(4.0*M_PI*i)<<std::endl;

  std::ofstream outfile;
  outfile.open("kernel.dat");
  outfile<<nx<<" "<<ny<<" "<<nz<<std::endl;
  for(i=0; i<m; i++)outfile<<std::setprecision(17)<<x[i]<<std::endl;

  outfile.close();

  return 0;
}

void insertCoefficient(int id, int i, int j, int k, 
                       double w, std::vector<T>& coeffs,
                       Eigen::VectorXd& b, int nx, int ny, int nz)
{
  int id1 = i+nx*(j+ny*k);
  coeffs.push_back(T(id,id1,w));       
}

void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int nx, int ny, int nz)
{
  b.setZero();
  b(0) = 1.0;

  // Finally -- the general case
  for(int k=0; k<nz; ++k)
    {
      for(int j=0; j<ny; ++j)
        {
          for(int i=0; i<nx; ++i)
            {
              
              int id = i+nx*(j+ny*k);

              double coef_pt=6.0;

              if(i==0) {
                //Reflect in the x-direction
                insertCoefficient(id, i+1,j,k, -2, coefficients, b, nx, ny, nz);
              } else if (i==nx-1) {
                // Falloff boundary conditions
                double r2pt = 1.0*i*i+j*j+k*k;
                double r2offpt = 1.0*(i+1.0)*(i+1.0)+j*j+k*k;
                coef_pt -= sqrt(r2pt/r2offpt);
                insertCoefficient(id, i-1,j,k, -1, coefficients, b, nx, ny, nz);
              } else {
                insertCoefficient(id, i-1,j,k, -1, coefficients, b, nx, ny, nz);
                insertCoefficient(id, i+1,j,k, -1, coefficients, b, nx, ny, nz);
              }                
                
              if(j==0) {
                //Reflect in the y-direction
                insertCoefficient(id, i,j+1,k, -2, coefficients, b, nx, ny, nz);
              } else if (j==ny-1) {
                // Falloff boundary conditions
                double r2pt = 1.0*i*i+j*j+k*k;
                double r2offpt = 1.0*i*i+(j+1.0)*(j+1.0)+k*k;
                coef_pt -= sqrt(r2pt/r2offpt);
                insertCoefficient(id, i,j-1,k, -1, coefficients, b, nx, ny, nz);
              } else {
                insertCoefficient(id, i,j-1,k, -1, coefficients, b, nx, ny, nz);
                insertCoefficient(id, i,j+1,k, -1, coefficients, b, nx, ny, nz);
              }                
                
              if(k==0) {
                //Reflect in the z-direction
                insertCoefficient(id, i,j,k+1, -2, coefficients, b, nx, ny, nz);
              } else if (k==nz-1) {
                // Falloff boundary conditions
                double r2pt = 1.0*i*i+j*j+k*k;
                double r2offpt = 1.0*i*i+j*j+(k+1.0)*(k+1.0);
                coef_pt -= sqrt(r2pt/r2offpt);
                insertCoefficient(id, i,j,k-1, -1, coefficients, b, nx, ny, nz);
              } else {
                insertCoefficient(id, i,j,k-1, -1, coefficients, b, nx, ny, nz);
                insertCoefficient(id, i,j,k+1, -1, coefficients, b, nx, ny, nz);
              }                
                
              insertCoefficient(id, i,j,k,coef_pt, coefficients, b, nx, ny, nz);
            }
        }
    }
}
