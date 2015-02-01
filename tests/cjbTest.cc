/*
simple test
*/

#include "SOGP.h"
#include <time.h>
#include <sys/time.h>
#include <string.h>

// our previous GP
#include "GP.h"

//Regression mse threshold
#define mseTHRESH 1e-3
//Numerical stability threshold
#define numTHRESH 1e-6

//How many points to run
#define N 10

int main(int argc, char **argv)
{
  int seed = time(NULL);
	bool view = true;

  printf("Seed is %d\n",seed);
  srand(seed);
  
  //The SOGPs
  SOGP *m_SOGP = NULL;
  
  //For calculating error
  double mse=0,mse_base=0;

  //For getting returns
  Matrix mus;

  //For outputting for gnuplot
  FILE *file=NULL;

  //The data
  RowVector inputs(N), outputs(N), testin(N),testout(N);

  // Eigen data
  GP::VectorType  eigInputs(N), eigOutputs(N);

  GP otherGP;

  SOGPParams params;
  params.s20 = 0.5;
  params.capacity = 100;

  dynamic_cast<RBFKernel *>(params.m_kernel)->init( sqrt(10.0) );
  dynamic_cast<RBFKernel *>(params.m_kernel)->setA( 100 );
  dynamic_cast<RBFKernel *>(params.m_kernel)->setC( 100 );

  //Generate data (train and test)
  printf("Generating data y = cos(5x)\n");
  file=fopen("output.txt","w"); 

  for(int n=1;n<=N;n++)
  {
    inputs(n)= -1 + rand()/(RAND_MAX/2.0);
    outputs(n)=cos(5*inputs(n));//sinc?

    eigInputs(n-1) = inputs(n);
    eigOutputs(n-1) = outputs(n);

    testin(n) = -1 + rand()/(RAND_MAX/2.0);
    testout(n) = cos(5*testin(n));
    fprintf(file,"%lf %lf\n",inputs(n),outputs(n));
  }
  fclose(file);

  printf("Other GP\n");
  for(int n=1;n<=N;n++)
    otherGP.updateKnownVariance( inputs(n), outputs(n), params.s20, 0 );
 
  //Test Regression
  printf("Regression test (default settings)\n");
  m_SOGP = new SOGP();

  m_SOGP->setParams(params);

  m_SOGP->addM(inputs,outputs);
  file=fopen("regress.txt","w");

  ColumnVector sigma;
  mus = m_SOGP->predictM(testin, sigma);

  ColumnVector eigSigma(N), eigMu(N);
  for(int n=1;n<=N;n++)
  {
     double vv;
     eigMu(n) = otherGP.predict( testin(n), &vv );
     eigSigma(n) = vv;
  }

  mse=0;
  for(int n=1;n<=N;n++){
    fprintf(file,"%lf %lf %f %f %f\n",testin(n), mus(1,n),
             mus(1,n) + 2*sigma(n),
             eigMu(n) + 2*eigSigma(n),
             eigMu(n) );
    double err = testout(n)-mus(1,n);
  }
  fclose(file);
  
  //Display for the user
	if(view)
		system("echo \'plot \"output.txt\";replot \"regress.txt\" using 1:2; ;replot \"regress.txt\" using 1:3;replot \"regress.txt\" using 1:4; ;replot \"regress.txt\" using 1:5\' | gnuplot -persist");
    //system("echo \'plot \"output.txt\";replot \"regress.txt\" using 1:2; replot \"regress.txt\" using 1:5\' | gnuplot -persist");

  return 0;
}

