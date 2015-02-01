/*
A little testing program to see if SOGP works properly.  
Dan Grollman
2008

TODO:
close plot automatically?  or reuse?
*/

#include "SOGP.h"
#include <time.h>
#include <sys/time.h>
#include <string.h>

//Regression mse threshold
#define mseTHRESH 1e-3
//Numerical stability threshold
#define numTHRESH 1e-6
//How many points to run
#define N 100

int main(int argc, char **argv){
  int seed = time(NULL);
  bool valgrind = true;
	bool view = true;
	bool captest=true;

  for(int argi=1;argi<argc;argi++){
    if(!strcmp(argv[argi],"-s"))
      seed = atoi(argv[++argi]);
		if(!strcmp(argv[argi],"-v"))
      valgrind = false;
    if(!strcmp(argv[argi],"-z"))
      view = false;
    if(!strcmp(argv[argi],"-c"))
      captest = false;
  }

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
  RowVector inputs(N),outputs(N), testin(N),testout(N);

  //Generate data (train and test)
  printf("Generating data y = cos(5x)\n");
  file=fopen("output.txt","w"); 
  for(int n=1;n<=N;n++){
    inputs(n)= -1 + rand()/(RAND_MAX/2.0);
    outputs(n)=cos(5*inputs(n));//sinc?
    testin(n) = -1 + rand()/(RAND_MAX/2.0);
    testout(n) = cos(5*testin(n));
    fprintf(file,"%lf %lf\n",inputs(n),outputs(n));
  }
  fclose(file);
 
  //Test Regression
  printf("Regression test (default settings)\n");
  m_SOGP = new SOGP();
  m_SOGP->addM(inputs,outputs);
  file=fopen("regress.txt","w");
  mus = m_SOGP->predictM(testin);
  mse=0;
  for(int n=1;n<=N;n++){
    fprintf(file,"%lf %lf\n",testin(n),mus(1,n));
    double err = testout(n)-mus(1,n);
    mse+=err*err;
  }
  mse/=N;
  printf("MSE = %.10lf \t\t\t\t\t %s\n",mse,mse<mseTHRESH?"PASS":"FAIL");
  fclose(file);
  //Save the base answer
  mse_base = mse;
  
  //Display for the user
	if(view)
		system("echo \'plot \"output.txt\";replot \"regress.txt\"\' | gnuplot -persist");

  //Test FILE IO
  printf("Testing File IO.");
  m_SOGP->save("savedSOGP.txt");
  delete m_SOGP;
  m_SOGP = new SOGP;
  m_SOGP->load("savedSOGP.txt");
  m_SOGP->save("resavedSOGP.txt");
  printf("\nTesting load/save.  Differences between stars\n******\n");
  system("diff -a savedSOGP.txt resavedSOGP.txt");//Do this better?
  printf("******\n");

  //Test that reloaded is correct
  printf("Testing that loaded works\n");
  mse = 0;
  mus = m_SOGP->predictM(testin);
  mse = SumSquare(testout-mus)/N;
  printf("MSE = %.10lf \t\t\t\t\t %s\n",mse,mse-mse_base<numTHRESH?"PASS":"FAIL");
  delete m_SOGP;
  
  //Test that Capacity works
  printf("Testing Capacity Limits\n");
  for(int cap=10;cap<=N;cap+=10){
    printf("\tTesting Capacity %3d.....",cap);
    m_SOGP = new SOGP(.001,.001,cap);//Small width and noise so lots of BV
    m_SOGP->addM(inputs,outputs);
    printf("%d \t\t\t %s\n",m_SOGP->size(),m_SOGP->size()==cap?"PASS":"FAIL");
    delete m_SOGP;  
  }
  
  //Test multidimensionality
  printf("Testing multidimensionality\n");
  for(int din=1;din<=2;din++){
    for(int dout=1;dout<=2;dout++){
      printf("%d => %d: ",din,dout);
      //Make data
      Matrix in(din,N), out(dout,N);
      for(int n=1;n<=N;n++){
				for(int d=1;d<=din;d++)
					in(d,n)=inputs(n);
				for(int d=1;d<=dout;d++)
					out(d,n)=outputs(n);
      }
      m_SOGP=new SOGP();
      m_SOGP->addM(in,out);
      mus = m_SOGP->predictM(in);
      mse = SumSquare(out-mus)/(dout*N);
      printf("MSE = %.10lf \t\t\t\t %s\n",mse,mse<mseTHRESH?"PASS":"FAIL");
      delete m_SOGP;
    }
  }
  
  //Test that widths go right way
  printf("Testing that as widths go up, number of BVs go down.");
  int BVsize=100;
  for(double wid=.1;wid<=2;wid+=.1){
    m_SOGP= new SOGP(wid,.1);
    m_SOGP->addM(inputs,outputs);
    int size = m_SOGP->size();
    if(size>BVsize+3){//Minor changes are OK
      printf("BV SIZE WENT UP! %d -> %d\n",BVsize,size);
      BVsize = -1;
    }
    else
      BVsize=size;
    delete m_SOGP;
  }
  printf("%3d  %s\n",BVsize,BVsize>0?"PASS":"FAIL");
  
  printf("Testing prediction from null GP\n");
  m_SOGP=new SOGP(1,.1,.1);
  mus = m_SOGP->predict(inputs.Column(1));
  printf("Returns Col with %d rows \t\t\t\t %s\n",mus.Nrows(),mus.Nrows()==0?"PASS":"FAIL");
  delete m_SOGP;

	
	printf("Testing logprob of zeros from empty GP\n");
	m_SOGP = new SOGP(.1,.1);
	ColumnVector in(2),out(2);
	in(1)=0;in(2)=0;out(1)=0;out(2)=0;
	printf("lp is %lf\n",m_SOGP->log_prob(in,out));
	m_SOGP->add(in,out);
	printf("lp is %lf\n",m_SOGP->log_prob(in,out));
	
	//Make a prediction for size:
	if(captest){
		printf("Predicting realtime (33ms) capacity\n");
		m_SOGP = new SOGP(.001,.001);
		int cap=1;
		int state=0;
		bool stop=false;
		double ms=0;
		timeval begin,end;
		//ColumnVector in(1), out(1);
		while(!stop){
			m_SOGP->change_capacity(cap);
			printf("%d ",cap);
			fflush(stdout);
			
			//Fill to capacity
			while(m_SOGP->size()!=cap){
				in(1)= -1 + rand()/(RAND_MAX/2.0);
				out(1)= -1 + rand()/(RAND_MAX/2.0);
				m_SOGP->add(in,out);
			}
			
			//Thrash 30 times, get average
			for(int i=0;i<30;i++){
				in(1)= -1 + rand()/(RAND_MAX/2.0);
				out(1)= -1 + rand()/(RAND_MAX/2.0);
				gettimeofday(&begin,NULL);
				m_SOGP->add(in,out);
				gettimeofday(&end,NULL);
				int diff = (end.tv_sec-begin.tv_sec)*1000000 + end.tv_usec - begin.tv_usec;
				ms += (diff/1000.0);
			}
			ms/=30.0;
			
			switch(state){
			case 0: if(ms<33){cap+=100;break;}else state=1; //Going up
			case 1: if(ms>33){cap-=10;break;}else state=2; //Going down
			case 2: if(ms<33){cap+=1;break;}else{stop = true;cap-=5;state=1;}
			}
		}
		printf("(%lf)\n",ms);
		delete m_SOGP;
	}

  //Test for memory leaks
  if(valgrind){//Doesn't run it?
    printf("Testing for memory leaks with valgrind, errors between stars\n");
    printf("*****\n");
    system("valgrind -q test -v");
    printf("*****\n");
  }

  //Clean
  system("rm -f *.txt");
}

