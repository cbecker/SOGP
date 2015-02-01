#include "SOGP_aux.h"
#include <string.h>

//Basic cascades - Not efficient
ReturnMatrix SOGPKernel::kernelM(const ColumnVector& in, const Matrix &BV){
  ColumnVector k(BV.Ncols());
  for(int i=1;i<=BV.Ncols();i++)
    k(i)=kernel(in,BV.Column(i));
  k.Release();
  return k;
}
double SOGPKernel::kstar(const ColumnVector& in){
  return kernel(in,in);
}
double SOGPKernel::kstar(){
  ColumnVector foo(1);
  foo(1)=0;
  return kernel(foo,foo);
}
//RBF
double RBFKernel::kernel(const ColumnVector &a, const ColumnVector &b){ 
 double d = a.Nrows();
  if(d!=widths.Ncols()){//Expand if necessary
    //printf("RBFKernel:  Resizing width to %d\n",(int)d);
    double wtmp=widths(1);
    widths.ReSize(d);
    for(int i=1;i<=d;i++)
      widths(i)=wtmp;
  }
  //I think this bumps up against numerical stability issues.
  return C + A*exp(- 0.5 * SumSquare(SP(a-b,widths.t())));
}
//POL
double POLKernel::kernel(const ColumnVector &a, const ColumnVector &b){
  double d = a.Nrows();
  double resp=1;
  double inner = (a.t()*b).AsScalar();
  for(int i=1;i<=scales.Ncols();i++)
    resp += pow((inner/(d*scales(i))),i);
  return resp;
}

//-------------------------------------------------------------
//Newmat printers
void printRV(RowVector rv,FILE *fp,const char *name,bool ascii){
  if(name)
    fprintf(fp,"%s ",name);
  fprintf(fp,"%d:",rv.Ncols());
  for(int i=0;i<rv.Ncols();i++)
    if(ascii)
      fprintf(fp,"%lf ",rv(i+1));
    else
      fwrite(&rv(i+1),sizeof(double),1,fp);
  fprintf(fp,"\n");
}
void readRV(RowVector &rv,FILE *fp,const char *name,bool ascii){
  if(name){
    char tn[128];//bad
    fscanf(fp,"%s ",tn);
    if(strcmp(tn,name))
      printf("readRV: Expected '%s', got '%s'\n",name,tn);
  }
  int len;
  fscanf(fp,"%d:",&len);
  rv.ReSize(len);
  for(int i=0;i<rv.Ncols();i++)
    if(ascii)
      fscanf(fp,"%lf ",&rv(i+1));
    else
      fread(&(rv(i+1)),sizeof(double),1,fp);
  fscanf(fp,"\n");
}
void printCV(ColumnVector cv,FILE *fp,const char *name,bool ascii){
  printRV(cv.t(),fp,name,ascii);
}
void readCV(ColumnVector &cv,FILE *fp,const char *name,bool ascii){
  RowVector rv;
  readRV(rv,fp,name,ascii);
  cv=rv.t();
}
void printMatrix(Matrix m,FILE *fp,const char *name,bool ascii){
  if(name)
    fprintf(fp,"%s ",name); 
  fprintf(fp,"(%d:%d)",m.Nrows(),m.Ncols());
  for(int i=0;i<m.Nrows();i++){
    for(int j=0;j<m.Ncols();j++){
      if(ascii)
	fprintf(fp,"%lf ",m(i+1,j+1));
      else
	fwrite(&(m(i+1,j+1)),sizeof(double),1,fp);
    }
    if(ascii)fprintf(fp,"\n");
  }
  if(ascii)fprintf(fp,"\n");
}
void readMatrix(Matrix &m,FILE *fp,const char *name,bool ascii){
  if(name){
    char tn[128];
    fscanf(fp,"%s ",tn);
    if(strcmp(tn,name))
      printf("readMatrix: Expected '%s', got '%s'\n",name,tn);
  }
  int wid,hgt;
  fscanf(fp,"(%d:%d)",&wid,&hgt);
  m.ReSize(wid,hgt);
  for(int i=0;i<m.Nrows();i++){
    for(int j=0;j<m.Ncols();j++){
      if(ascii)
	fscanf(fp,"%lf ",&m(i+1,j+1));
      else
	fread(&(m(i+1,j+1)),sizeof(double),1,fp);
    } 
    if(ascii)fscanf(fp,"\n");
  }
  if(ascii)fscanf(fp,"\n");
}
