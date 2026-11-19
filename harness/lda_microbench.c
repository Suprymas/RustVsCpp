// Microbenchmark: the same tiled kernel with a variable row stride (lda).
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define B 64
static void band(const double *restrict a,const double *restrict b,
                 double *restrict c,size_t n,size_t lda){
  for(size_t ii=0;ii<n;ii+=B){size_t im=ii+B<n?ii+B:n;
   for(size_t kk=0;kk<n;kk+=B){size_t km=kk+B<n?kk+B:n;
    for(size_t jj=0;jj<n;jj+=B){size_t jm=jj+B<n?jj+B:n,w=jm-jj;
     for(size_t i=ii;i<im;i++){const double*ar=a+i*lda;double*cr=c+i*lda+jj;
      for(size_t k=kk;k<km;k++){const double aik=ar[k];const double*br=b+k*lda+jj;
       for(size_t j=0;j<w;j++) cr[j]+=aik*br[j];}}}}}
}
int main(int argc,char**argv){
  size_t n=2048, lda=(size_t)atol(argv[1]);
  double*a=aligned_alloc(4096,n*lda*8),*b=aligned_alloc(4096,n*lda*8),*c=aligned_alloc(4096,n*lda*8);
  for(size_t i=0;i<n*lda;i++){a[i]=i*1e-9;b[i]=i*2e-9;c[i]=0;}
  struct timespec t0,t1; double best=1e9;
  for(int r=0;r<3;r++){clock_gettime(CLOCK_MONOTONIC,&t0);band(a,b,c,n,lda);clock_gettime(CLOCK_MONOTONIC,&t1);
    double d=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9; if(d<best)best=d;}
  printf("%.4f\n",best); return c[0]==12345.0;
}
