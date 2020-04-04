#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {

  if (n == 0) return;
  int number_of_threads = 8; 
  omp_set_num_threads(number_of_threads);
  int part_size; 
  if((n-1)%number_of_threads == 0)
    part_size = (n-1)/number_of_threads; 
  else 
    part_size = ((n-1)/number_of_threads) + 1; 


  long *offset = (long*) malloc(number_of_threads * sizeof(long)); 
  offset[0] = 0; 
  #pragma omp parallel
    {
      int id = omp_get_thread_num();
      long beg = id*part_size ; 
    

      for (long i = beg ; i < (beg+part_size) && i < n ; i++) {
        if(i==0)
          prefix_sum[i] = 0; 
        else if(i==beg)
          prefix_sum[i] = A[i-1];
        else 
          prefix_sum[i] = prefix_sum[i-1] + A[i-1];
        
      }

      if(id<number_of_threads-1)
      offset[id+1] = prefix_sum[beg+part_size-1]; 

    }


  for(int i = 1; i<number_of_threads;i++)
    offset[i] += offset[i-1]; 



  #pragma omp parallel
    {
      int id = omp_get_thread_num();
      long beg = id*part_size ; 
      for (long i = beg ; i < (beg+part_size) && i < n ; i++) {
          prefix_sum[i] +=  offset[id];
      }
 

    }

}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));

  for (long i = 0; i < N; i++) A[i] = rand();
  //for (long i = 0; i < N; i++) A[i] = i+1; 
  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
