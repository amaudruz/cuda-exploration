#include <cstdio>
#include <random>
#include <chrono>
#include <ratio>
#include <omp.h>


using namespace std;
using namespace std::chrono;

void print_matrix(float* matrix, int m, int n, string mode="normal") {
  if (mode == "python"){
    printf("[");
  }
  for (int i = 0; i < m; i++) {
    if (mode == "python"){
      printf("[");
    }
    for (int j = 0; j < n; j++) {
      printf("%.8f ", matrix[(i*n)+j]);
      if (mode == "python"){
        printf(",");
      }
    }     
    if (mode == "python"){
      printf("],");
    }
    printf("\n");
  }
  if (mode == "python"){
    printf("]");
  }
    printf("\n");
}

float* create_matrix(int m, int n, string init = "uniform") {
  float* pointer = (float*) malloc(m*n*sizeof(float));


  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<float> dist(-1, 1);
  for (int i = 0; i < m*n; i++) {
      if (init == "uniform") {
        *(pointer + i) = dist(gen);
      } else if (init == "zero") {
        *(pointer + i) = 0.0;
      } else {
        exit(1);
      }
  }
  return pointer;
}

float* transpose_matrix(float* matrix, int m, int n) {
  float* transposed = create_matrix(n, m,"zero");
  for (int i = 0; i < m; i++) {
       for (int j = 0; j < n; j++){
          *(transposed + (j * m + i)) = *(matrix + (i * n + j));
        }
  }
  return transposed;

}

float * basic_matmul(float* m1, float*m2, int m, int l, int n) {
  float* result = create_matrix(m, n, "zero");
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < l; k++) {
        *(result + (i * n + j)) +=  *(m1 + (i * l + k)) * *(m2 + (k * n + j));
      }
    }
  } 
  return result;
}

float * cache_aware_matmul(float* m1, float*m2, int m, int l, int n) {
  float* result = create_matrix(m, n, "zero");
  for (int i = 0; i < m; i++) {
    for (int k = 0; k < l; k++) {
      for (int j = 0; j < n; j++) {
        *(result + (i * n + j)) +=  *(m1 + (i * l + k)) * *(m2 + (k * n + j));
      }
    }
  } 
  return result;
}

float * parallel_matmul(float* m1, float*m2, int m, int l, int n) {
  float* result = create_matrix(m, n, "zero");
  
  #pragma omp parallel for
  for (int i = 0; i < m; i++) {
    for (int k = 0; k < l; k++) {
      for (int j = 0; j < n; j++) {
        *(result + (i * n + j)) +=  *(m1 + (i * l + k)) * *(m2 + (k * n + j));
      }
    }
  } 
  return result;
}

int main() {
  int m = 4096;
  int l = 4096;
  int n = 4096;


  float* m1 = create_matrix(m, l);
  float* m2 = create_matrix(l, n);

  auto start = high_resolution_clock::now();
  float* result = parallel_matmul(m1, m2, m, l, n);
  auto end = std::chrono::high_resolution_clock::now();

  auto du = duration<double, std::milli>(end - start);

  float count_seconds = (du.count() / 1000.0);
  float flops = (2.0*m*n*l) / (float)count_seconds;
  printf("GFLOPs %.f\n", flops / 1000000000);
  printf("Duration %.4f\n", count_seconds);

  return 0;
}
