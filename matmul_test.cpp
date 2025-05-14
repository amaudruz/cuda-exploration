#include <random>
#include <chrono>
#include <ratio>

using namespace std;
using namespace std::chrono;

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

float * matmul(float* m1, float*m2, int m, int l, int n, bool verbose = false) {
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

int main() {
  int m = 1000;
  int l = 1000;
  int n = 1000;


  //printf("M1:\n");
  float* m1 = create_matrix(m, l);
  //print_matrix(m1, m, l);
  //printf("");

  //printf("M2:\n");
  float* m2 = create_matrix(l, n);
  //print_matrix(m2, l, n);
  //printf("");

  //printf("Matrix multiplication result:\n");

  auto start = high_resolution_clock::now();
  float* result = matmul(m1, m2, m, l, n);
  auto end = std::chrono::high_resolution_clock::now();

  auto du = duration<double, std::milli>(end - start);
  //print_matrix(result, m, n);
  float flops = (2*m*n*l) / (du.count() / 1000);
  printf("MFLOPs %.f", flops / 1000000);

  return 0;
}
