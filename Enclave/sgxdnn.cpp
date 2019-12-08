#define USE_EIGEN_TENSOR

#include "sgxdnn_main.hpp"

#include "tensor_types.h"

#include "Enclave.h"
#include "Enclave_t.h"

#include "Crypto.h"

using namespace tensorflow;

void ecall_load_model_float(char* model_json, float** filters)
{
	load_model_float(model_json, filters);
}

void ecall_predict_float(float* input, float* output, int batch_size)
{
	predict_float(input, output, batch_size);
}


void ecall_sgxdnn_benchmarks(int num_threads) {
	sgxdnn_benchmarks(num_threads);
}
