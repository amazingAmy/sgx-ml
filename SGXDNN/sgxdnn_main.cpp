

#define USE_EIGEN_TENSOR

#ifndef USE_SGX
#define EIGEN_USE_THREADS
#else
#include "Enclave.h"
#include "sgx_trts.h"
#endif

#include "sgxdnn_main.hpp"
#include "layers/eigen_maxpool.h"
#include "randpool.hpp"
#include "utils.hpp"
#include "benchmark.hpp"

#include <unsupported/Eigen/CXX11/Tensor>
#include "model.hpp"
#include <iostream>
#include <memory>
#include <chrono>
#include <string>
#include <cstring>
#include <deque>
#include <stack>

#include "Crypto.h"

using namespace SGXDNN;

// prime P chosen for data blinding. Chosen such that P + P/2 < 2^24
int p_int = (1 << 23) + (1 << 21) + 7;
float p = (float) p_int;
float mid = (float) (p_int / 2);

// prime used for Freivalds checks. Largest prime smaller than 2^24
int p_verif = ((1 << 24) - 3);
double inv_p_verif = 1.0 / p_verif;

// some vectorized constants
__m256 p8f = _mm256_set1_ps(p);
__m256 mid8f = _mm256_set1_ps(mid);
__m256 negmid8f = _mm256_set1_ps(-mid);
__m256 zero8f = _mm256_set1_ps((float)(0));
__m256 inv_shift8f = _mm256_set1_ps((float)(1.0/256));
__m256 six8f = _mm256_set1_ps((float) 6 * 256 * 256);

// unblind data mod p, compute activation and write to output buffer
template <typename F>
inline void unblind(F func, float* inp, float* blind, float* out, int num_elements) {
	for(size_t i = 0; i < num_elements; i += 8) {
			const __m256 inp8f = _mm256_load_ps( &inp[i] );             // blinded input
			const __m256 blind8f = _mm256_load_ps( &blind[i] );   		// blinding factor
			const __m256 sub8f = _mm256_sub_ps(inp8f, blind8f);         // unblinded

			const __m256 if_geq = _mm256_cmp_ps(sub8f, mid8f, 0x0d);    // unblinded >= mid
			const __m256 if_lt = _mm256_cmp_ps(sub8f, negmid8f, 0x01);  // unblinded < -mid
			const __m256 then8f = _mm256_sub_ps(sub8f, p8f);            // unblinded - p
			const __m256 elif8f = _mm256_add_ps(sub8f, p8f);            // unblinded + p
			const __m256 res8f = _mm256_blendv_ps(
										_mm256_blendv_ps(
												sub8f,
												elif8f,
												if_lt),
										then8f,
										if_geq);

			_mm256_stream_ps(&out[i], func(res8f));
    }
}

extern "C" {

	Model<float> model_float;

	bool slalom_privacy;
	bool slalom_integrity;
	int batch_size;
	aes_stream_state producer_PRG;
	aes_stream_state consumer_PRG;
	std::deque<sgx_aes_gcm_128bit_iv_t*> aes_gcm_ivs;
	std::deque<sgx_aes_gcm_128bit_tag_t*> aes_gcm_macs;
	float* temp_buffer;
	float* temp_buffer2;
	Tensor<float, 1> buffer_t;
	Tensor<float, 1> buffer2_t;
	int act_idx;
	bool verbose;
	std::vector<int> activation_idxs;

	#ifdef EIGEN_USE_THREADS
	int n_threads = 1;
	Eigen::ThreadPool pool(n_threads);
	Eigen::ThreadPoolDevice device(&pool, n_threads);
	#endif

	// load a model into the enclave
	void load_model_float(char* model_json, float** filters) {
		model_float.load_model(model_json, filters, false, false);
		#ifdef EIGEN_USE_THREADS
		Eigen::setNbThreads(n_threads);
		#endif
	}

	// forward pass
	void predict_float(float* input, float* output, int batch_size) {
		//对于vgg16来说，这三个依次是batch_size、224、224、3
		array4d input_dims = {batch_size,
							  model_float.input_shape[0],
							  model_float.input_shape[1],
							  model_float.input_shape[2]};

		int input_size = batch_size * model_float.input_shape[0] * model_float.input_shape[1] * model_float.input_shape[2];
		assert(input_size != 0);

		// copy input into enclave
		float* inp_copy = model_float.mem_pool->alloc<float>(input_size);
		std::copy(input, input + input_size, inp_copy);

		auto map_in = TensorMap<float, 4>(inp_copy, input_dims);
		TensorMap<float, 4>* in_ptr = &map_in;

		sgx_time_t start_time;
        sgx_time_t end_time;
        double elapsed;

        start_time = get_time_force();

        // loop over all layers
		for (int i=0; i<model_float.layers.size(); i++) {
			if (TIMING) {
				printf("before layer %d (%s)\n", i, model_float.layers[i]->name_.c_str());
			}

			sgx_time_t layer_start = get_time();
			#ifdef EIGEN_USE_THREADS
			auto temp_output = model_float.layers[i]->apply(*in_ptr, (void*) &device);
			#else
			auto temp_output = model_float.layers[i]->apply(*in_ptr);
			#endif

			//将上一次的输出作为下一次的输入
			in_ptr = &temp_output;

			sgx_time_t layer_end = get_time();
			if (TIMING) {
				printf("layer %d required %4.4f sec\n", i, get_elapsed_time(layer_start, layer_end));
			}
		}

		// copy output outside enclave
		std::copy(((float*)in_ptr->data()), ((float*)in_ptr->data()) + ((int)in_ptr->size()), output);
		model_float.mem_pool->release(in_ptr->data());

		end_time = get_time_force();
        printf("total time: %4.4f sec\n", get_elapsed_time(start_time, end_time));
	}


	// back propagation
	void train(float*input,float*labels,int batch_size)
    {
	    std::stack<float*>input_stack;
        int output_size = model_float.layers.back()->output_size();

        array4d input_dims = {batch_size,
                              model_float.input_shape[0],
                              model_float.input_shape[1],
                              model_float.input_shape[2]};
        array4d output_dims = {batch_size,1,1,output_size};

        int input_size = batch_size * model_float.input_shape[0] * model_float.input_shape[1] * model_float.input_shape[2];
        assert(input_size != 0);

        // copy input into enclave
        float* inp_copy = model_float.mem_pool->alloc<float>(input_size);
        std::copy(input, input + input_size, inp_copy);
        float* label_copy = model_float.mem_pool->alloc<float>(output_size);
        std::copy(labels,labels+output_size,label_copy);

        auto map_in = TensorMap<float, 4>(inp_copy, input_dims);
        TensorMap<float, 4>* in_ptr = &map_in;

        auto map_labels = TensorMap<float,4>(label_copy,output_dims);
        TensorMap<float,4>* label_ptr = &map_labels;

        sgx_time_t start_time;
        sgx_time_t end_time;
        double elapsed;

        start_time = get_time_force();

        input_stack.push(in_ptr->data());
        // loop over all layers
        for (int i=0; i<model_float.layers.size(); i++) {
            if (TIMING) {
                printf("before layer %d (%s)\n", i, model_float.layers[i]->name_.c_str());
            }

            sgx_time_t layer_start = get_time();
#ifdef EIGEN_USE_THREADS
            auto temp_output = model_float.layers[i]->apply(*in_ptr, (void*) &device, false);
#else
            auto temp_output = model_float.layers[i]->apply(*in_ptr,NULL,false);
#endif

            //将上一次的输出作为下一次的输入
            in_ptr = &temp_output;
            input_stack.push(in_ptr->data());
            sgx_time_t layer_end = get_time();
            if (TIMING) {
                printf("layer %d required %4.4f sec\n", i, get_elapsed_time(layer_start, layer_end));
            }
        }

        // for(int i=0;i<9;++i)
        // {
        //     cout<<input_stack.top()<<" ";
        //     input_stack.pop();
        // }

        printf("forward pass end!\nstart back propagate(stack size = %d)...\n",input_stack.size());

        auto der = model_float.mem_pool->alloc<float>(output_size);
        TensorMap<float,4>der_map(der,1,1,1,output_size);
        TensorMap<float,4>*back_der = &der_map;
        std::string activation_name;
        //loop over layers to back propagate
        for(int i=0;i<model_float.layers.size();++i)
        {
            if(i==0)
            {
                printf("last layer start\n");
                auto last_layer = model_float.layers.back();
                assert(last_layer->name_=="activation");
                activation_name = dynamic_pointer_cast<Activation<float>>(last_layer)->activation_type();
                auto result = last_layer->last_back(*in_ptr,*label_ptr,*back_der,"mse");
                back_der = &result;
                input_stack.pop();
                printf("last layer end\n");
            }
            else
            {
                auto layer = model_float.layers[model_float.layers.size()-1-i];
                printf("%s layer start\n",layer->name_.c_str());
                if(layer->name_=="activation")
                {
                    activation_name = dynamic_pointer_cast<Activation<float>>(layer)->activation_type();
                    input_stack.pop();
                    printf("%s layer end\n",activation_name.c_str());
                    continue;
                }
                else
                {
                    auto output = input_stack.top();
                    input_stack.pop();
                    auto input = input_stack.top();
                    auto dense_layer = dynamic_pointer_cast<Dense<float>>(layer);
                    auto result = layer->back_prop(input,*back_der,activation_name,0.2);
                    back_der = &result;
                    printf("dense result size:%d\n",back_der->size());
                    model_float.mem_pool->release(output);
                }
                printf("%s layer end\n",layer->name_.c_str());
            }
        }

        end_time = get_time_force();
        printf("total time: %4.4f sec\n", get_elapsed_time(start_time, end_time));
    }
	

	void sgxdnn_benchmarks(int num_threads) {
		benchmark(num_threads);
	}
}
