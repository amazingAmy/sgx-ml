#ifndef SGXDNN_ACTIVATION_H_
#define SGXDNN_ACTIVATION_H_


#include <cassert>
#include <iostream>
#include <string>
#include "layer.hpp"

#ifdef USE_SGX
#include "Enclave.h"
#endif

using namespace tensorflow;

namespace SGXDNN
{

	template <typename T> class Activation : public Layer<T>
	{
	public:
		explicit Activation(
				const std::string& name,
				const array4d input_shape,
				const std::string& activation_type,
				const int bits_w,
				const int bits_x,
				bool verif_preproc
				): Layer<T>(name, input_shape),
				activation_type_(activation_type),
				bits_w_(bits_w),
				bits_x_(bits_x),
				verif_preproc_(verif_preproc)
		{
			shift_w = (1 << bits_w_);
			shift_x = (1 << bits_x_);
			inv_shift = 1.0/shift_w;

			if (!(activation_type == "relu" or
				  activation_type == "softmax" or
				  activation_type == "linear" or
				  activation_type == "relu6"))
			{
				printf("unknown activation %s\n", activation_type_.c_str());
				assert(false);
			}
			printf("loading activation %s\n", activation_type_.c_str());

			output_shape_ = input_shape;
			if (input_shape[0] == 0)
			{
				output_size_ = input_shape[1] * input_shape[2] * input_shape[3];
			}
			else
			{
				assert(input_shape[2] == 0);
				output_size_ = input_shape[3];
			}
		}

		array4d output_shape() override
		{
			return output_shape_;
		}

		int output_size() override
		{
			return output_size_;
		}

		std::string activation_type()const
		{
			return activation_type_;
		}

	protected:

		TensorMap<T, 4> apply_impl(TensorMap<T, 4> input, void* device_ptr = NULL, bool release_input = true) override
		{
		    std::cout<<"activation:"<<input.dimensions()<<std::endl;
			#ifdef EIGEN_USE_THREADS
			Eigen::ThreadPoolDevice* d = static_cast<Eigen::ThreadPoolDevice*>(device_ptr);
			#endif

			if (activation_type_ == "relu")
			{
				input = input.cwiseMax(static_cast<T>(0));
				return input;
			}
			else if (activation_type_ == "relu6")
            {
                input = input.cwiseMax(static_cast<T>(0)).cwiseMin(static_cast<T>(6));
                return input;
            }
			else if (activation_type_ == "softmax")
			{
				const int batch = input.dimension(2);
				const int num_classes = input.dimension(3);
				array4d dims4d = {1, 1, batch, 1};
				array4d bcast = {1, 1, 1, num_classes};
				array1d depth_dim = {3};

				input = ((input - input.maximum(depth_dim).eval().reshape(dims4d).broadcast(bcast))).exp();
                input = input / (input.sum(depth_dim).eval().reshape(dims4d).broadcast(bcast));
				return input;
			}
			else if (activation_type_ == "linear")
			{
				return input;
			}
			else
			{
				assert(false);
				return input;
			}
		}

		TensorMap<T,4> last_back(TensorMap<T,4>output,TensorMap<T,4>labels,TensorMap<T,4>der,std::string error_func)
        {
		    VectorMap<T>output_matrix_map(output.data(),output_size_);
            VectorMap<T>der_matrix_map(der.data(),output_size_);
            VectorMap<T>labels_matrix_map(labels.data(),output_size_);

		    if(error_func=="MSE"||error_func=="mse")
		    {
		    	//均方误差损失函数
		        der_matrix_map = output_matrix_map -labels_matrix_map;
		        der_matrix_map = der_matrix_map / output_size_;
            }
		    else if(error_func=="CrossEntropy"||error_func=="crossentropy")
			{
		    	if(activation_type_=="softmax")
				{
		    		der_matrix_map = output_matrix_map.array() - 1;
		    		return der;
				}
		    	//交叉熵损失函数
		    	der_matrix_map = (-labels_matrix_map).cwiseProduct(output_matrix_map.cwiseInverse());

			}
            VectorMap<T>der_matrix(der.data(),output_size_);
            if(activation_type_=="relu")
            {
                for(int i=0;i<output_matrix_map.size();++i)
                {
                    der_matrix_map(i) *= output_matrix_map(i) > 0 ? 1 : 0;
                }
            }
            else if(activation_type_=="relu6")
            {
                for(int i=0;i<output_matrix_map.size();++i)
                {
					 der_matrix_map(i) *= (output_matrix_map(i) == 0 || output_matrix_map(i) == 6) ? 0 : 1;
                }
            }
            else if(activation_type_=="softmax")
            {
				der_matrix.cwiseProduct(((output_matrix_map.transpose()*(-output)).rowwise().sum()).transpose()+output_matrix_map);
			}
			return der;
        }


		const std::string activation_type_;
		const int bits_w_;
		const int bits_x_;
		bool verif_preproc_;
		array4d output_shape_;
		int output_size_;

		int shift_w;
		int shift_x;
		T inv_shift;
	};
}
#endif
