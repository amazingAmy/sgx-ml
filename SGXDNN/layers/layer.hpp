#ifndef SGXDNN_LAYER_H_
#define SGXDNN_LAYER_H_

#define USE_EIGEN_TENSOR

#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>

#include "tensor_types.h"
#include "shape.h"
#include "../utils.hpp"

using namespace tensorflow;

namespace SGXDNN
{
    template <typename T>
    class Layer {

    public:
        explicit Layer(const std::string& name,
        			   const array4d input_shape)
            : name_(name),
			  input_shape_(input_shape)
        {
        }

        virtual ~Layer()
        {
        }

        TensorMap<T, 4> apply(TensorMap<T, 4> input_map, void* device_ptr = NULL, bool release_input = true)  {
            auto result = apply_impl(input_map, device_ptr, release_input);
            return result;
        }
        virtual TensorMap<T,4> back_prop(TensorMap<T,4>input,TensorMap<T,4>der,float learn_rate){}
        virtual TensorMap<T,4> last_back(TensorMap<T,4>output,TensorMap<T,4>labels,TensorMap<T,4>der,std::string loss_func,std::pair<T,T>&loss_acc){}

        virtual array4d output_shape() = 0;
        virtual int output_size() = 0;

		virtual int num_linear() {
			return 0;
		}

        std::string name_;
        const array4d input_shape_;

    protected:
        virtual TensorMap<T, 4> apply_impl(TensorMap<T, 4> input_map, void* device_ptr = NULL, bool release_input = true) = 0;

    };

}
#endif
