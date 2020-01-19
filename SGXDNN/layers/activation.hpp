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

    template<typename T>
    class Activation : public Layer<T>
    {
    public:
        explicit Activation(
                const std::string &name,
                const array4d input_shape,
                const std::string &activation_type,
                const int bits_w,
                const int bits_x,
                bool verif_preproc
        ) : Layer<T>(name, input_shape),
            activation_type_(activation_type),
            bits_w_(bits_w),
            bits_x_(bits_x),
            verif_preproc_(verif_preproc)
        {
            shift_w = (1 << bits_w_);
            shift_x = (1 << bits_x_);
            inv_shift = 1.0 / shift_w;

            if (!(activation_type == "relu" or
                  activation_type == "softmax" or
                  activation_type == "linear" or
                  activation_type == "relu6" or
                  activation_type == "sigmoid"))
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

        std::string activation_type() const
        {
            return activation_type_;
        }

    protected:

        TensorMap<T, 4> apply_impl(TensorMap<T, 4> input, void *device_ptr = NULL, bool release_input = true) override
        {
            // std::cout<<"activation:"<<input.dimensions()<<std::endl;
#ifdef EIGEN_USE_THREADS
            Eigen::ThreadPoolDevice *d = static_cast<Eigen::ThreadPoolDevice *>(device_ptr);
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
            else if (activation_type_ == "sigmoid")
            {
                input = (static_cast<T>(1.0) + (-input).exp()).inverse();
            }
            else
            {
                assert(false);
                return input;
            }
        }

        T get_loss(TensorMap<T, 4> output, TensorMap<T, 4> label, std::string loss_func)
        {
            int batch = label.dimension(0);
            Tensor<T, 0> loss;
            if (loss_func == "MSE" || loss_func == "mse")
            {
                loss = (label - output).square().sum() / static_cast<T>(output_size_ * batch);
            }
            else if (loss_func == "CrossEntropy" || loss_func == "crossentropy")
            {
                loss = -(label * output.log()).sum() / static_cast<T>(batch);
            }
            return loss(0);
        }

        T get_accuracy(TensorMap<T, 4> output, TensorMap<T, 4> label)
        {
            int batch = label.dimension(0);
            int match_number = 0;
            assert(output_size_ == label.size() / batch);
            for (int i = 0; i < batch; ++i)
            {
                VectorMap<T> output_vec_map(output.data() + i * output_size_, output_size_);
                VectorMap<T> label_vec_map(label.data() + i * output_size_, output_size_);
                Eigen::Index out_max, label_max;
                output_vec_map.maxCoeff(&out_max);
                label_vec_map.maxCoeff(&label_max);
                match_number += (out_max == label_max);
            }
            return match_number / static_cast<T> (batch);
        }

        TensorMap<T, 4>
        last_back(TensorMap<T, 4> output, TensorMap<T, 4> labels, TensorMap<T, 4> der, std::string loss_func,
                  std::pair<T, T> &loss_acc)
        {
            new(&output)TensorMap<T, 4>(output.data(), der.dimensions());
            int batch = labels.dimension(0);
            MatrixMap<T> output_matrix_map(output.data(), batch, output_size_);
            MatrixMap<T> der_matrix_map(der.data(), batch, output_size_);
            MatrixMap<T> labels_matrix_map(labels.data(), batch, output_size_);

            loss_acc = std::make_pair(get_loss(output, labels, loss_func), get_accuracy(output, labels));

            if (loss_func == "MSE" || loss_func == "mse")
            {
                //均方误差损失函数
                der_matrix_map = output_matrix_map - labels_matrix_map;
                der_matrix_map = (2 * der_matrix_map) * (1.0f / output_size_);
            }
            else if (loss_func == "CrossEntropy" || loss_func == "crossentropy")
            {
                //交叉熵损失函数
                if (activation_type_ == "softmax")
                {
                    der_matrix_map = output_matrix_map - labels_matrix_map;
                    return der;
                }
                //交叉熵损失函数
                der_matrix_map = (-labels_matrix_map).cwiseProduct(output_matrix_map.cwiseInverse());
            }
            if (activation_type_ == "relu")
            {
                for (int i = 0; i < batch; ++i)
                    for (int j = 0; j < output_matrix_map.cols(); ++j)
                    {
                        der_matrix_map(i, j) *= output_matrix_map(i, j) > 0 ? 1 : 0;
                    }
            }
            else if (activation_type_ == "relu6")
            {
                for (int i = 0; i < batch; ++i)
                    for (int j = 0; j < output_matrix_map.cols(); ++j)
                    {
                        der_matrix_map(i, j) *= (output_matrix_map(i, j) == 0 || output_matrix_map(i, j) == 6) ? 0 : 1;
                    }
            }
            else if (activation_type_ == "softmax")
            {
                for (int i = 0; i < batch; ++i)
                {
                    VectorMap<T> output_vec_map(output.data() + output_size_ * i, output_size_);
                    VectorMap<T> der_vec_map(der.data() + output_size_ * i, output_size_);
                    der_vec_map = der_vec_map * (output_vec_map.transpose() * -output_vec_map) +
                                  der_vec_map.cwiseProduct(output_vec_map);
                }
            }
            else if (activation_type_ == "sigmoid")
            {
                der = der * (output * (static_cast<T>(1.0) - output));
            }
            return der;
        }

        TensorMap<T, 4> back_prop(TensorMap<T, 4> input, TensorMap<T, 4> der, float learn_rate) override
        {
            new(&der)TensorMap<T, 4>(der.data(), input.dimensions());
            if (activation_type_ == "relu")
            {
                VectorMap<T> input_vec_map(input.data(), input.size());
                VectorMap<T> der_vec_map(der.data(),der.size());
                for (int i = 0; i < input_vec_map.size(); ++i)
                {
                    der_vec_map(i) *= input_vec_map(i) > 0 ? 1 : 0;
                }
            }
            else if (activation_type_ == "relu6")
            {
                VectorMap<T> input_vec_map(input.data(), input.size());
                VectorMap<T> der_vec_map(der.data(),der.size());
                for (int i = 0; i < input_vec_map.size(); ++i)
                {
                    der_vec_map(i) *= (input_vec_map(i) == 0 || input_vec_map(i) == 6) ? 0 : 1;
                }
            }
            else if (activation_type_ == "softmax")
            {
                int batch;
                if (input.dimension(0) == 1 && input.dimension(1) == 1)
                {
                    batch = input.dimension(2);
                }
                else
                {
                    batch = input.dimension(0);
                }
                int image_size = input.size() / batch;
                for (int i = 0; i < batch; ++i)
                {
                    VectorMap<T> input_vec_map(input.data() + image_size * i, image_size);
                    VectorMap<T> result_vec_map(der.data() + image_size * i, image_size);
                    result_vec_map = result_vec_map * (input_vec_map.transpose() * -input_vec_map) +
                                     result_vec_map.cwiseProduct(input_vec_map);
                }
            }
            else if (activation_type_ == "sigmoid")
            {
                der = der * (input * (static_cast<T>(1.0) - input));
            }
            else if (activation_type_ == "linear");
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
