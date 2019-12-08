#ifndef SGXDNN_DEPTHCONV2D_H_
#define SGXDNN_DEPTHCONV2D_H_

#define EIGEN_USE_TENSOR

#include <stdio.h>
#include <iostream>
#include <string>
#include <type_traits>
#include <assert.h>

#include "../mempool.hpp"
#include "layer.hpp"
#include <cmath>

#ifndef USE_SGX
#include <chrono>
#else
#include "Enclave.h"
#include "sgx_tcrypto.h"
#include "Crypto.h"
#endif

using namespace tensorflow;

namespace SGXDNN
{
	struct DepthwiseArgs {
		  // Input layer dimensions
		  int batch;
		  int in_rows;
		  int in_cols;
		  int in_depth;
		  int filter_rows;
		  int filter_cols;
		  int depth_multiplier;
		  int stride;
		  int pad_rows;
		  int pad_cols;

		  // Output layer dimensions
		  int out_rows;
		  int out_cols;
		  int out_depth;

		  DepthwiseArgs()
			  : batch(0),
				in_rows(0),
				in_cols(0),
				in_depth(0),
				filter_rows(0),
				filter_cols(0),
				depth_multiplier(0),
				stride(0),
				pad_rows(0),
				pad_cols(0),
				out_rows(0),
				out_cols(0),
				out_depth(0) {}
	};

	template <typename T>
	void depthwise_conv(const DepthwiseArgs& args,
    	                const T* input, const T* depthwise_filter, T* output);

	template <typename T>
	class DepthwiseConv2D : public Layer<T>
	{
	public:
		DepthwiseConv2D(const std::string& name,
			   const array4d input_shape,
               const array4d& kernel_shape,
               const int row_stride,
               const int col_stride,
               const Eigen::PaddingType& padding,
               T* r_left, T* kernel, T* bias, 
			   MemPool* mem_pool,
			   bool is_verif_mode,
			   bool verif_preproc
			   ): Layer<T>(name, input_shape),
               row_stride_(row_stride),
               col_stride_(col_stride),
               padding_(padding),
               kernel_data_(nullptr),
               bias_data_(nullptr),
               bias_(NULL, kernel_shape[2]),
			   mem_pool_(mem_pool),
			   verif_preproc_(verif_preproc)
		{
	
			const int input_rows = input_shape[1];
			const int input_cols = input_shape[2];

			const int filter_rows = kernel_shape[0];
			const int filter_cols = kernel_shape[1];
			const int ch_in = kernel_shape[2];

			printf("Depthwise conv:\n");
			printf("in: %d x %d, strides: %d,%d\n", input_rows, input_cols, row_stride_, col_stride_);

			int out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
			GetWindowedOutputSize(input_rows, filter_rows, row_stride_,
								  padding_, &out_rows, &pad_rows);
			GetWindowedOutputSize(input_cols, filter_cols, col_stride_,
								  padding_, &out_cols, &pad_cols);

			printf("out: %d x %d, padding: %d\n", out_rows, out_cols, padding_);

			output_shape_ = {0, out_rows, out_cols, ch_in};
			output_size_ = out_rows * out_cols * ch_in;
			input_shape_ = {0, input_rows, input_cols, ch_in};
			input_size_ = input_rows * input_cols * ch_in;

			assert(row_stride == col_stride);

			// copy kernel and bias
			long kernel_size = kernel_shape[0] * kernel_shape[1] * kernel_shape[2];
			kernel_data_ = mem_pool_->alloc<T>(kernel_size);
			std::copy(kernel, kernel + kernel_size, kernel_data_);

			long bias_size = ch_in;
			bias_data_ = mem_pool_->alloc<T>(bias_size);
			std::copy(bias, bias + bias_size, bias_data_);
			new (&bias_) typename TTypes<T>::ConstVec(bias_data_, ch_in);

			args.in_rows = input_rows;
			args.in_cols = input_cols;
			args.in_depth = ch_in;
			args.filter_rows = filter_rows;
			args.filter_cols = filter_cols;
			args.depth_multiplier = 1;
			args.stride = row_stride;
			args.pad_rows = pad_rows;
			args.pad_cols = pad_cols;
			args.out_rows = out_rows;
			args.out_cols = out_cols;
			args.out_depth = ch_in;
		}

		array4d output_shape() override
		{
			return output_shape_;
		}

		int output_size() override
		{
			return output_size_;
		}

		int num_linear() override
		{
			return 1;
		}

	protected:

		TensorMap<T, 4> apply_impl(TensorMap<T, 4> input, void* device_ptr = NULL, bool release_input = true) override
		{
			#ifndef USE_SGX
			Eigen::ThreadPoolDevice* d = static_cast<Eigen::ThreadPoolDevice*>(device_ptr);
			#endif
  
			sgx_time_t start = get_time();
			const int batch = input.dimension(0);
			output_shape_[0] = batch;

			T* output_mem_ = mem_pool_->alloc<T>(batch * output_size_);
			auto output_map = TensorMap<T, 4>(output_mem_, output_shape_);

			args.batch = batch;
			depthwise_conv<T>(args, input.data(), kernel_data_, output_mem_);

			const int bias_size = bias_.dimension(0);
            const int rest_size = output_map.size() / bias_size;
            Eigen::DSizes<int, 1> one_d(output_map.size());
            Eigen::DSizes<int, 1> bcast(rest_size);

            output_map.reshape(one_d) = output_map.reshape(one_d) + bias_.broadcast(bcast).reshape(one_d);
			mem_pool_->release(input.data());

			sgx_time_t end = get_time();
            double elapsed = get_elapsed_time(start, end);
            if (TIMING) {
                printf("depthwise convd (%ld x %ld x %ld,  s%d) took %.4f seconds\n",
                		input.dimension(1), input.dimension(2), input.dimension(3), row_stride_, elapsed);
            }
			
			return output_map;
		}

		T* kernel_data_;
		T* bias_data_;
		TensorMap<T, 1> bias_;

		const Eigen::PaddingType padding_;
		const int row_stride_;
		const int col_stride_;

		MemPool* mem_pool_;

		array4d input_shape_;
		int input_size_;

		array4d output_shape_;
		int output_size_;

		DepthwiseArgs args;

		bool verif_preproc_;

	};

template <typename T>
struct DepthwiseInputCopyOp {
  static void Run(const DepthwiseArgs& args,
                  const int64 padded_filter_inner_dim_size, const int64 out_r,
                  const int64 out_c, const T* input, T* input_buffer) {

    typedef typename Eigen::internal::packet_traits<T>::type Packet;
    static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

    // Calculate vectorized and scalar (residual) lengths for 'in_depth'.
    const int64 input_vectorized_size =
        (args.in_depth / kPacketSize) * kPacketSize;
    const int64 input_scalar_size = args.in_depth % kPacketSize;

    // Calculate vectorized and scalar (residual) lengths for
    // 'depth_multiplier'. This is used to efficiently replicate data for
    // when 'depth_multiplier' > kPacketSize.
    const int64 dm_vectorized_size =
        (args.depth_multiplier / kPacketSize) * kPacketSize;
    const int64 dm_scalar_size = args.depth_multiplier % kPacketSize;

    // Calculate output padding length.
    const int64 output_scalar_size = args.out_depth % kPacketSize;
    const int64 output_pad_size =
        output_scalar_size > 0 ? kPacketSize - output_scalar_size : 0;

    const int64 replicated_packet_size = kPacketSize * args.depth_multiplier;

    // Iterate through all rows x cols reading 'in_depth' from 'input' and
    // replicating by 'depth_multiplier' into 'input_buffer' (otherwise
    // zero-padding input buffer as needed).
    auto* in_buf = input_buffer;
    const int64 in_r_start = out_r * args.stride - args.pad_rows;
    const int64 in_c_start = out_c * args.stride - args.pad_cols;
	
	for (int64 f_r = 0; f_r < args.filter_rows; ++f_r) {
      const int64 in_r = in_r_start + f_r;

      for (int64 f_c = 0; f_c < args.filter_cols; ++f_c) {
        const int64 in_c = in_c_start + f_c;

        if (in_r >= 0 && in_r < args.in_rows && in_c >= 0 &&
            in_c < args.in_cols) {
          auto* in = input + (in_r * args.in_cols + in_c) * args.in_depth;
          // Copy vectorized portion of inner dimension.
          for (int64 d = 0; d < input_vectorized_size; d += kPacketSize) {
            auto v = Eigen::internal::ploadu<Packet>(in + d);
            for (int dm = 0; dm < args.depth_multiplier; ++dm) {
              Eigen::internal::pscatter<T, Packet>(in_buf + dm, v,
                                                   args.depth_multiplier);
            }
            in_buf += replicated_packet_size;
          }

          // Copy scalar portion of inner dimension.
          for (int64 d = 0; d < input_scalar_size; ++d) {
            T v = in[input_vectorized_size + d];
            const int64 base = d * args.depth_multiplier;
            if (dm_vectorized_size > 0) {
              // Copy vectorized portion of replicated output.
              // This branch is only taken if 'args.depth_multiplier' is
              // vectorizable (i.e. args.depth_multiplier >= register width).
              auto p = Eigen::internal::pset1<Packet>(v);
              for (int64 dm = 0; dm < dm_vectorized_size; dm += kPacketSize) {
                Eigen::internal::pstoreu<T>(in_buf + base + dm, p);
              }
              // Copy scalar portion of replicated output.
              for (int64 dm = 0; dm < dm_scalar_size; ++dm) {
                in_buf[base + dm_vectorized_size + dm] = v;
              }
            } else {
              // Depth multiplier is less than one packet: scalar copy.
              for (int dm = 0; dm < args.depth_multiplier; ++dm) {
                in_buf[base + dm] = v;
              }
            }
          }
          in_buf += input_scalar_size * args.depth_multiplier;
		  
		  // Pad the remainder of the output to vector register boundary.
          for (int64 d = 0; d < output_pad_size; ++d) {
            in_buf[d] = static_cast<T>(0);
          }
          in_buf += output_pad_size;

        } else {
          // Zero pad.
          memset(in_buf, 0, sizeof(T) * padded_filter_inner_dim_size);
          in_buf += padded_filter_inner_dim_size;
        }
      }
    }
  }
};

template <typename T>
struct DepthwiseConv2DKernel {
  static void Run(const DepthwiseArgs& args,
                  const int64 padded_filter_inner_dim_size, const int64 out_r,
                  const int64 out_c, const T* filter, const T* input_buffer,
                  T* output) {

    typedef typename Eigen::internal::packet_traits<T>::type Packet;
    static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

    const int64 out_depth = args.out_depth;
    const int64 filter_spatial_size = args.filter_rows * args.filter_cols;
    const int64 output_scalar_size = out_depth % kPacketSize;
    const int64 output_vectorized_size =
        (out_depth / kPacketSize) * kPacketSize;
    const int64 base_output_index = (out_r * args.out_cols + out_c) * out_depth;

    for (int i = 0; i < output_vectorized_size; i += kPacketSize) {
      // Reset accumulator.
      auto vaccum = Eigen::internal::pset1<Packet>(static_cast<T>(0));
      for (int j = 0; j < filter_spatial_size; ++j) {
        // Calculate index.
        const int64 index = i + j * padded_filter_inner_dim_size;
		// Load filter.
        // TODO(andydavis) Unroll 'out_c' loop in caller so we can load
        // multiple inputs here to amortize the cost of each filter block load.
        const auto filter_block =
            Eigen::internal::ploadu<Packet>(filter + index);
        // Load input.
        const auto data_block =
            Eigen::internal::ploadu<Packet>(input_buffer + index);
        // Vector multiply-add.
        vaccum =
            Eigen::internal::pmadd<Packet>(filter_block, data_block, vaccum);
      }
      // Store vector accumulator to output.
      Eigen::internal::pstoreu<T>(output + base_output_index + i, vaccum);
    }

    if (output_scalar_size > 0) {
      auto vaccum = Eigen::internal::pset1<Packet>(static_cast<T>(0));
      for (int j = 0; j < filter_spatial_size; ++j) {
        const int64 index =
            output_vectorized_size + j * padded_filter_inner_dim_size;
        const auto filter_block =
            Eigen::internal::ploadu<Packet>(filter + index);
        const auto data_block =
            Eigen::internal::ploadu<Packet>(input_buffer + index);
        vaccum =
            Eigen::internal::pmadd<Packet>(filter_block, data_block, vaccum);
      }
      // Load accumulator into an array and loop through output.
      T out_buf[kPacketSize];
      Eigen::internal::pstoreu<T>(out_buf, vaccum);
      const int64 last_output_index =
          base_output_index + output_vectorized_size;
      for (int j = 0; j < output_scalar_size; ++j) {
        output[last_output_index + j] = out_buf[j];
      }
    }
  }
};

template <typename T>
void depthwise_conv(const DepthwiseArgs& args,
                    const T* input, const T* depthwise_filter, T* output) {

    typedef typename Eigen::internal::packet_traits<T>::type Packet;
    static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

    // Pad 'depthwise_filter' to vector register width (if needed).
    const bool pad_filter = (args.out_depth % kPacketSize) == 0 ? false : true;
    assert(pad_filter == false);

    const T* filter_data = depthwise_filter;

    // Computes one shard of depthwise conv2d output.
    auto shard = [&args, &input, &filter_data, &output](
                     int64 start, int64 limit) {
      static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));
      const int64 input_image_size =
          args.in_rows * args.in_cols * args.in_depth;
      const int64 output_image_size =
          args.out_rows * args.out_cols * args.out_depth;
      const int64 filter_spatial_size = args.filter_rows * args.filter_cols;
      const int64 padded_filter_inner_dim_size =
          ((args.out_depth + kPacketSize - 1) / kPacketSize) * kPacketSize;
	  
	  // Allocate buffer for local input regions.
      Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex> input_buffer(filter_spatial_size, padded_filter_inner_dim_size);
      T* input_buffer_data = input_buffer.data();

      for (int64 i = start; i < limit; ++i) {
        const int64 b = i / args.out_rows;
        const int64 in_base = b * input_image_size;
        const int64 out_base = b * output_image_size;

        const int64 out_r = i % args.out_rows;

        for (int64 out_c = 0; out_c < args.out_cols; ++out_c) {
          // Populate 'input_buffer_data' with data from local input region.
          DepthwiseInputCopyOp<T>::Run(args, padded_filter_inner_dim_size,
                                             out_r, out_c, input + in_base,
                                             input_buffer_data);

          // Process buffered input across all filters and store to output.
          DepthwiseConv2DKernel<T>::Run(
              args, padded_filter_inner_dim_size, out_r, out_c, filter_data,
              input_buffer_data, output + out_base);
        }
      }
	  input_buffer.resize(0,0);
    };

    const int64 total_shards = args.batch * args.out_rows;

    // Empirically tested to give reasonable performance boosts at batch size 1
    // without reducing throughput at batch size 32.
    const float kCostMultiplier = 2.5f;

	// TODO(andydavis): Estimate shard cost (in cycles) based on the number of
    // flops/loads/stores required to compute one shard.
    const int64 shard_cost = kCostMultiplier * args.out_cols * args.out_depth;

    shard(0, total_shards);
}

} // SGXDNN namespace
#endif
