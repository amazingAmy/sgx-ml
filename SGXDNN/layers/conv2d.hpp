#ifndef SGXDNN_CONV2D_H_
#define SGXDNN_CONV2D_H_

#define EIGEN_USE_TENSOR

#include <stdio.h>
#include <iostream>
#include <string>
#include <type_traits>
#include <assert.h>

#include "../mempool.hpp"
#include "../utils.hpp"
#include "../Crypto.h"
#include "layer.hpp"
#include "activation.hpp"
#include "eigen_spatial_convolutions.h"
#include <cmath>
#include "immintrin.h"

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

	template <class T1, class T2, class T3>
	void conv2d_im2col(const T1* input_data, int input_batches, int input_height, int input_width, int input_depth, 
					   T2* filter_data, int filter_height, int filter_width, int filter_count, 
					   int stride_rows, int stride_cols, Eigen::PaddingType padding, 
					   T3* output_data, int output_height, int output_width, MemPool* mem_pool_);

	template <typename T>
	class Conv2D : public Layer<T>
	{
	public:
		Conv2D(const std::string& name,
			   const array4d input_shape,//上一层的输出规模
               const array4d& kernel_shape,
               const int row_stride,
               const int col_stride,
               const Eigen::PaddingType& padding,
               T* r_left, T* r_right, T* kernel, T* bias,
			   MemPool* mem_pool,
			   bool is_verif_mode,
			   bool verif_preproc,
			   const std::string& activation_type
			   ): Layer<T>(name, input_shape),
			   kernel_shape_(kernel_shape),
               row_stride_(row_stride),
               col_stride_(col_stride),
               padding_(padding),
               kernel_data_(nullptr),
               bias_data_(nullptr),
               kernel_(NULL, kernel_shape),
               bias_(NULL, kernel_shape[3]),
			   mem_pool_(mem_pool),
			   activation_type_(activation_type),
			   h(input_shape[1]),
			   w(input_shape[2]),
			   ch_in(kernel_shape[2]),
			   h_out(0),
			   w_out(0),
			   ch_out(kernel_shape[3]),
			   patch_size(kernel_shape[0] * kernel_shape[1]),
			   image_size(input_shape[1] * input_shape[2]),
			   out_image_size(0)
		{
			const int filter_rows = kernel_shape[0];//定义卷积核的参数
			const int filter_cols = kernel_shape[1];//定义卷积核的参数

			//分别获取卷积之后的输出大小以及需要增加的padding规模
			GetWindowedOutputSize(h, filter_rows, row_stride_,
								  padding_, &h_out, &pad_rows_);
			GetWindowedOutputSize(w, filter_cols, col_stride_,
								  padding_, &w_out, &pad_cols_);

			printf("in Conv2D with out_shape = (%d, %d, %d) and padding size:(%d,%d)\n", h_out, w_out, ch_out,pad_rows_,pad_cols_);
			output_shape_ = {0, h_out, w_out, ch_out};
			output_size_ = h_out * w_out * ch_out;
			input_shape_ = {0, h, w, ch_in};
			input_size_ = h * w * ch_in;
			out_image_size = h_out * w_out;
			long kernel_size = kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3];

			//load data into enclave
			lazy_load_ = false;
			#ifdef USE_SGX
			if (mem_pool_->allocated_bytes >= 50 * 1024 * 1024) {
				lazy_load_ = true;
				printf("lazy loading convolution of size %ld\n", kernel_size);
			}
			#endif
			
			// copy kernel and bias
			if (lazy_load_) {
				kernel_data_ = kernel;
				mac = new MAC();
			} else {
				kernel_data_ = mem_pool_->alloc<T>(kernel_size); 
				std::copy(kernel, kernel + kernel_size, kernel_data_);
				new (&kernel_) typename TTypes<T, 4>::Tensor(kernel_data_, kernel_shape);
			}

			long bias_size = kernel_shape[3];
			bias_data_ = new T[bias_size];
			std::copy(bias, bias + bias_size, bias_data_);
			new (&bias_) typename TTypes<T>::ConstVec(bias_data_, kernel_shape[3]);
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

		void set_activation_type(const std::string act) {
			activation_type_ = act;
		}

		int h;
		int w;
		int ch_in;//表示输入的通道数，也就是输出的层数
		int h_out;
		int w_out;
		int ch_out;//表示输出的通道数，也就是卷积核的个数
		int patch_size;
		int image_size;
		int out_image_size;

	protected:

		TensorMap<T, 4> apply_impl(TensorMap<T, 4> input, void* device_ptr = NULL, bool release_input=true) override
		{
			#ifdef EIGEN_USE_THREADS
			Eigen::ThreadPoolDevice* d = static_cast<Eigen::ThreadPoolDevice*>(device_ptr);
			#endif

			T* kernel_temp;
			if (lazy_load_) {
				long kernel_size = kernel_shape_[0] * kernel_shape_[1] * kernel_shape_[2] * kernel_shape_[3];
                kernel_temp = mem_pool_->alloc<T>(kernel_size); 
                std::copy(kernel_data_, kernel_data_ + kernel_size, kernel_temp);
                new (&kernel_) typename TTypes<T, 4>::Tensor(kernel_temp, kernel_shape_);
				// TODO actually check mac...
                Tag tag = mac->mac((uint8_t*) kernel_temp, kernel_size * sizeof(T));
			}
  
			const int batch = input.dimension(0);
			output_shape_[0] = batch;

			// allocate memory to store the output
			T* output_mem_ = mem_pool_->alloc<T>(batch * output_size_);
			auto output_map = TensorMap<T, 4>(output_mem_, output_shape_);
		
			sgx_time_t start = get_time();
			output_map = Eigen::SpatialConvolution(input, kernel_, col_stride_, row_stride_, padding_);
			sgx_time_t end = get_time();

			if (TIMING) { printf("convd (%ld x %ld x %ld) took %.4f seconds\n", input.dimension(1), input.dimension(2), input.dimension(3), get_elapsed_time(start, end)); };

			if (lazy_load_) {
				mem_pool_->release(kernel_temp);
			}

			// add bias
			const int bias_size = bias_.dimension(0);
			const int rest_size = output_map.size() / bias_size;
			Eigen::DSizes<int, 1> one_d(output_map.size());
			Eigen::DSizes<int, 1> bcast(rest_size);
	
			output_map.reshape(one_d) = output_map.reshape(one_d) + bias_.broadcast(bcast).reshape(one_d);
			if (release_input) {
				mem_pool_->release(input.data());
			}

			return output_map;
		}

		T* kernel_data_;//保存记录权重的指针
		T* bias_data_;//保存记录偏置的指针
		TensorMap<T, 4> kernel_;//权重矩阵
		TensorMap<T, 1> bias_;//偏置矩阵

		const Eigen::PaddingType padding_;
		const int row_stride_;//步长
		const int col_stride_;
		int pad_rows_;
		int pad_cols_;

		MemPool* mem_pool_;

		array4d input_shape_;
		array4d kernel_shape_;
		int input_size_;

		array4d output_shape_;
		int output_size_;

		bool lazy_load_;

		std::string activation_type_;
		MAC* mac;
	};

 const size_t kMaxChunkSize = (1 * 1024 * 1024);

 // adapted from tensorflow repository
 template <class T1, class T2, class T3>
 void conv2d_im2col(const T1* input_data,
          int input_batches, int input_height, int input_width,
          int input_depth, T2* filter_data, int filter_height,
          int filter_width, int filter_count, int stride_rows,
          int stride_cols, Eigen::PaddingType padding, T3* output_data,
          int output_height, int output_width, MemPool* mem_pool_) {


    // These calculations define how the patches will be positioned within the
    // input image. The actual definitions are quite complex, and rely on the
    // previously-calculated output size.
    int filter_left_offset;
    int filter_top_offset;
    if (padding == Eigen::PaddingType::PADDING_VALID) {
      filter_left_offset =
          ((output_width - 1) * stride_cols + filter_width - input_width + 1) /
          2;
      filter_top_offset = ((output_height - 1) * stride_rows + filter_height -
                           input_height + 1) /
                          2;
    } else {
      filter_left_offset =
          ((output_width - 1) * stride_cols + filter_width - input_width) / 2;
      filter_top_offset =
          ((output_height - 1) * stride_rows + filter_height - input_height) /
          2;
    }

    // The im2col buffer has # of patches rows, and # of filters cols.
    // It's laid out like this, in row major order in memory:
    //        < filter value count >
    //   ^   +---------------------+
    // patch |                     |
    // count |                     |
    //   v   +---------------------+
    // Each patch row contains a filter_width x filter_height patch of the
    // input, with the depth channel as the most contiguous in memory, followed
    // by the width, then the height. This is the standard memory order in the
    // image world if it helps to visualize it.
    const int filter_value_count = filter_width * filter_height * input_depth;
    assert((filter_value_count * sizeof(T1)) <= kMaxChunkSize);
    const int64 patches_per_chunk =
        kMaxChunkSize / (filter_value_count * sizeof(T1));
    const int64 chunk_value_count =
        (kMaxChunkSize + (sizeof(T1) - 1)) / sizeof(T1);

	// Because memory allocation is very expensive on mobile platforms, try to
    // allocate a persistent buffer that will be kept around between calls. We
    // use TensorFlow's resource management to ensure that the memory will be
    // released when the session is over.
	T1* im2col_buffer = mem_pool_->alloc<T1>(chunk_value_count);

    const int64 patch_count = (input_batches * output_height * output_width);
    const int64 chunk_count =
        (patch_count + (patches_per_chunk - 1)) / patches_per_chunk;
    for (int64 chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
      const int64 patch_index_start = chunk_index * patches_per_chunk;
      const int64 patch_index_end =
          std::min(patch_index_start + patches_per_chunk, patch_count);
      for (int64 patch_index = patch_index_start; patch_index < patch_index_end;
           ++patch_index) {
        const int64 batch = patch_index / (output_height * output_width);
        const int64 out_y = (patch_index / output_width) % output_height;
        const int64 out_x = patch_index % output_width;
        const T1* input_batch_start =
            input_data + (batch * input_height * input_width * input_depth);
        const int in_y_origin = (out_y * stride_rows) - filter_top_offset;
        const int in_x_origin = (out_x * stride_cols) - filter_left_offset;
        const int patch_index_within_chunk = patch_index % patches_per_chunk;
		T1* im2col_patch_start =
            im2col_buffer + (patch_index_within_chunk * filter_value_count);
        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
          const int in_y = in_y_origin + filter_y;
          T1* im2col_row_start =
              im2col_patch_start + (filter_y * filter_width * input_depth);
          // If we're off the top or the bottom of the input, fill the
          // whole row with zeroes.
          if ((in_y < 0) || (in_y >= input_height)) {
            T1* im2col_row_end =
                im2col_row_start + (filter_width * input_depth);
            std::fill(im2col_row_start, im2col_row_end, T1(0));
          } else {
            // What we're doing here is trying to copy and fill the im2col
            // buffer as efficiently as possible, using functions to set or
            // duplicate values en masse. We know we don't have to worry about
            // vertical edges because we dealt with that case above, so we
            // just need to handle filters that overlap the left or right
            // edges. Here's what that looks like:
            //
            // < left_zero_count > < center_copy_count > < right_zero_count >
            // +------------------+---------------------+--------------------+
            // |     (filter)     |       (image)       |      (filter)      |
            // +------------------+---------------------+--------------------+
            // in_x_origin        0                 input_width       in_x_end
            //
            // In reality it's unlikely that a filter patch will be wider
            // than an input, but this shows all the edge cases.
            // We use std::fill() to set the left and right sections to zeroes
            // and std::copy() to copy over the input data for the center.
            const int in_x_end = in_x_origin + filter_width;
            const int left_zero_count = std::max(0, 0 - in_x_origin);
            const int right_zero_count = std::max(0, in_x_end - input_width);
            const int center_copy_count =
                filter_width - (left_zero_count + right_zero_count);
            if (left_zero_count > 0) {
              T1* im2col_left_start = im2col_row_start;
              T1* im2col_left_end =
                  im2col_left_start + (left_zero_count * input_depth);
              std::fill(im2col_left_start, im2col_left_end, T1(0));
            }
            if (center_copy_count > 0) {
              const T1* input_row_start =
                  input_batch_start + (in_y * input_width * input_depth) +
                  (std::max(0, in_x_origin) * input_depth);
			  const T1* input_row_end =
                  input_row_start + (center_copy_count * input_depth);
              T1* im2col_center_start =
                  im2col_row_start + (left_zero_count * input_depth);
              std::copy(input_row_start, input_row_end, im2col_center_start);
            }
            if (right_zero_count > 0) {
              T1* im2col_right_start =
                  im2col_row_start +
                  ((left_zero_count + center_copy_count) * input_depth);
              T1* im2col_right_end =
                  im2col_right_start + (right_zero_count * input_depth);
              std::fill(im2col_right_start, im2col_right_end, T1(0));
            }
          }
        }
      }
      // Now we've assembled a set of image patches into a matrix, apply a
      // GEMM matrix multiply of the patches as rows, times the filter
      // weights in columns, to get partial results in the output matrix.
      const int how_many_patches = patch_index_end - patch_index_start;
      const int m = how_many_patches;
      const int n = filter_count;
      const int k = filter_value_count;
      const int lda = filter_value_count;
      const int ldb = filter_count;
      const int ldc = filter_count;
      T3* chunk_output_data = output_data + (patch_index_start * filter_count);

      Eigen::Map<Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> buffer_map(im2col_buffer, m, k);
      Eigen::Map<Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> kernel_map(filter_data, k, n);
      Eigen::Map<Eigen::Matrix<T3, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> output_map(chunk_output_data, m, n);
      output_map = buffer_map.template cast<T3>() * kernel_map.template cast<T3>();

   }
   mem_pool_->release(im2col_buffer);
 }
} //SGXDNN namespace

#endif
