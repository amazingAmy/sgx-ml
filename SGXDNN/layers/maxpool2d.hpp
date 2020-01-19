#ifndef SGXDNN_MAXPOOL2D_H_
#define SGXDNN_MAXPOOL2D_H_

#include <iostream>
#include <string>

#include "../mempool.hpp"
#include "layer.hpp"
#include "eigen_maxpool.h"

using namespace tensorflow;

namespace SGXDNN
{

	template<typename T>
	void fast_maxpool(T* input, T* output,
					  int batch, int input_rows_, int input_cols_, int input_depth_, int out_rows_, int out_cols_,
					  int window_rows_, int window_cols_, int pad_rows_, int pad_cols_, int row_stride_, int col_stride_,
					  bool avg_pool = false)
	{
		typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> ConstEigenMatrixMap;
		typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> EigenMatrixMap;

		ConstEigenMatrixMap in_mat(input, input_depth_,
								   input_cols_ * input_rows_ * batch);

		EigenMatrixMap out_mat(output, input_depth_, out_rows_ * out_cols_ * batch);

		// The following code basically does the following:
		// 1. Flattens the input and output tensors into two dimensional arrays.
		//    tensor_in_as_matrix:
		//      depth by (tensor_in_cols * tensor_in_rows * tensor_in_batch)
		//    output_as_matrix:
		//      depth by (out_width * out_height * tensor_in_batch)
		//
		// 2. Walks through the set of columns in the flattened
		// tensor_in_as_matrix,
		//    and updates the corresponding column(s) in output_as_matrix with the
		//    max value.
		auto shard = [&in_mat, &out_mat, input_rows_, input_cols_, input_depth_, out_rows_, out_cols_,
					  window_rows_, window_cols_, pad_rows_, pad_cols_, row_stride_, col_stride_, avg_pool](long start, long limit) {
			const int in_rows = input_rows_;
			const int in_cols = input_cols_;
			const int window_rows = window_rows_;//每次求最大值或平均值的窗口行数
			const int window_cols = window_cols_;//每次求最大值或平均值的窗口行数
			const int pad_rows = pad_rows_;
			const int pad_cols = pad_cols_;
			const int row_stride = row_stride_;
			const int col_stride = col_stride_;
			const int out_height = out_rows_;
			const int out_width = out_cols_;
			const int input_depth = input_depth_;

			{
		  		// Initializes the output tensor with MIN<T>.
				//将从start开始的,limit-start个输出图片size，归为一个shard
		  		const int output_image_size = out_height * out_width * input_depth;
		  		EigenMatrixMap out_shard(out_mat.data() + start * output_image_size,
								   1, (limit - start) * output_image_size);

				if (avg_pool) {
		  			out_shard.setConstant((T) 0.0);
				} else {
		  			out_shard.setConstant(Eigen::NumTraits<T>::lowest());
				}
			}

			for (int b = start; b < limit; ++b) {
			    const int out_offset_batch = b * out_height;
			    for (int h = 0; h < in_rows; ++h) {
				  for (int w = 0; w < in_cols; ++w) {
		  	  	    // (h_start, h_end) * (w_start, w_end) is the range that the input
			  	    // vector projects to.
					//注意h_end，w_end是取不到的界
			  	    const int hpad = h + pad_rows;
			  	    const int wpad = w + pad_cols;
			  	    const int h_start = (hpad < window_rows)
											? 0
											: (hpad - window_rows) / row_stride + 1;
			  	    const int h_end = std::min(hpad / row_stride + 1, out_height);
			  	    const int w_start = (wpad < window_cols)
											? 0
											: (wpad - window_cols) / col_stride + 1;
			  	    const int w_end = std::min(wpad / col_stride + 1, out_width);
			  	    // compute elementwise max
					//同一个输入(h,w)出现的位置是output中的(h_start, h_end) * (w_start, w_end)
					//相应位置对应的那个window
			  	    const int in_offset = (b * in_rows + h) * in_cols + w;
			  	    for (int ph = h_start; ph < h_end; ++ph) {
					  const int out_offset_base =
					  	  (out_offset_batch + ph) * out_width;
					  for (int pw = w_start; pw < w_end; ++pw) {
				  	    const int out_offset = out_offset_base + pw;
						if (avg_pool) {
							out_mat.col(out_offset) += in_mat.col(in_offset) / ((T)(window_rows * window_cols));
						} else {
				  	    	out_mat.col(out_offset) = out_mat.col(out_offset).cwiseMax(in_mat.col(in_offset));
						}
					  }
			  	    }
				  }
			    }
			}
		};

		shard(0, batch);
	}


	template <typename T> class MaxPool2D : public Layer<T>
	{
	public:
		explicit MaxPool2D(
                const std::string& name,
				const array4d input_shape,
                const int window_rows,
                const int window_cols,
                const int row_stride,
                const int col_stride,
                const Eigen::PaddingType& padding,
				const bool avg_pool,
				MemPool* mem_pool
                ): Layer<T>(name, input_shape),
                window_rows_(window_rows),
                window_cols_(window_cols),
                row_stride_(row_stride),
                col_stride_(col_stride),
                padding_(padding),
				avg_pool_(avg_pool),
				mem_pool_(mem_pool)
		{
			input_rows_ = input_shape[1];
			input_cols_ = input_shape[2];
			input_depth_ = input_shape[3];

			GetWindowedOutputSize(input_rows_, window_rows_, row_stride_,
								  padding_, &out_rows_, &pad_rows_);
			GetWindowedOutputSize(input_cols_, window_cols_, col_stride_,
								  padding_, &out_cols_, &pad_cols_);

			output_shape_ = {0, out_rows_, out_cols_, input_depth_};
			output_size_ = out_rows_ * out_cols_ * input_depth_;

			printf("in Pool2D with window = (%d, %d), stride = (%d, %d), padding = %d, out_shape = (%d, %d, %d), pad = (%d, %d)\n",
			        window_rows_, window_cols_, row_stride_, col_stride_, padding_, out_rows_, out_cols_, input_depth_, pad_rows_, pad_cols_);
		}

		array4d output_shape() override
		{
			return output_shape_;
		}

		int output_size() override
		{
			return output_size_;
		}

	protected:

		TensorMap<T, 4> apply_impl(TensorMap<T, 4> input, void* device_ptr = NULL, bool release_input = true) override
		{
			int batch = input.dimension(0);
			output_shape_[0] = batch;
			T* output_mem_ = mem_pool_->alloc<T>(batch * output_size_);
			auto output_map = TensorMap<T, 4>(output_mem_, output_shape_);

			fast_maxpool(input.data(), output_map.data(),
                         batch, input_rows_, input_cols_, input_depth_, out_rows_, out_cols_,
                      	 window_rows_, window_cols_, pad_rows_, pad_cols_, row_stride_, col_stride_, avg_pool_);
            if(release_input)
    			mem_pool_->release(input.data());
			// std::cout<<"MaxPool input:"<<input.dimensions()<<"\nWindows:"<<window_rows_<<"x"<<window_cols_<<"\n avg pool:"<<avg_pool_<<std::endl;
			return output_map;
		}

		TensorMap<T,4> average_bp(TensorMap<T,4> &result_map,TensorMap<T,4>&der)
        {
            typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> EigenMatrixMap;
            int batch = result_map.dimension(0);
            EigenMatrixMap der_mat(der.data(),input_depth_,batch*out_rows_*out_cols_);
            EigenMatrixMap result_mat(result_map.data(),input_depth_,batch*input_cols_*input_rows_);

            auto offset = [&](int height,int width) -> int
            {
                return height * input_cols_ + width;
            };

            auto excluded_padding = [&](int h_begin,int w_begin) ->int
            {
                int top_excluded = pad_rows_-h_begin;
                int left_excluded =pad_cols_-w_begin;
                int bottom_excluded = h_begin+window_rows_-input_rows_;
                int right_excluded =  w_begin+window_cols_-input_cols_;
                top_excluded = top_excluded>0?top_excluded:0;
                left_excluded = left_excluded>0?left_excluded:0;
                bottom_excluded = bottom_excluded>0?bottom_excluded:0;
                right_excluded = right_excluded>0?right_excluded:0;
                return (window_rows_-top_excluded-bottom_excluded)*(window_cols_-left_excluded-right_excluded);
            };

            for(int b=0;b<batch;++b)
            {
                int in_offset_base = b * input_rows_ * input_cols_;
                for(int out_h=0;out_h<out_rows_;++out_h)
                    for(int out_w=0;out_w<out_cols_;++out_w)
                    {
                        int out_offset = (b * out_rows_ + out_h) * out_cols_ + out_w;
                        int h_begin = out_h * row_stride_;
                        int w_begin = out_w * col_stride_;

                        for(int h_in = 0;h_in < window_rows_;++h_in)
                            for(int w_in = 0;w_in < window_cols_;++w_in)
                            {
                                if((h_in+h_begin)<pad_rows_||(w_in+w_begin)<pad_cols_)
                                {
                                    continue;
                                }
                                else if((h_in+h_begin)<(pad_rows_+input_rows_)&&(w_in+w_begin)<(pad_cols_+input_cols_))
                                {
                                    int in_offset = in_offset_base + offset(h_in + h_begin - pad_rows_, w_in + w_begin - pad_cols_);
                                    result_mat.col(in_offset) += der_mat.col(out_offset) / static_cast<T>(window_cols_*window_rows_);
                                }

                            }
                    }
            }
        }

        TensorMap<T,4> max_bp(TensorMap<T,4>&input,TensorMap<T,4> &result_map,TensorMap<T,4>&der)
        {
		    int batch = input.dimension(0);
            auto find_max = [&](int h_begin,int w_begin,int b,int d) -> array4d
            {
                int h_max=h_begin<pad_rows_?pad_rows_:h_begin;
                int w_max=w_begin<pad_cols_?pad_cols_:w_begin;
                array4d result;
                for (int h = h_max; h < h_begin + window_rows_; ++h)
                    for (int w = h_max; w < w_begin + window_cols_; ++w)
                    {
                        if(h<pad_rows_+input_rows_&&w<pad_cols_+input_cols_)
                        {
                            if (input(b, h-pad_rows_, w-pad_cols_, d) > input(b, h_max-pad_rows_, w_max-pad_cols_, d))
                            {
                                h_max = h;
                                w_max = w;
                            }
                        }
                    }
                result = {b,h_max-pad_rows_,w_max-pad_cols_,d};
                return result;
            };

            for(int i=0;i<batch;++i)
                for(int j=0;j<input_depth_;++j)
                {
                    for (int out_h = 0; out_h < out_rows_; ++out_h)
                        for (int out_w = 0; out_w < out_cols_; ++out_w)
                        {
                            auto max = find_max(out_h * row_stride_, out_w * col_stride_, i, j);
                            result_map(max) += der(i, out_h, out_w, j);
                        }
                }
        }

        TensorMap<T, 4> back_prop(TensorMap<T, 4> input, TensorMap<T, 4> der, float learn_rate)override
        {
		    const int batch = input.dimension(0);
		    //保证返回导数张量形状正确
            new(&der)TensorMap<T,4>(der.data(),batch,out_rows_,out_cols_,input_depth_);
		    T*result_data = mem_pool_->alloc<T>(input.size());
		    TensorMap<T,4>result_map(result_data,input.dimensions());
		    result_map.setZero();
		    if(avg_pool_)
            {
		        average_bp(result_map,der);
            } else
            {
		        max_bp(input,result_map,der);
            }
		    mem_pool_->release(der.data());
            return result_map;
        }

		int input_rows_;
		int input_cols_;
		int input_depth_;

		int out_rows_;
		int out_cols_;
		int pad_rows_;
		int pad_cols_;

		const Eigen::PaddingType padding_;
		const int window_rows_;
		const int window_cols_;
		const int row_stride_;
		const int col_stride_;
		const bool avg_pool_;
		MemPool* mem_pool_;

		array4d output_shape_;
		int output_size_;
	};
}
#endif
