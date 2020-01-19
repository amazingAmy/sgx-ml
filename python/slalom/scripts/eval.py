from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import to_categorical

# from python import imagenet
from python.slalom.models import get_model, get_test_model
from python.slalom.quant_layers import transform, DenseQ, Dense
from python.slalom.utils import Results, timer
from python.slalom.sgxdnn import model_to_json, SGXDNNUtils, mod_test

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_USE_DEEP_CONV2D"] = '0'

DTYPE_VERIFY = np.float32


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.set_random_seed(1234)

    with tf.Graph().as_default():
        # Prepare graph
        num_batches = args.max_num_batches

        sgxutils = None

        if args.mode == 'tf-gpu':
            assert not args.use_sgx

            device = '/gpu:0'
            config = tf.ConfigProto(log_device_placement=False)
            config.allow_soft_placement = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.90
            config.gpu_options.allow_growth = True

        elif args.mode == 'tf-cpu':
            assert not args.verify and not args.use_sgx

            device = '/cpu:0'
            # config = tf.ConfigProto(log_device_placement=False)
            config = tf.ConfigProto(log_device_placement=False, device_count={'CPU': 1, 'GPU': 0})
            config.intra_op_parallelism_threads = 1
            config.inter_op_parallelism_threads = 1

        else:
            assert args.mode == 'sgxdnn'

            device = '/gpu:0'
            config = tf.ConfigProto(log_device_placement=False)
            config.allow_soft_placement = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
            config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            with tf.device(device):
                # model, model_info = get_model(args.model_name, args.batch_size, include_top=not args.no_top)
                model, model_info = get_test_model(args.batch_size)
            model_copy = model
            model, linear_ops_in, linear_ops_out = transform(model, log=False, quantize=args.verify,
                                                             verif_preproc=args.preproc,
                                                             bits_w=model_info['bits_w'],
                                                             bits_x=model_info['bits_x'])
            # dataset_images, labels = imagenet.load_validation(args.input_dir, args.batch_size,
            #                                                 preprocess=model_info['preprocess'],
            #                                              num_preprocessing_threads=1)

            if args.mode == 'sgxdnn':
                # check weight equal or not
                # sgxutils = SGXDNNUtils(args.use_sgx, num_enclaves=args.batch_size)
                # sgxutils = SGXDNNUtils(args.use_sgx, num_enclaves=2)
                sgxutils = SGXDNNUtils(args.use_sgx)

                dtype = np.float32 if not args.verify else DTYPE_VERIFY
                model_json, weights = model_to_json(sess, model, args.preproc, dtype=dtype,
                                                    bits_w=model_info['bits_w'], bits_x=model_info['bits_x'])
                sgxutils.load_model(model_json, weights, dtype=dtype, verify=args.verify, verify_preproc=args.preproc)

            num_classes = np.prod(model.output.get_shape().as_list()[1:])
            print("num_classes: {}".format(num_classes))

            print_acc = (num_classes == 10)
            res = Results(acc=print_acc)
            coord = tf.train.Coordinator()
            init = tf.initialize_all_variables()
            # sess.run(init)
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            # from multiprocessing.dummy import Pool as ThreadPool
            # pool = ThreadPool(3)

            (X_train, y_train), (X_test, y_test) = cifar10.load_data()
            y_train = y_train.reshape(y_train.shape[0])
            y_test = y_test.reshape(y_test.shape[0])
            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')
            X_train /= 255
            X_test /= 255
            y_train = to_categorical(y_train, num_classes)
            y_test = to_categorical(y_test, num_classes)

            num_batches = int(X_train.shape[0] / args.batch_size)
            print('training batch number :{}'.format(num_batches))
            lr = 0.001
            for k in range(args.epoch):
                if (k + 1) % 10:
                    lr *= 0.95
                print('Epoch {}/{}'.format(k + 1, args.epoch))
                for i in range(num_batches):
                    done_number = int(30 * (i + 1) / num_batches)
                    wait_to_be_done = 30 - done_number
                    print("\r{}/{} [{}>{}] {:.2f}% ".format((i + 1) * args.batch_size, X_train.shape[0],
                                                            '=' * done_number, '.' * wait_to_be_done,
                                                            100 * (i + 1) / num_batches), end='')
                    images = X_train[(i * args.batch_size):((i + 1) * args.batch_size)]
                    labels = y_train[(i * args.batch_size):((i + 1) * args.batch_size)]
                    if args.train:
                        loss_batch, acc_batch = sgxutils.train(images, labels, num_classes=num_classes,
                                                               learn_rate=lr)
                        print(' - loss :{:.4f} - acc :{:.4f}'.format(loss_batch, acc_batch), end='')
                sys.stdout.flush()
            #        res.start_timer()

            #        # no verify
            #        def func(data):
            #            return sgxutils.predict(data[1], num_classes=num_classes, eid_idx=0)

            #        def get_gradient(model_copy,layer_index,images):
            #           # 下面是求出layer层导数，用来debug
            #           # layer = model_copy.layers[layer_index+1 if layer_index>0 else layer_index]
            #           layer = model_copy.layers[layer_index]
            #           print(layer.name)
            #           grad = model_copy.optimizer.get_gradients(model_copy.total_loss,layer.output)
            #           input_tensors = [model_copy.inputs[0], # input data
            #                            model_copy.sample_weights[0], # how much to weight each sample by
            #                            model_copy.targets[0], # labels
            #                            K.learning_phase(), # train or test mode
            #                            ]
            #           get_gradients = K.function(inputs=input_tensors, outputs=grad)
            #           inputs = [images, # X
            #                     np.ones(args.batch_size), # sample weights
            #                     labels, # y
            #                     0 # learning phase in TEST mode
            #                     ]
            #           grad = get_gradients(inputs)[0]
            #           return grad
            # images = np.random.random((200, 32, 32, 3))
            # labels = np.zeros((200, 10))
            # for i in range(200):
            #     index = np.random.randint(0, 10)
            #     labels[i][index] = 1
            model_copy.fit(X_train, y_train, batch_size=32, epochs=1)
            coord.request_stop()
            coord.join(threads)
    if sgxutils is not None:
        sgxutils.destroy()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', type=str,
                        choices=['vgg_16', 'vgg_19', 'inception_v3', 'mobilenet', 'mobilenet_sep',
                                 'resnet_18', 'resnet_34', 'resnet_50', 'resnet_101', 'resnet_152'])

    parser.add_argument('mode', type=str, choices=['tf-gpu', 'tf-cpu', 'sgxdnn'])

    parser.add_argument('--input_dir', type=str,
                        default='../imagenet/',
                        help='Input directory with images.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='How many images process at one time.')
    parser.add_argument('--max_num_batches', type=int, default=2,
                        help='Max number of batches to evaluate.')
    parser.add_argument('--verify', action='store_true',
                        help='Activate verification.')
    parser.add_argument('--preproc', action='store_true',
                        help='Use preprocessing for verification.')
    parser.add_argument('--use_sgx', action='store_true')
    parser.add_argument('--verify_batched', action='store_true',
                        help='Use batched verification.')
    parser.add_argument('--no_top', action='store_true',
                        help='Omit top part of network.')
    parser.add_argument('--train', action='store_true',
                        help='Train instead of verify.')
    parser.add_argument('--epoch', type=int, default=1,
                        help='How many times you want to train the whole data set.')
    args = parser.parse_args()

    tf.app.run()
