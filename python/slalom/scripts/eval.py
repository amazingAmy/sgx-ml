from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from keras import backend
from keras.models import Model

from python import imagenet
from python.slalom.models import get_model, get_test_model
from python.slalom.quant_layers import transform,DenseQ,Dense
from python.slalom.utils import Results, timer
from python.slalom.sgxdnn import model_to_json, SGXDNNUtils, mod_test
from python.slalom.utils import get_all_layers

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
                #model, model_info = get_model(args.model_name, args.batch_size, include_top=not args.no_top)
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
                weight = []
                all_new_layer = get_all_layers(model)
                all_layer = get_all_layers(model_copy)
                # all_layer = get_all_layers(model_copy)
                for new_layer in all_new_layer:
                    if isinstance(new_layer,DenseQ):
                        weight.append(new_layer.get_weights()[0])
                print("find {} Dense layers.".format(len(weight)))
                for layer in all_layer:
                    if isinstance(layer,Dense):
                        print("find Dense and check.")
                        new_layer_weight = weight.pop(0)
                        assert ((layer.get_weights()[0]==new_layer_weight).all())
                print("...weight check success...")

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

            print('start preparing data...')

            # from multiprocessing.dummy import Pool as ThreadPool
            # pool = ThreadPool(3)

            dataset_images = np.random.rand(50*args.batch_size,args.batch_size, 32, 32, 3)
            labels = np.zeros((50*args.batch_size,args.batch_size,10))
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    random_number = np.random.randint(0,9)
                    labels[i][j][random_number] = 1
            #dataset_images, labels = tf.train.batch([dataset_images, labels], batch_size=args.batch_size,
            #                                       num_threads=1, capacity=5 * args.batch_size)
            # print(type(dataset_images), type(labels), sep='\n')
            # print(dataset_images.shape(),labels.shape())
            #print(sess.run(tf.shape(dataset_images)), sess.run(tf.shape(labels)))
            #images, true_labels = sess.run([dataset_images, labels])
            # images = dataset_images.eval()
            # true_labels = labels.eval()
            print('...data prepared')
            for i in range(num_batches):
                # images, true_labels = sess.run([dataset_images, labels])
                images = dataset_images[i]
                true_labels = labels[i]
                print("input images: {}".format(np.sum(np.abs(images))))

                if args.mode in ['tf-gpu', 'tf-cpu']:
                    res.start_timer()
                    preds = sess.run(model.outputs[0], feed_dict={model.inputs[0]: np.array(images),
                                                                  backend.learning_phase(): 0},
                                     options=run_options, run_metadata=run_metadata)

                    print(np.sum(np.abs(images)), np.sum(np.abs(preds)))
                    preds = np.reshape(preds, (args.batch_size, num_classes))
                    print(preds)
                    res.end_timer(size=len(images))
                    res.record_acc(preds, true_labels)
                    res.print_results()

                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline.json', 'w') as f:
                        f.write(ctf)

                else:
                    res.start_timer()

                    # no verify
                    def func(data):
                        return sgxutils.predict(data[1], num_classes=num_classes, eid_idx=0)

                    #mse
                    def mse(array_a,array_b):
                        array_a = np.array(array_a)
                        array_b = np.array(array_b)
                        return ((array_a-array_b)**2).mean()

                    # all_data = [(i, images[i:i+1]) for i in range(args.batch_size)]
                    # preds = np.vstack(pool.map(func, all_data))
                    # np.set_printoptions(suppress=True, threshold=np.nan,precision=4)

                    if args.train:
                        print("*********sgxdnn:*********")
                        for k in range(args.epoch):
                            print("Epoch {}/{}".format(k+1,args.epoch),end=' ')
                            final_pred = []
                            final_label = [np.argmax(x) for x in true_labels]
                            loss = 0

                            res,time = sgxutils.train(images,true_labels,num_classes=num_classes,learn_rate=0.01)
                            if "mse" in model.loss:
                                loss = np.sum((res-true_labels)**2)/res.size
                            elif "crossentropy" in model.loss:
                                loss = -np.sum(true_labels*np.log(res))/args.batch_size

                            for x in res:
                                final_pred.append(np.argmax(x))
                            final_pred = np.array(final_pred)
                            final_label = np.array(final_label)
                            print("- time:{}s {}ms".format(round(time),round((time-round(time))*1000)),"- acc:{0:.4f}".format(sum(final_pred==final_label)/args.batch_size),"- loss:{0:.4f}".format(loss))

                            # preds = []
                            # for j in range(args.batch_size):
                            #     pred = sgxutils.predict(images[j:j + 1], num_classes=num_classes)
                            #     preds.append(pred)
                            # preds = np.vstack(preds)
                        preds = sgxutils.predict(images,num_classes=num_classes)
                        if model.loss=='mse':
                            loss = np.sum((preds-true_labels)**2)/preds.size
                        elif 'crossentropy' in model.loss:
                            loss = -np.sum(true_labels*np.log(preds))/args.batch_size

                        preds = [np.argmax(x) for x in preds]
                        real = [np.argmax(x) for x in true_labels]
                        print("pred:",preds,"label:",real,"accuracy: {0:.4f}".format(np.sum(np.array(preds)==np.array(real))/args.batch_size),"loss:",loss)
                        print("\n*************************************\n**************sgxdnn over************\n*************************************\n")

                        model_copy.fit(images,true_labels,batch_size=args.batch_size,epochs=args.epoch)
                        preds = model_copy.predict(images)

                        if model.loss=='mse':
                            loss = np.sum((preds-true_labels)**2)/preds.size
                        elif 'crossentropy' in model.loss:
                            loss = -np.sum(true_labels*np.log(preds))/args.batch_size

                        preds = [x.argmax() for x in preds]
                        print("pred:",preds,"label:",real,"accuracy: {0:.4f}".format(np.sum(np.array(preds)==np.array(real))/args.batch_size),"loss:",loss)
                    else:
                        pred = sgxutils.predict(images,num_classes=num_classes)
                        print(pred.shape)
                        # for j in range(args.batch_size):
                            # pred = sgxutils.predict(images[j:j + 1], num_classes=num_classes)
                            #preds.append(pred)
                            # print(pred)
                        preds2 = model_copy.predict(images)
                        # preds = np.vstack(preds)
                        print(pred,preds2,pred==preds2)
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
                       help='Train instead of verify.' )
    parser.add_argument('--epoch',type=int,default=1,
                        help='How many times you want to train the whole data set.')
    args = parser.parse_args()

    tf.app.run()
