# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import sys
import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow.python.ops import init_ops
import dlib


class DummyScope(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class GPUNetworkBuilder(object):
    """This class provides convenient methods for constructing feed-forward
    networks with internal data layout of 'NCHW'.
    """

    def __init__(self,
                 is_training,
                 dtype=tf.float32,
                 activation='RELU',
                 use_batch_norm=True,
                 batch_norm_config={'decay': 0.9,
                                    'epsilon': 1e-4,
                                    'scale': True,
                                    'zero_debias_moving_mean': False},
                 use_xla=False):
        self.dtype = dtype
        self.activation_func = activation
        self.is_training = is_training
        self.use_batch_norm = use_batch_norm
        self.batch_norm_config = batch_norm_config
        self._layer_counts = defaultdict(lambda: 0)
        if use_xla:
            self.jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
        else:
            self.jit_scope = DummyScope

    def _count_layer(self, layer_type):
        idx = self._layer_counts[layer_type]
        name = layer_type + str(idx)
        self._layer_counts[layer_type] += 1
        return name

    def _get_variable(self, name, shape, dtype=None,
                      initializer=None, seed=None):
        if dtype is None:
            dtype = self.dtype
        if initializer is None:
            initializer = init_ops.glorot_uniform_initializer(seed=seed)
        elif (isinstance(initializer, float) or
              isinstance(initializer, int)):
            initializer = tf.constant_initializer(float(initializer))
        return tf.get_variable(name, shape, dtype, initializer)

    def _to_nhwc(self, x):
        return tf.transpose(x, [0, 2, 3, 1])

    def _from_nhwc(self, x):
        return tf.transpose(x, [0, 3, 1, 2])

    def _bias(self, input_layer):
        num_outputs = input_layer.get_shape().as_list()[1]
        biases = self._get_variable('biases', [num_outputs], input_layer.dtype,
                                    initializer=0)
        if len(input_layer.get_shape()) == 4:
            return tf.nn.bias_add(input_layer, biases,
                                  data_format='NCHW')
        else:
            return input_layer + biases

    def _batch_norm(self, input_layer, scope):
        return tf.contrib.layers.batch_norm(input_layer,
                                            is_training=self.is_training,
                                            scope=scope,
                                            data_format='NCHW',
                                            fused=True,
                                            **self.batch_norm_config)

    def _bias_or_batch_norm(self, input_layer, scope, use_batch_norm):
        if use_batch_norm is None:
            use_batch_norm = self.use_batch_norm
        if use_batch_norm:
            return self._batch_norm(input_layer, scope)
        else:
            return self._bias(input_layer)

    def input_layer(self, input_layer):
        """Converts input data into the internal format"""
        with self.jit_scope():
            x = self._from_nhwc(input_layer)
            x = tf.cast(x, self.dtype)
            # Rescale and shift to [-1,1]
            x = x * (1. / 127.5) - 1
        return x

    def conv(self, input_layer, num_filters, filter_size,
             filter_strides=(1, 1), padding='SAME',
             activation=None, use_batch_norm=None):
        """Applies a 2D convolution layer that includes bias or batch-norm
        and an activation function.
        """
        num_inputs = input_layer.get_shape().as_list()[1]
        kernel_shape = [filter_size[0], filter_size[1],
                        num_inputs, num_filters]
        strides = [1, 1, filter_strides[0], filter_strides[1]]
        with tf.variable_scope(self._count_layer('conv')) as scope:
            kernel = self._get_variable('weights', kernel_shape,
                                        input_layer.dtype)
            if padding == 'SAME_RESNET':  # ResNet models require custom padding
                kh, kw = filter_size
                rate = 1
                kernel_size_effective = kh + (kw - 1) * (rate - 1)
                pad_total = kernel_size_effective - 1
                pad_beg = padgit_total // 2
                pad_end = pad_total - pad_beg
                padding = [[0, 0], [0, 0],
                           [pad_beg, pad_end], [pad_beg, pad_end]]
                input_layer = tf.pad(input_layer, padding)
                padding = 'VALID'
            x = tf.nn.conv2d(input_layer, kernel, strides, padding=padding, data_format='NCHW')
            x = self._bias_or_batch_norm(x, scope, use_batch_norm)
            x = self.activate(x, activation)
            return x

    def deconv(self, input_layer, num_filters, filter_size,
               filter_strides=(2, 2), padding='SAME',
               activation=None, use_batch_norm=None):
        """Applies a 'transposed convolution' layer that includes bias or
        batch-norm and an activation function.
        """
        num_inputs = input_layer.get_shape().as_list()[1]
        ih, iw = input_layer.get_shape().as_list()[2:]
        output_shape = [-1, num_filters,
                        ih * filter_strides[0], iw * filter_strides[1]]
        kernel_shape = [filter_size[0], filter_size[1],
                        num_filters, num_inputs]
        strides = [1, 1, filter_strides[0], filter_strides[1]]
        with tf.variable_scope(self._count_layer('deconv')) as scope:
            kernel = self._get_variable('weights', kernel_shape,
                                        input_layer.dtype)
            x = tf.nn.conv2d_transpose(input_layer, kernel, output_shape,
                                       strides, padding=padding,
                                       data_format='NCHW')
            x = self._bias_or_batch_norm(x, scope, use_batch_norm)
            x = self.activate(x, activation)
            return x

    def activate(self, input_layer, funcname=None):
        """Applies an activation function"""
        if isinstance(funcname, tuple):
            funcname = funcname[0]
            params = funcname[1:]
        if funcname is None:
            funcname = self.activation_func
        if funcname == 'LINEAR':
            return input_layer
        activation_map = {
            'RELU': tf.nn.relu,
            'RELU6': tf.nn.relu6,
            'ELU': tf.nn.elu,
            'SIGMOID': tf.nn.sigmoid,
            'TANH': tf.nn.tanh,
            'LRELU': lambda x, name: tf.maximum(params[0] * x, x, name=name)
        }
        return activation_map[funcname](input_layer, name=funcname.lower())

    def pool(self, input_layer, funcname, window_size,
             window_strides=(2, 2),
             padding='VALID'):
        """Applies spatial pooling"""
        pool_map = {
            'MAX': tf.nn.max_pool,
            'AVG': tf.nn.avg_pool
        }
        kernel_size = [1, 1, window_size[0], window_size[1]]
        kernel_strides = [1, 1, window_strides[0], window_strides[1]]
        return pool_map[funcname](input_layer, kernel_size, kernel_strides,
                                  padding, data_format='NCHW',
                                  name=funcname.lower())

    def spatial_avg(self, input_layer):
        """Averages over spatial dimensions (4D->2D)"""
        return tf.reduce_mean(input_layer, [2, 3], name='spatial_avg')

    def fully_connected(self, input_layer, num_outputs, activation=None):
        """Applies a fully-connected set of weights"""
        num_inputs = input_layer.get_shape().as_list()[1]
        kernel_size = [num_inputs, num_outputs]
        with tf.variable_scope(self._count_layer('fully_connected')):
            kernel = self._get_variable('weights', kernel_size,
                                        input_layer.dtype)
            x = tf.matmul(input_layer, kernel)
            x = self._bias(x)
            x = self.activate(x, activation)
            return x

    def inception_module(self, input_layer, name, cols):
        """Applies an inception module with a given form"""
        with tf.name_scope(name):
            col_layers = []
            col_layer_sizes = []
            for c, col in enumerate(cols):
                col_layers.append([])
                col_layer_sizes.append([])
                x = input_layer
                for l, layer in enumerate(col):
                    ltype, args = layer[0], layer[1:]
                    if ltype == 'conv':
                        x = self.conv(x, *args)
                    elif ltype == 'pool':
                        x = self.pool(x, *args)
                    elif ltype == 'share':
                        # Share matching layer from previous column
                        x = col_layers[c - 1][l]
                    else:
                        raise KeyError("Invalid layer type for " +
                                       "inception module: '%s'" % ltype)
                    col_layers[c].append(x)
            catdim = 1
            catvals = [layers[-1] for layers in col_layers]
            x = tf.concat(catvals, catdim)
            return x


def inference_googlenet(net, input_layer):
    """GoogLeNet model
    https://arxiv.org/abs/1409.4842
    """
    net.use_batch_norm = False

    def inception_v1(net, x, k, l, m, n, p, q):
        cols = [[('conv', k, (1, 1))],
                [('conv', l, (1, 1)), ('conv', m, (3, 3))],
                [('conv', n, (1, 1)), ('conv', p, (5, 5))],
                [('pool', 'MAX', (3, 3), (1, 1), 'SAME'), ('conv', q, (1, 1))]]
        return net.inception_module(x, 'incept_v1', cols)

    print('input_layer=', input_layer)
    x = net.input_layer(input_layer)
    print('x=', x)
    x = net.conv(x, 64, (7, 7), (2, 2))
    print('x=', x)
    x = net.pool(x, 'MAX', (3, 3), padding='SAME')
    print('x=', x)
    x = net.conv(x, 64, (1, 1))
    print('x=', x)
    x = net.conv(x, 192, (3, 3))
    print('x=', x)
    x = net.pool(x, 'MAX', (3, 3), padding='SAME')
    print('x=', x)
    x = inception_v1(net, x, 64, 96, 128, 16, 32, 32)
    x = inception_v1(net, x, 128, 128, 192, 32, 96, 64)
    x = net.pool(x, 'MAX', (3, 3), padding='SAME')
    x = inception_v1(net, x, 192, 96, 208, 16, 48, 64)
    x = inception_v1(net, x, 160, 112, 224, 24, 64, 64)
    x = inception_v1(net, x, 128, 128, 256, 24, 64, 64)
    x = inception_v1(net, x, 112, 144, 288, 32, 64, 64)
    x = inception_v1(net, x, 256, 160, 320, 32, 128, 128)
    x = net.pool(x, 'MAX', (3, 3), padding='SAME')
    x = inception_v1(net, x, 256, 160, 320, 32, 128, 128)
    x = inception_v1(net, x, 384, 192, 384, 48, 128, 128)
    x = net.spatial_avg(x)
    return x


def eval_func(images, var_scope):
    net = GPUNetworkBuilder(
        False, dtype=tf.float32, use_xla=False)
    #    images = net._to_nhwc(images)
    #    model_func = inference_googlenet
    print('>>> eval_func: images=', images)
    output = inference_googlenet(net, images)
    logits_g = net.fully_connected(output, 8, activation='LINEAR')
    if logits_g.dtype != tf.float32:
        logits_g = tf.cast(logits_g, tf.float32)
    with tf.device('/cpu:0'):
        logits_g = tf.nn.softmax(logits_g)

    return logits_g


def sess_eval(sess, image, logits_g):
    with tf.Graph().as_default() as g:
        flogits_g = sess.run([logits_g], feed_dict={images: image})
        gender_result = None

    global gender
    #    print('values=',flogits_g, flogits_a)
    gender_result = gender[np.argmax(flogits_g[0])]
    #    print('argmax g=', np.argmax(flogits_g[0]))
    #    print('argmax a=', np.argmax(flogits_a[0]))
    print('gender=', gender_result)
    return gender_result


# graph construction for evaluation
gender = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']
with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    images = tf.placeholder(dtype=tf.uint8, shape=[256, 256, 3])
    checkpoint_dir = './ckpt_data_facial_expression'

    # Build a Graph that computes the logits predictions from the
    # inference model.
    # Build inference Graph.
    # print('>>>>> input original size = ', res_images)
    #   in case of images less than or larger than 227x227
    #   images = tf.image.resize_images(images, [227,227] )
    #   print('>>>>> input resized = ',images)
    with tf.variable_scope('GPU_%i' % 0, reuse=tf.AUTO_REUSE) as var_scope, \
            tf.name_scope('tower_%i' % 0):
        images1 = tf.image.central_crop(images, 224. / 256.)
        images2 = tf.image.resize_images(images1, [224, 224], tf.image.ResizeMethod.BILINEAR, align_corners=False)
        #        res_images = tf.cast(images, dtype=tf.float32)
        #        images3 = tf.image.per_image_standardization(images2)
        res_images = tf.reshape(images2, [1, 224, 224, 3])
        logits_g = eval_func(res_images, var_scope)

        # Restore the moving average version of the learned variables for eval.
    saver = tf.train.Saver()

    gpu_options = tf.GPUOptions(allocator_type='BFC', allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
        print('No checkpoint file found')
        sys.exit()

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print('cv version=', major_ver, minor_ver, subminor_ver)
if __name__ == '__main__':
    #    if __package__ is None:
    #        import sys
    #        from os import path
    #        print(path.dirname( path.dirname( path.abspath(__file__) ) ))
    #        sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
    #        import eval_googleNet

    # Set up tracker.
    # Instead of MIL, you can also use
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MULTI']
    tracker_type = tracker_types[6]

    if int(major_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MULTI':
            tracker = cv2.MultiTracker_create()

    video = cv2.VideoCapture(0)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        video.release()
        cv2.destroyAllWindows()
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        video.release()
        cv2.destroyAllWindows()
        sys.exit()
    print('frame.shape=', frame.shape)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # create a CLAHE object (Arguments are optional).
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # gray = clahe.apply(gray)

    # Define an initial bounding box with face detector
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    # face_cascade = cv2.CascadeClassifier('lpb_cascade.xml')
    # faces = face_cascade.detectMultiScale(frame, scaleFactor= 1.1, minNeighbors=8, minSize=(55, 55), flags=cv2.CASCADE_SCALE_IMAGE)
    print('type of faces = ', type(faces))
    no_faces = len(faces)

    while no_faces < 1:
        ok, frame = video.read()
        # lor(frame, cv2.COLOR_BGR2GRAY)
        # create a CLAHE object (Arguments are optional).
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # gray = clahe.apply(gray)
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        no_faces = len(faces)
        print('no faces = ', no_faces)

    # Initialize multi-tracker with first frame and bounding box
    print('len(faces)=', len(faces))
    bbox = list(faces)
    no_faces = len(faces)
    gender_list = []
    gender_count = [0, 0]
    bbox_list = []


    def check_faces(no_faces, faces, bbox, frame):
        for i in range(no_faces):
            print(faces[i])
            bbox[i] = tuple(faces[i])
            print('bbox[', i, ']=', bbox[i])
            p1 = (bbox[i][0], bbox[i][1])
            p2 = ((bbox[i][0] + bbox[i][2]), (bbox[i][1] + bbox[i][3]))

            image = frame[p1[0]:p2[0], p1[1]:p2[1]]
            print(image.shape)
            if image.shape[0] <= 10:
                break
            if image.shape[1] <= 10:
                break
            print('p1=', p1, 'p2=', p2)
            if image is None:
                break
            image = cv2.resize(image, (256, 256))

            gender = sess_eval(sess, image, logits_g)
            gender_list.append(gender)
            bbox_list.append(bbox[i])
            ok = tracker.add(cv2.TrackerBoosting_create(), frame, bbox[i])


    check_faces(no_faces, faces, bbox, frame)

    frame_count = 0

    while video.isOpened():
        frame_count = (frame_count + 1) % 20
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # gray = clahe.apply(gray)

        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        no_faces = len(faces)

        if no_faces > 0:
            tracker = None
            tracker = cv2.MultiTracker_create()

            bbox = None
            bbox = list(faces)
            gender_list[:] = []
            bbox_list[:] = []

            # Over python3.3 code
            # gender_list.clear()
            # age_list.clear()
            # bbox_list.clear()
            check_faces(no_faces, faces, bbox, frame)

            # print('ok=', ok, 'bbox=', bbox, 'no faces = ', len(bbox))
        #        print('ok=', ok, 'bbox=', bbox, 'no faces = ', no_faces)

        #        if len(bbox) < 1:
        #           print('retry - face detect')
        #           faces = face_cascade.detectMultiScale(frame, 1.3, 5)

        #        rno_faces = len(bbox)
        #        while not ok:
        #            print('not ok')
        #            ok, frame = video.read()
        #            ok, bbox = tracker.update(frame)
        #            rno_faces = len(bbox)
        #            print('ok=', ok, 'bbox=', bbox, 'no faces = ', rno_faces)

        # Calculate Frames per secorinnd (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        i = 0
        # starting with here, treat with multiple face bbox
        while gender_list:
            # Draw bounding box
            # Tracking success
            bbox = bbox_list.pop()
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            #            if not gender_list:
            #                print('no_gender_list, i=',i)
            #                break
            #            if not age_list:
            #                print('no age_list, i=',i)
            #                break
            gender_result = gender_list.pop()
            print('gender_result=', gender_result)
            if gender_result == 'MAN':
                gender_count[1] = gender_count[1] + 1
            else:
                gender_count[0] = gender_count[0] + 1

            cv2.putText(frame, "face" + str(i) + ", result: " + gender_result,(int(bbox[0]), int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            i = i + 1
            # Tracking failure
            # cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, '[WOMEN, MEN]: ' + str(gender_count), (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                    2)

        # Display FPS on frame
        # cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        if gender_count[0] < gender_count[1]:
            cv2.putText(frame, 'Majority: MAN', (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif gender_count[0] > gender_count[1]:
            cv2.putText(frame, 'Majority: WOMAN', (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'Majority: EQUAL', (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display result
        cv2.imshow("Tracking", frame)

        #        gender_count = [0, 0]

        # Exit if ESC pressed
        k = cv2.waitKey(10)
        if k == 27:
            break

video.release()
cv2.destroyAllWindows()
