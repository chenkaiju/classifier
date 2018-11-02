from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# imports
import numpy as np
import tensorflow as tf
import os

import cifar10_data

tf.logging.set_verbosity(tf.logging.INFO)

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm(inputs, is_training):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    #
    return tf.layers.batch_normalization(
        inputs=inputs,
        axis=3,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        training=is_training,
        fused=True
    )


def fix_padding(inputs, kernel_size):
    """Pads the input along the spatial dimensions independently of input size.
    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
        kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                    Should be a positive integer.
        data_format: The input format ('channels_last' or 'channels_first').
    Returns:
        A tensor with the same format as the input with the data either intact
        (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size-1
    pad_begin = pad_total//2
    pad_end = pad_total-pad_begin

    padded_input = tf.pad(
        inputs, [[0, 0], [pad_begin, pad_end], [pad_begin, pad_end], [0, 0]])

    return padded_input


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, name):

    with tf.name_scope(name) as scope:
        if strides > 1:
            inputs = fix_padding(inputs, kernel_size)

        return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                padding=('SAME' if strides == 1 else 'VALID'),
                                use_bias=False,
                                kernel_initializer=tf.variance_scaling_initializer()
                                )

def block_bottleneck_v1(inputs, filters, strides, projection_shortcut, is_training, name):
    with tf.name_scope(name) as scope:
        shortcut = inputs
        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)
            shortcut = batch_norm(shortcut, is_training)

        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=1, strides=1, name='conv1')
        inputs = batch_norm(inputs=inputs, is_training=is_training)
        inputs = tf.nn.relu(features=inputs)

        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides, name='conv2')
        inputs = batch_norm(inputs=inputs, is_training=is_training)
        inputs = tf.nn.relu(features=inputs)

        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=4*filters, kernel_size=1, strides=1, name='conv3')
        inputs = batch_norm(inputs=inputs, is_training=is_training)
        inputs = tf.add(inputs, shortcut)
        inputs = tf.nn.relu(features=inputs)

        return inputs

def block_v1(inputs, filters, strides, projection_shortcut, is_training, name):
    """A single block for ResNet v1, without a bottleneck.
    Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
        filters: The number of filters for the convolutions.
        training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
        projection_shortcut: The function to use for projection shortcuts
        (typically a 1x1 convolution when downsampling the input).
        strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
        data_format: The input format ('channels_last' or 'channels_first').
    Returns:
        The output tensor of the block; shape should match inputs.
    """
    with tf.name_scope(name) as scope:
        shortcut = inputs
        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)
            shortcut = batch_norm(shortcut, is_training)

        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides, name='conv1')
        inputs = batch_norm(inputs=inputs, is_training=is_training)
        inputs = tf.nn.relu(features=inputs)

        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=1, name='conv2')
        inputs = batch_norm(inputs=inputs, is_training=is_training)
        inputs = tf.add(inputs, shortcut)
        inputs = tf.nn.relu(features=inputs)

        return inputs


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides, is_training, name):

    with tf.name_scope(name) as scope:

        filters_out = filters*4 if bottleneck else filters

        def projection_shortcut(inputs):
            return conv2d_fixed_padding(inputs, filters=filters_out, kernel_size=1, strides=strides, name='projection_shortcut')

        inputs = block_fn(inputs, filters, strides,
                        projection_shortcut, is_training, 'block_1')

        for block_id in range(1, blocks):
            inputs = block_fn(inputs, filters, 1, None, is_training, 'block_{}'.format(block_id+1))

        return tf.identity(inputs, name)


def resnet_model_fn(features, labels, mode):

    inputs = tf.reshape(features["x"], [-1, 32, 32, 3])

    tf.summary.image('inputs', inputs, max_outputs=10)

    is_training = True if (mode == tf.estimator.ModeKeys.TRAIN) else False
    num_filters = 64

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=num_filters, kernel_size=7, strides=2, name='initial_conv')
    inputs = tf.identity(inputs, 'initial_conv')

    inputs = batch_norm(inputs, is_training)
    inputs = tf.nn.relu(inputs)

    inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=3, strides=2, padding='SAME'
    )

    inputs = tf.identity(inputs, 'initial_max_pool')

    model_collection = {
        '18': [2, 2, 2, 2],
        '34': [3, 4, 6, 3],
        '50': [3, 4, 6, 3],
        '101': [3, 4, 6, 3],
        '152': [3, 8, 36, 3]
    }

    model_type = '101'
    model_layers=model_collection[model_type]

    block_type = block_bottleneck_v1
    bottleneck = True
    if model_type=='18' or model_type=='34':
        block_type = block_v1
        bottleneck = False

    layer_num_filters = num_filters
    for i, num_blocks in enumerate(model_layers):
        layer_num_filters = num_filters * (2**i)
        inputs = block_layer(inputs=inputs,
                                filters=layer_num_filters,
                                bottleneck=bottleneck,
                                block_fn=block_type,
                                blocks=num_blocks,
                                strides=2,
                                is_training=is_training,
                                name='block_layer{}'.format(i + 1)
                                )

    inputs = tf.layers.average_pooling2d(
        inputs=inputs,
        pool_size=1,
        strides=1,
        padding='valid',
        data_format='channels_last',
        name='average_pool'
    )

    inputs = tf.squeeze(inputs, axis=[1, 2])
    logits = tf.layers.dense(inputs=inputs, units=10, name='last_dense')

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    tf.summary.scalar('cross_entropy', loss)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(
            labels=labels, 
            predictions=predictions["classes"],
            name='acc_op')
    metrics = {"accuracy": accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    # evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)

    # training mode
    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def main(unused_argv):

    root_path = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(root_path, 'cifar10-data')
    cifar10_data.maybe_download_and_extract(data_folder)
    data_folder = os.path.join(data_folder, 'cifar-10-batches-py')

    cifar10 = cifar10_data.load_dataset(data_folder)
    train_data = cifar10["images_train"]  # Returns np.array
    train_labels = np.asarray(cifar10["labels_train"], dtype=np.int32)
    eval_data = cifar10["images_test"]  # Returns np.array
    eval_labels = np.asarray(cifar10["labels_test"], dtype=np.int32)


# #     root_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root_path, 'resnet_cifar10')
#     print('model path: ', model_path)
    # Create the Estimator
    cifar10_classifier = tf.estimator.Estimator(model_fn=resnet_model_fn, model_dir=model_path)

#   # Set up logging for predictions
#   # Log the values in the "Softmax" tensor with label "probabilities"
#     #tensors_to_log = {"probabilities": "softmax_tensor"}
#     # logging_hook = tf.train.LoggingTensorHook(
#     #     tensors=tensors_to_log, every_n_iter=50)

#     log_path = os.path.join(model_path, 'log')
#     print('log path: ', log_path)

#     # summary_hook = tf.train.SummarySaverHook(
#     #     save_steps=50,
#     #     output_dir=log_path,
#     #     scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all())
#     # )

    #Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    cifar10_classifier.train(input_fn=train_input_fn, steps=20000)

  # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = cifar10_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
