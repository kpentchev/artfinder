import tensorflow as tf # tensorflow module
import numpy as np # numpy module
import os # path join
from pathlib import Path
import time
from cnn import CNN
from image import image


DATA_DIR = "/Users/kpentchev/artmimir/train_data_tfrecords/"
MODEL_DIR = "/Users/kpentchev/artmimir/models/cnn2/"
TRAINING_SET_SIZE = 9009
BATCH_SIZE = 15
N_CLASSES = 3
IMAGE_SIZE = 384


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features = {
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/class/label": tf.FixedLenFeature([], tf.int64),})
    image_encoded = features["image/encoded"]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=3)
    image_object = image()
    image_object.image = tf.image.resize_image_with_crop_or_pad(image_raw, IMAGE_SIZE, IMAGE_SIZE)
    image_object.height = features["image/height"]
    image_object.width = features["image/width"]
    image_object.filename = features["image/filename"]
    image_object.label = tf.cast(features["image/class/label"], tf.int64)
    return image_object

def artmimir_input(if_random = True, if_training = True):
    if(if_training):
        filenames = [os.path.join(DATA_DIR, "train-0000%d-of-00004" % i) for i in range(0, 1)]
    else:
        filenames = [os.path.join(DATA_DIR, "validation-0000%d-of-00004" % i) for i in range(0, 1)]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError("Failed to find file: %s" % f)
    filename_queue = tf.train.string_input_producer(filenames)
    image_object = read_and_decode(filename_queue)
    image = tf.image.per_image_standardization(image_object.image)
#    image = image_object.image
#    image = tf.image.adjust_gamma(tf.cast(image_object.image, tf.float32), gamma=1, gain=1) # Scale image to (0, 1)
    label = image_object.label
    filename = image_object.filename

    if(if_random):
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(TRAINING_SET_SIZE * min_fraction_of_examples_in_queue)
        print("Filling queue with %d images before starting to train. " "This will take a few minutes." % min_queue_examples)
        num_preprocess_threads = 1
        image_batch, label_batch, filename_batch = tf.train.shuffle_batch(
            [image, label, filename],
            batch_size = BATCH_SIZE,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples + 3 * BATCH_SIZE,
            min_after_dequeue = min_queue_examples)
        return image_batch, label_batch, filename_batch
    else:
        image_batch, label_batch, filename_batch = tf.train.batch(
            [image, label, filename],
            batch_size = BATCH_SIZE,
            num_threads = 1)
        return image_batch, label_batch, filename_batch

def conv_net(x):
    x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    cnn = CNN()
    cnn.input_layer(input=x_image, num_inputs=3, filter_size=3, num_filters=64)
    cnn.conv_layer(num_inputs=64, filter_size=3, num_filters=64)
    cnn.conv_layer(num_inputs=64, filter_size=3, num_filters=128)
    cnn.conv_layer(num_inputs=128, filter_size=3, num_filters=256)
    cnn.flat_layer()
    cnn.fc_layer(num_outputs=2048)
    cnn.output_layer(num_inputs=2048, num_outputs=3)
    
    return cnn.build()


def artmimir_train():
    image_batch_out, label_batch_out, filename_batch = artmimir_input(if_random = False, if_training = True)

    image_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    image_batch = tf.reshape(image_batch_out, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

    label_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_CLASSES])
    label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="label_batch_offset")
    label_batch_one_hot = tf.one_hot(tf.add(label_batch_out, label_offset), depth=3, on_value=1.0, off_value=0.0)

    cnn = conv_net(x=image_batch_placeholder)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=cnn, labels=label_batch_placeholder)
    #print("labels: %s" % label_batch_placeholder)
    #print("logits: %s" % logits_out)
    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if Path(MODEL_DIR + "checkpoint-train.ckpt").is_file():
            saver.restore(sess, MODEL_DIR + "checkpoint-train.ckpt")
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)

        n_samples = int(TRAINING_SET_SIZE  * 1)

        t_start = time.perf_counter()
        for i in range(int(n_samples / BATCH_SIZE + 1)):
            
            image_out, label_out, label_batch_one_hot_out, filename_out = sess.run([image_batch, label_batch_out, label_batch_one_hot, filename_batch])

            sess.run(optimizer, feed_dict={image_batch_placeholder: image_out, label_batch_placeholder: label_batch_one_hot_out})

            t_stop = time.perf_counter()
            perf = BATCH_SIZE / (t_stop - t_start)
            print("Done processing %s / %s images. Speed is %s img/sec" % ((i+1)*BATCH_SIZE, n_samples, perf))
            t_start = time.perf_counter()

        saver.save(sess, MODEL_DIR + "checkpoint-train.ckpt")
        coord.request_stop()
        coord.join(threads)
        sess.close()



def artmimir_eval():
    N_SAMPLES_PER_LABEL = 10
    image_batch_out, label_batch_out, filename_batch = artmimir_input(if_random = False, if_training = False)

    image_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    image_batch = tf.reshape(image_batch_out, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

    label_tensor_placeholder = tf.placeholder(tf.int64, shape=[BATCH_SIZE])
    label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="label_batch_offset")
    label_batch = tf.add(label_batch_out, label_offset)

    cnn = conv_net(x=image_batch_placeholder)

    logits_out = tf.reshape(cnn, [BATCH_SIZE, N_CLASSES])
    logits_batch = tf.to_int64(tf.argmax(logits_out, axis=1))

    correct_prediction = tf.equal(logits_batch, label_tensor_placeholder)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, MODEL_DIR + "checkpoint-train.ckpt")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)

        accuracy_accu = 0

        for i in range(N_SAMPLES_PER_LABEL):
            image_out, label_out, filename_out = sess.run([image_batch, label_batch, filename_batch])

            accuracy_out, logits_batch_out = sess.run([accuracy, logits_batch], feed_dict={image_batch_placeholder: image_out, label_tensor_placeholder: label_out})
            accuracy_accu += accuracy_out

            print(i)
            #print(image_out.shape)
            #print("label_out: ")
            print(filename_out)
            print("true %s" % label_out)
            print("eval %s" % logits_batch_out)

        print("Accuracy: ")
        print(accuracy_accu / N_SAMPLES_PER_LABEL)

        coord.request_stop()
        coord.join(threads)
        sess.close()

artmimir_train()
#artmimir_eval()