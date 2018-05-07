import numpy as np
import tensorflow as tf

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import time
import sys
import numpy.random as random
from layers.subpixel import SubPixel1D, SubPixel1D_v2


from scipy import interpolate
from scipy.signal import decimate, spectrogram
from scipy.signal import butter, lfilter

import sys
import librosa
# from keras.layers import add

os.environ["CUDA_VISIBLE_DEVICES"]="1"

# music output


layers = 8
restore = False
batch_size = 32
epoch = 400
lr = 0.0002
Apa = 1.0
b1 = 0.99
b2 = 0.999

#1speaker 225 // iter 19000, batch 32
# md = '../../asrunet_md/sp225/'
md = './'

# parameter

down_sampling_layers = []
up_sampling_layers = []
n_filters = [128, 256, 512, 512, 512, 512, 512, 512]
n_filter_sizes = [65, 33, 17, 9, 9, 9, 9, 9, 9]



def create_path(params):
    path = ''
    for key in params.keys():
        path += key + '=' + str(params[key]) + '/'
    return path

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_biases(shape):
    return tf.Variable(tf.zeros(shape))

def psnr_metric(y_true, y_pred):
    l2_loss = tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)))
    psnr = 20. * tf.log(tf.reduce_max(y_true)/tf.sqrt(l2_loss)+1e-8)/tf.log(10.)
    return psnr

def leaky_relu(x, alpha):
    return tf.nn.relu(x)-alpha*tf.nn.relu(-x)
    
def asrUnet(X, layers=layers):
# input
    x = X

    down_sampling_layers = []
    n_filters = [128, 256, 512, 512, 512, 512, 512, 512]
    n_filter_sizes = [65, 33, 17, 9, 9, 9, 9, 9 ,9]

# downsampling
    for layer, nf, fs in zip(range(layers), n_filters, n_filter_sizes):
        x = tf.layers.conv1d(x, filters=nf ,kernel_size=fs, strides=2, activation=None, padding='same', bias_initializer=tf.truncated_normal_initializer(stddev=0.02)) # cf: fitler, fs:kernel_size
        x = leaky_relu(x, 0.2)
        print ('D-Block: ', x.shape)
        down_sampling_layers.append(x)

# bottleneck
    x = tf.layers.conv1d(x, filters=n_filters[-1], kernel_size=n_filter_sizes[-1], strides=2, activation=None, padding='same', bias_initializer=tf.truncated_normal_initializer(stddev=0.02))
    x = tf.layers.dropout(x, rate=0.5)
    x = leaky_relu(x, 0.2)

# upsampling
    for layer, nf, fs, scl in reversed(list(zip(range(layers), n_filters, n_filter_sizes, down_sampling_layers))):
        x = tf.layers.conv1d(x, filters=nf*2, kernel_size=fs, activation=None, padding='same', bias_initializer=tf.truncated_normal_initializer(stddev=0.02))
        x = tf.layers.dropout(x, rate=0.5)
        x = tf.nn.relu(x)
        # print x.shape
        x = SubPixel1D(x, r=2)
        # print x.shape
        x = tf.concat([x, scl], axis=2)
        print ('U-Block: ', x.shape)

# output
    x = tf.layers.conv1d(x, filters=2, kernel_size=9, activation=None, padding='same', bias_initializer=tf.truncated_normal_initializer(stddev=0.02))
    # print x.shape
    x = SubPixel1D(x, r=2)
    print ("final conv layer shape: ", x.shape)
    # g = add([x,X])
    g = tf.add(x, X)
    print ('final shape: ', g.shape)
    return g


def read_tfrecords(filenames):
    tfrecords_queue = tf.train.string_input_producer(filenames, name='queue')
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecords_queue)

    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                        features={
                            'low': tf.FixedLenFeature([], tf.string),
                            'high': tf.FixedLenFeature([], tf.string),
                            'shape': tf.FixedLenFeature([], tf.string),
                                }, name='features')

    low = tf.decode_raw(tfrecord_features['low'], tf.uint8)
    high = tf.decode_raw(tfrecord_features['high'], tf.uint8)
    shape = tf.decode_raw(tfrecord_features['shape'], tf.int32)
    
    low = tf.reshape(low, shape)
    high = tf.reshape(high, shape)
    return low, high, shape

def get_spectrum(x, n_fft=2048):
    S = librosa.stft(x, n_fft)
    p = np.angle(S)
    S = np.log(np.abs(S)+0.00005)
    return S

def save_spectrum(S, lim=800, outfile='spectrogram.png'):
    plt.imshow(S.T, aspect=10)
    plt.tight_layout()
    plt.savefig()




def main(_):

    ## load dataset(joblib)
    dim = 8192

    # ld = LoadData()
    # Y_val, X_val = ld._load_data(experiments[0]['file_list'], 1, 20, 16000.)
    #ld = joblib.load(SAVE_DIR + 'ld_unet_lp'+str(0))
    # ld = joblib.load(SAVE_DIR + 'ld_unet_Y')
    #val_ld = joblib.load(SAVE_DIR + 'ld_unet_Y')

    def parser(record):
        keys_to_features = {
                            'low': tf.FixedLenFeature([], tf.string),
                            'high': tf.FixedLenFeature([], tf.string),
        }
        parsed = tf.parse_single_example(record, keys_to_features)

    # Perform additional preprocessing on the parsed data.
        # content = tf.cast(parsed['s1'], tf.int32)
        # channel = tf.cast(parsed['s2'], tf.int32)
        # sha = tf.stack([content, channel])
        low = tf.decode_raw(parsed['low'], tf.float32)
        high = tf.decode_raw(parsed['high'], tf.float32)
        #Y = tf.reshape(high, sha)
        return low, high

    # filenames = ['train0.tfrecords','train1.tfrecords','train2.tfrecords',\
    #              'train3.tfrecords','train4.tfrecords','train5.tfrecords']
    # valfilenames =['valid0.tfrecords']
    filenames = ['./train_jazz_32k.tfrecords']
    valfilenames =['./valid_jazz_32k.tfrecords']
    tfrecord ='./train_jazz_32k.tfrecords'


    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=1000+3*50)
    
    dataset = dataset.repeat(epoch)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    lowt, hight = iterator.get_next()

    datasetv = tf.data.TFRecordDataset(valfilenames)
    datasetv = datasetv.map(parser)

    datasetv = datasetv.repeat(epoch*2000)
    datasetv = datasetv.batch(32)

    iteratorv = datasetv.make_one_shot_iterator()
    lowv, highv = iteratorv.get_next()

    is_valid = tf.placeholder(dtype=bool, shape=())
    low, high = tf.cond(is_valid, lambda: [lowv, highv], lambda: [lowt, hight])

    # save inputs
    X, Y = tf.reshape(low, [-1, dim, 1]), tf.reshape(high, [-1 ,dim, 1])
    inputs = (X, Y)
    tf.add_to_collection('inputs', X)
    tf.add_to_collection('inputs', Y)
    # tf.add_to_collection('inputs', alpha)

    # create model outputs
    pred_y = asrUnet(X, layers=8)
    tf.add_to_collection('preds', pred_y)

    # create training updates
    
    ## loss define
    sqrt_l2_loss = tf.sqrt(tf.reduce_mean((pred_y-Y)**2 + 1e-6, axis=[1,2]))
    sqrt_l2_norm = tf.sqrt(tf.reduce_mean(Y**2, axis=[1,2]))
    snr = 20 * tf.log(sqrt_l2_norm / sqrt_l2_loss + 1e-8)/tf.log(10.)

    avg_sqrt_l2_loss = tf.reduce_mean(sqrt_l2_loss, axis=0)
    avg_snr = tf.reduce_mean(snr, axis=0)

    ## track losses
    tf.summary.scalar('l2_loss', avg_sqrt_l2_loss)
    tf.summary.scalar('snr', avg_snr)

    ## save loss
    tf.add_to_collection('losses', avg_sqrt_l2_loss)
    tf.add_to_collection('losses', avg_snr)

    loss_f = avg_sqrt_l2_loss

    ## define params
    params = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'soundnet' not in v.name]

    ## lr
    #lr_ = tf.train.exponential_decay(lr, 350000, 1000, 0.85)

    ## optimizer
    optmizer = tf.train.AdamOptimizer(lr, b1, b2)
    # optm = tf.train.AdamOptimizer(lr, b1, b2).minimize(loss_f)

    ## compute grads
    grads = optmizer.compute_gradients(loss_f, params)
    grads, _ = zip(*grads)

    ## grads update
    with tf.name_scope('optimizer'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        grads = [1.0*g for g in grads]
        gv = zip(grads, params)
        train_op = optmizer.apply_gradients(gv, global_step=global_step)

    ## initialize optimizer variables
    optimizer_vars = [v for v in tf.global_variables() if 'optimizer/' in v.name]
    init = tf.variables_initializer(optimizer_vars)

    tf.add_to_collection('train_op', train_op)    


    # # saver
    saver = tf.train.Saver()

    # restorer
    print (" Reading checkpoints...")
    parameters = []
    for v in tf.trainable_variables():
        parameters.append(v)

    print ("Generator variables : {}".format(parameters))

    restorer = tf.train.Saver(parameters)
    checkpoint_path = tf.train.latest_checkpoint(md)
    
    NUM_THREADS = 1
    config = tf.ConfigProto(inter_op_parallelism_threads=NUM_THREADS,\
                            intra_op_parallelism_threads=NUM_THREADS,\
                            allow_soft_placement=True,\
                            device_count = {'CPU': 1},\
                            )
    counter = 0
    num_examples = 0
    for record in tf.python_io.tf_record_iterator(tfrecord):
        num_examples += 1
    print ('total num of patches in tfrecords : ' + str(num_examples))
    num_batches = num_examples / batch_size
    print ('batches per epoch: ' + str(num_batches))

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        # sess.run(init)
        # tf.initialize_all_variables().run()

        coord = tf.train.Coordinator() # Set Coordinator to Manage Queue Runners
        threads = tf.train.start_queue_runners(sess, coord=coord) # Set Threads


        # if restore == True:
        #     restorer.restore(sess, checkpoint_path)


        try:

            for i in range(350000):
                if coord.should_stop():
                    break

                _, loss = sess.run([train_op, loss_f], feed_dict={is_valid: False})
                print (i, '_th_loss: ', loss)

                if i % 1000 == 0 and i > 0:
                    snr, loss = sess.run([avg_snr, loss_f], feed_dict={is_valid: True})
                    print (i, '_th snr: ', snr, 'loss: ', loss)

                if i % 200 == 0:
                    # store spectrogram
                    low_, high_, x_pr = sess.run([low, high, pred_y], feed_dict={is_valid: False})
                    # print x_pr.shape, low[:len(x_pr)].shape, Y_batch[:len(x_pr)].shape

                    x_pr = x_pr.flatten()
                    Sl = get_spectrum(low_[:len(x_pr)].flatten(), n_fft=2048)
                    Sh = get_spectrum(high_[:len(x_pr)].flatten(), n_fft=2048)
                    Sp = get_spectrum(x_pr, n_fft=2048)
                    print (Sl.shape, Sh.shape, Sp.shape)
                    S = np.concatenate((Sl.reshape(Sh.shape[0], Sh.shape[1]), Sh, Sp), axis=1)
                    fig = Figure(figsize=S.shape[::-1], dpi=1, frameon=False)
                    print (i, 'th')
                    canvas = FigureCanvas(fig)
                    fig.figimage(S, cmap='jet')
                    fig.savefig('./spec/train/' + 'batch_index' + str(i) + '-th_pr.png')
                    #plt.savefig(OUTPUT_DIR + 'spec/train/' + 'batch_index' + str(i) + '-th.pr.png')

                if i % 200 == 0:
                    low_, high_, y_pr = sess.run([low, high, pred_y], feed_dict={is_valid: True})
                    y_pr = y_pr.flatten()
                    Sl = get_spectrum(low_[:len(y_pr)].flatten(), n_fft=2048)
                    Sh = get_spectrum(high_[:len(y_pr)].flatten(), n_fft=2048)
                    Sp = get_spectrum(y_pr, n_fft=2048)
                    S = np.concatenate((Sl.reshape(Sh.shape[0], Sh.shape[1]), Sh, Sp), axis=1)
                    fig = Figure(figsize=S.shape[::-1], dpi=1, frameon=False)
                    print (i, 'th valid')
                    canvas = FigureCanvas(fig)
                    fig.figimage(S, cmap='jet')
                    fig.savefig('./spec/valid/' + 'batch_index' + str(i) + '-th_pr.png')

                if i % 1000 == 0:
                    librosa.output.write_wav('./wav/' + 'batch_index' + str(i) + '-th_ypr.wav', y_pr,
                                             16000)
                    librosa.output.write_wav('./wav/' + 'batch_index' + str(i) + '-th_ylr.wav',
                                             low_.flatten(), 16000)
                    librosa.output.write_wav('./wav/' + 'batch_index' + str(i) + '-th_yhr.wav',
                                             high_.flatten(), 16000)

                if i % 2000 == 0 and i > 0:
                    save_path = saver.save(sess, md + str(i) +'_asr_unet_model.ckpt')              


        except Exception as e:
            coord.request_stop(e)

        except tf.errors.OutOfRangeError():
            print ('end of dataset')

        finally:
            coord.request_stop()
            coord.join(threads)
       




if __name__ == "__main__":
    tf.app.run()


