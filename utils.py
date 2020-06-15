

import torch
import torch.nn as nn

import time
import numpy as np
import threading


### COMMON FUNCTIONS ###    
def _rgb2ycbcr(img, maxVal=255):
#    r = img[:,:,0]
#    g = img[:,:,1]
#    b = img[:,:,2]

    O = np.array([[16],
                  [128],
                  [128]])
    T = np.array([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                  [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                  [0.439215686274510, -0.367788235294118, -0.071427450980392]])

#    ycbcr = np.empty([img.shape[0], img.shape[1], img.shape[2]])

    if maxVal == 1:
        O = O / 255.0

#    ycbcr[:,:,0] = ((T[0,0] * r) + (T[0,1] * g) + (T[0,2] * b) + O[0])
#    ycbcr[:,:,1] = ((T[1,0] * r) + (T[1,1] * g) + (T[1,2] * b) + O[1])
#    ycbcr[:,:,2] = ((T[2,0] * r) + (T[2,1] * g) + (T[2,2] * b) + O[2])

    t = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
    t = np.dot(t, np.transpose(T))
    t[:, 0] += O[0]
    t[:, 1] += O[1]
    t[:, 2] += O[2]
    ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])

#    print(np.all((ycbcr - ycbcr_) < 1/255.0/2.0))

    return ycbcr


def _load_img_array(path, color_mode='RGB', channel_mean=None, modcrop=[0,0,0,0]):
    '''Load an image using PIL and convert it into specified color space,
    and return it as an numpy array.

    https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
    The code is modified from Keras.preprocessing.image.load_img, img_to_array.
    '''
    ## Load image
    from PIL import Image
    img = Image.open(path)
    if color_mode == 'RGB':
        cimg = img.convert('RGB')
        x = np.asarray(cimg, dtype='float32')

    elif color_mode == 'YCbCr' or color_mode == 'Y':
        cimg = img.convert('YCbCr')
        x = np.asarray(cimg, dtype='float32')
        if color_mode == 'Y':
            x = x[:,:,0:1]

    ## To 0-1
    x *= 1.0/255.0

    if channel_mean:
        x[:,:,0] -= channel_mean[0]
        x[:,:,1] -= channel_mean[1]
        x[:,:,2] -= channel_mean[2]

    if modcrop[0]*modcrop[1]*modcrop[2]*modcrop[3]:
        x = x[modcrop[0]:-modcrop[1], modcrop[2]:-modcrop[3], :]

    return x

    
def PSNR(y_true, y_pred, shave_border=4):
    '''
        Input must be 0-255, 2D
    '''

    target_data = np.array(y_true, dtype=np.float32)
    ref_data = np.array(y_pred, dtype=np.float32)

    diff = ref_data - target_data
    if shave_border > 0:
        diff = diff[shave_border:-shave_border, shave_border:-shave_border]
    rmse = np.sqrt(np.mean(np.power(diff, 2)))

    return 20 * np.log10(255./rmse)


# MATLAB imresize function
# Key difference from other resize funtions is antialiasing when downsampling
# This function only for downsampling
def DownSample2DMatlab(tensor, scale, method='cubic', antialiasing=True, cuda=True):
    '''
    This gives same result as MATLAB downsampling
    tensor: 4D tensor [Batch, Channel, Height, Width],
            height and width must be divided by the denominator of scale factor
    scale: Even integer denominator scale factor only (e.g. 1/2,1/4,1/8,...)
           Or list [1/2, 1/4] : [V scale, H scale]
    method: 'cubic' as default, currently cubic supported
    antialiasing: True as default
    '''

    # For cubic interpolation,
    # Cubic Convolution Interpolation for Digital Image Processing, ASSP, 1981
    def cubic(x):
        absx = np.abs(x)
        absx2 = np.multiply(absx, absx)
        absx3 = np.multiply(absx2, absx)

        f = np.multiply((1.5*absx3 - 2.5*absx2 + 1), np.less_equal(absx, 1)) + \
            np.multiply((-0.5*absx3 + 2.5*absx2 - 4*absx + 2), \
            np.logical_and(np.less(1, absx), np.less_equal(absx, 2)))

        return f

    # Generate resize kernel (resize weight computation)
    def contributions(scale, kernel, kernel_width, antialiasing):
        if scale < 1 and antialiasing:
          kernel_width = kernel_width / scale

        x = np.ones((1, 1))

        u = x/scale + 0.5 * (1 - 1/scale)

        left = np.floor(u - kernel_width/2)

        P = int(np.ceil(kernel_width) + 2)

        indices = np.tile(left, (1, P)) + np.expand_dims(np.arange(0, P), 0)

        if scale < 1 and antialiasing:
          weights = scale * kernel(scale * (np.tile(u, (1, P)) - indices))
        else:
          weights = kernel(np.tile(u, (1, P)) - indices)

        weights = weights / np.expand_dims(np.sum(weights, 1), 1)

        save = np.where(np.any(weights, 0))
        weights = weights[:, save[0]]

        return weights

    # Resize along a specified dimension
    def resizeAlongDim(tensor, scale_v, scale_h, kernel_width, weights):#, indices):
        if scale_v < 1 and antialiasing:
           kernel_width_v = kernel_width / scale_v
        else:
           kernel_width_v = kernel_width
        if scale_h < 1 and antialiasing:
           kernel_width_h = kernel_width / scale_h
        else:
           kernel_width_h = kernel_width

        # Generate filter
        f_height = np.transpose(weights[0][0:1, :])
        f_width = weights[1][0:1, :]
        f = np.dot(f_height, f_width)
        f = f[np.newaxis, np.newaxis, :, :]
        F = torch.from_numpy(f.astype('float32'))

        # Reflect padding
        i_scale_v = int(1/scale_v)
        i_scale_h = int(1/scale_h)
        pad_top = int((kernel_width_v - i_scale_v) / 2)
        if i_scale_v == 1:
            pad_top = 0
        pad_bottom = int((kernel_width_h - i_scale_h) / 2)
        if i_scale_h == 1:
            pad_bottom = 0
        pad_array = ([pad_bottom, pad_bottom, pad_top, pad_top])
        kernel_width_v = int(kernel_width_v)
        kernel_width_h = int(kernel_width_h)

        #
        tensor_shape = tensor.size()
        num_channel = tensor_shape[1]
        FT = nn.Conv2d(1, 1, (kernel_width_v, kernel_width_h), (i_scale_v, i_scale_h), bias=False)
        FT.weight.data = F
        if cuda:
           FT.cuda()
        FT.requires_grad = False

        # actually, we want 'symmetric' padding, not 'reflect'
        outs = []
        for c in range(num_channel):
            padded = nn.functional.pad(tensor[:,c:c+1,:,:], pad_array, 'reflect')
            outs.append(FT(padded))
        out = torch.cat(outs, 1)

        return out

    if method == 'cubic':
        kernel = cubic

    kernel_width = 4

    if type(scale) is list:
        scale_v = float(scale[0])
        scale_h = float(scale[1])

        weights = []
        for i in range(2):
            W = contributions(float(scale[i]), kernel, kernel_width, antialiasing)
            weights.append(W)
    else:
        scale = float(scale)

        scale_v = scale
        scale_h = scale

        weights = []
        for i in range(2):
            W = contributions(scale, kernel, kernel_width, antialiasing)
            weights.append(W)

    # np.save('bic_x4_downsample_h.npy', weights[0])

    tensor = resizeAlongDim(tensor, scale_v, scale_h, kernel_width, weights)

    return tensor


def Huber(input, target, delta=0.01, reduce=True):
    abs_error = torch.abs(input - target)
    quadratic = torch.clamp(abs_error, max=delta)

    # The following expression is the same in value as
    # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
    # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
    # This is necessary to avoid doubling the gradient, since there is already a
    # nonzero contribution to the gradient from the quadratic term.
    linear = (abs_error - quadratic)
    losses = 0.5 * torch.pow(quadratic, 2) + delta * linear
    
    if reduce:
        return torch.mean(losses)
    else:
        return losses


def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))



### DATASET HANDLING ###    
class GeneratorEnqueuer(object):
    """Builds a queue out of a data generator.
    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.
    # Arguments
        generator: a generator function which endlessly yields data
        pickle_safe: use multiprocessing if True, otherwise threading

    **copied from https://github.com/fchollet/keras/blob/master/keras/engine/training.py

    Usage:
    enqueuer = GeneratorEnqueuer(generator, pickle_safe=pickle_safe)
    enqueuer.start(max_q_size=max_q_size, workers=workers)

    while enqueuer.is_running():
        if not enqueuer.queue.empty():
            generator_output = enqueuer.queue.get()
            break
        else:
            time.sleep(wait_time)
    """

    def __init__(self, generator, use_multiprocessing=True, wait_time=0.00001, random_seed=int(time.time())):
        self.wait_time = wait_time
        self._generator = generator
        self._use_multiprocessing = use_multiprocessing
        self._threads = []
        self._stop_event = None
        self.queue = None
        self.random_seed = random_seed



    def start(self, workers=1, max_q_size=10):
        """Kicks off threads which add data from the generator into the queue.
        # Arguments
            workers: number of worker threads
            max_q_size: queue size (when full, threads could block on put())
            wait_time: time to sleep in-between calls to put()
        """

        def data_generator_task():
            while not self._stop_event.is_set():
                try:
                    if self._use_multiprocessing or self.queue.qsize() < max_q_size:
                        generator_output = next(self._generator)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(self.wait_time)
                except Exception:
                    self._stop_event.set()
                    raise

        try:
            import multiprocessing
            try:
                import queue
            except ImportError:
                import Queue as queue

            if self._use_multiprocessing:
                self.queue = multiprocessing.Queue(maxsize=max_q_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = queue.Queue()
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._use_multiprocessing:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed(self.random_seed)
                    thread = multiprocessing.Process(target=data_generator_task)
                    thread.daemon = True
                    if self.random_seed is not None:
                        self.random_seed += 1
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stop running threads and wait for them to exit, if necessary.
        Should be called by the same thread which called start().
        # Arguments
            timeout: maximum time to wait on thread.join()
        """
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                if self._use_multiprocessing:
                    thread.terminate()
                else:
                    thread.join(timeout)

        if self._use_multiprocessing:
            if self.queue is not None:
                self.queue.close()

        self._threads = []
        self._stop_event = None
        self.queue = None


    def dequeue(self):
        while self.is_running():
            if not self.queue.empty():
                return self.queue.get()
                break
            else:
                time.sleep(self.wait_time)



#################################################################
### Batch Iterators #############################################
#################################################################
class Iterator(object):
    '''
    https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
    '''
    def __init__(self, N, batch_size, shuffle, seed, infinite):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(N, batch_size, shuffle, seed, infinite)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None, infinite=True):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(N)
                if shuffle:
                    index_array = np.random.permutation(N)

            if infinite == True:
                current_index = (self.batch_index * batch_size) % N
                if N >= current_index + batch_size:
                    current_batch_size = batch_size
                    self.batch_index += 1
                else:
                    current_batch_size = N - current_index
                    self.batch_index = 0
            else:
                current_index = (self.batch_index * batch_size)
                if current_index >= N:
                    self.batch_index = 0
                    raise StopIteration()
                elif N >= current_index + batch_size:
                    current_batch_size = batch_size
                else:
                    current_batch_size = N - current_index
                self.batch_index += 1
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


def _flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

class DirectoryIterator_NTIRE2020(Iterator):
  def __init__(self,
               train_dir = './train/',
               out_batch_size = 16,
               crop_size= 64,
               scale_factor=4,
               crop_per_frame=4,
               shuffle = True,
               seed = None,
               infinite = True):

    self.train_dir = train_dir
    self.crop_size = crop_size
    self.crop_per_frame = crop_per_frame
    self.out_batch_size = out_batch_size
    self.r = scale_factor

    import glob
    import random


    gt_files = glob.glob(train_dir+'/*.png')
    gt_files.sort()

    if shuffle:
        def shuffle_list(*ls):
            l = list(zip(*ls))
            random.shuffle(l)
            return zip(*l)

        # blur_pngs, sharp_pngs = shuffle_list(blur_pngs, sharp_pngs)
        random.shuffle(gt_files)
        
    self.total_count = len(gt_files)
    self.gt_files = gt_files

    print('Found %d trainnig images' % self.total_count)

    super(DirectoryIterator_NTIRE2020, self).__init__(self.total_count, int(out_batch_size/crop_per_frame), shuffle, seed, infinite)

  def next(self):
    with self.lock:
        index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_sharp = []
    
    i = 0
    while len(batch_sharp) < int(self.out_batch_size):
        sharps = self.gt_files[(current_index+i) % self.total_count]
        i += 1

        try:
            S_ = _load_img_array(sharps) # H,W,C
        except:
            print("File open error: {}".format(sharps))
            continue

        ss = S_.shape
        sh = np.random.randint(0, ss[0]-self.crop_size+1, self.crop_per_frame)
        sw = np.random.randint(0, ss[1]-self.crop_size+1, self.crop_per_frame)

        for j in range(self.crop_per_frame):
            S = S_[sh[j]:(sh[j]+self.crop_size), sw[j]:(sw[j]+self.crop_size)]

            # Random Aug
            # Rot
            ri = np.random.randint(0,4)
            S = np.rot90(S, ri)

            # LR flip
            if np.random.random() < 0.5:
                S = _flip_axis(S, 1)

            batch_sharp.append(S)

    batch_sharp = np.stack(batch_sharp, 0).astype(np.float32).transpose((0,3,1,2))   # B, C, H, W

    return  batch_sharp