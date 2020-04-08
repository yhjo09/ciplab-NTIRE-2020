

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, Function
from torch.utils.data import DataLoader

import PIL
from PIL import Image

import numpy as np
import time
from os import listdir, mkdir
from os.path import isfile, join, isdir
import math
from tqdm import tqdm
import glob
import sys

import threading

def PSNR(y_true,y_pred, shave_border=4):
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
               train_dir = '/host/media/ssd1/users/yhjo/dataset/NTIRE2020/train/',
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

    print('Found %d trainnig samples' % self.total_count)

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






# LPIPS
import LPIPS.models.dist_model as dm

model_LPIPS = dm.DistModel()
model_LPIPS.initialize(model='net-lin',net='alex',use_gpu=True)

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
# def im2tensor(image, imtype=np.uint8, cent=1., factor=1.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def load_image(path):
    if(path[-3:] == 'dng'):
        import rawpy
        with rawpy.imread(path) as raw:
            img = raw.postprocess()
        # img = plt.imread(path)
    elif(path[-3:]=='bmp' or path[-3:]=='jpg' or path[-3:]=='png'):
        import cv2
        return cv2.imread(path)[:,:,::-1]
    else:
        img = (255*plt.imread(path)[:,:,:3]).astype('uint8')

    return img







## ESRGAN
import RRDBNet_arch as arch
model_ESRGAN = arch.RRDBNetx4x4(3, 3, 64, 23, gc=32).cuda()




# Residual block for the discriminator
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=nn.Conv2d, which_bn=nn.BatchNorm2d, wide=True,
                preactivation=True, activation=nn.LeakyReLU(0.1, inplace=False), downsample=nn.AvgPool2d(2, stride=2)):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample
            
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels, kernel_size=3, padding=1)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels, 
                                            kernel_size=1, padding=0)

        self.bn1 = self.which_bn(self.hidden_channels)
        self.bn2 = self.which_bn(out_channels)

    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x
        
    def forward(self, x):
        if self.preactivation:
            # h = self.activation(x) # NOT TODAY SATAN
            # Andy's note: This line *must* be an out-of-place ReLU or it 
            #              will negatively affect the shortcut connection.
            h = self.activation(x)
        else:
            h = x    
        h = self.bn1(self.conv1(h))
        # h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)     
            
        return h #+ self.shortcut(x)
        

class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                which_conv=nn.Conv2d, which_bn=nn.BatchNorm2d, activation=nn.LeakyReLU(0.1, inplace=False), 
                upsample=nn.Upsample(scale_factor=2, mode='nearest')):
        super(GBlock, self).__init__()
        
        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels, 
                                            kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(out_channels)
        self.bn2 = self.which_bn(out_channels)
        # upsample layers
        self.upsample = upsample

    def forward(self, x):
        h = self.activation(x)
        if self.upsample:
            h = self.upsample(h)
            # x = self.upsample(x)
        h = self.bn1(self.conv1(h))
        # h = self.activation(self.bn2(h))
        # h = self.conv2(h)
        # if self.learnable_sc:       
        #     x = self.conv_sc(x)
        return h #+ x


        
class UnetD(torch.nn.Module):
    def __init__(self):
        super(UnetD, self).__init__()

        self.enc_b1 = DBlock(3, 64, preactivation=False)
        self.enc_b2 = DBlock(64, 128)
        self.enc_b3 = DBlock(128, 192)
        self.enc_b4 = DBlock(192, 256)
        self.enc_b5 = DBlock(256, 320)
        self.enc_b6 = DBlock(320, 384)

        self.enc_out = nn.Conv2d(384, 1, kernel_size=1, padding=0)

        self.dec_b1 = GBlock(384, 320)
        self.dec_b2 = GBlock(320*2, 256)
        self.dec_b3 = GBlock(256*2, 192)
        self.dec_b4 = GBlock(192*2, 128)
        self.dec_b5 = GBlock(128*2, 64)
        self.dec_b6 = GBlock(64*2, 32)

        self.dec_out = nn.Conv2d(32, 1, kernel_size=1, padding=0)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                # print(classname)
                nn.init.kaiming_normal(m.weight)
                nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


    def forward(self, x):
        e1 = self.enc_b1(x)
        e2 = self.enc_b2(e1)
        e3 = self.enc_b3(e2)
        e4 = self.enc_b4(e3)
        e5 = self.enc_b5(e4)
        e6 = self.enc_b6(e5)

        e_out = self.enc_out(F.leaky_relu(e6, 0.1))
        # print(e1.size())
        # print(e2.size())
        # print(e3.size())
        # print(e4.size())
        # print(e5.size())
        # print(e6.size())

        d1 = self.dec_b1(e6)
        d2 = self.dec_b2(torch.cat([d1, e5], 1))
        d3 = self.dec_b3(torch.cat([d2, e4], 1))
        d4 = self.dec_b4(torch.cat([d3, e3], 1))
        d5 = self.dec_b5(torch.cat([d4, e2], 1))
        d6 = self.dec_b6(torch.cat([d5, e1], 1))

        d_out = self.dec_out(F.leaky_relu(d6, 0.1))

        return e_out, d_out, [e1,e2,e3,e4,e5,e6], [d1,d2,d3,d4,d5,d6]

model_D = UnetD().cuda()







# USER PARAMS
EXP_NAME = "PESR"
VERSION = "1"

UPSCALE = 16     # upscaling factor
NB_BATCH = 2   # 1*4
NB_CROP_FRAME = 2
NB_ITER = 200000    # Total number of iterations

I_DISPLAY = 100
I_VALIDATION = 100
I_SAVE = 1000

best_avg_psnr = 26    # DUF-16: 26.8, FRVSR-10-128: 26.9, RLSP-7-128: 27.46
best_avg_lpips = 0.15



# model_ESRGAN.load_state_dict(torch.load('./model.pth').state_dict(), strict=True)

TRAIN_DIR = '../../../../../../../ssd1/users/yhjo/dataset/NTIRE2020/train/'
VAL_DIR = '../../../../../../../ssd1/users/yhjo/dataset/NTIRE2020/val/'





# from tensorboardX import SummaryWriter
# writer = SummaryWriter(log_dir='../pt_log/{}/v{}'.format(EXP_NAME, str(VERSION)))




# Iteration
print('===> Training start')
l_accum = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
l_accum_n = 0.
dT = 0.
rT = 0.





if not isdir('{}'.format(EXP_NAME)):
    mkdir('{}'.format(EXP_NAME))
if not isdir('{}/checkpoint'.format(EXP_NAME)):
    mkdir('{}/checkpoint'.format(EXP_NAME))
if not isdir('{}/result'.format(EXP_NAME)):
    mkdir('{}/result'.format(EXP_NAME))
if not isdir('{}/checkpoint/v{}'.format(EXP_NAME, str(VERSION))):
    mkdir('{}/checkpoint/v{}'.format(EXP_NAME, str(VERSION)))
if not isdir('{}/result/v{}'.format(EXP_NAME, str(VERSION))):
    mkdir('{}/result/v{}'.format(EXP_NAME, str(VERSION)))




params_G = list(filter(lambda p: p.requires_grad, model_ESRGAN.parameters()))
opt_G = optim.Adam(params_G, lr=0.00001)

params_D = list(filter(lambda p: p.requires_grad, model_D.parameters()))
opt_D = optim.Adam(params_D, lr=0.00001)




START_ITER = 0
if START_ITER > 0:
    lm = torch.load('{}/checkpoint/v{}/model_ESRGAN_i{:06d}.pth'.format(EXP_NAME, str(141), START_ITER))
    model_ESRGAN.load_state_dict(lm.state_dict(), strict=True)
    



# Training dataset
Iter_H = GeneratorEnqueuer(DirectoryIterator_NTIRE2020(  #GeneratorEnqueuer
                                   TRAIN_DIR,    # ciplab-seven
                                #    '/host/home/ciplab/users/yhjo/dataset/REDS/train/',    # ciplab-seven
                                 out_batch_size=NB_BATCH, 
                                 crop_size=384+UPSCALE*4,
                                 scale_factor=UPSCALE,
                                 crop_per_frame = NB_CROP_FRAME,
                                 shuffle=True))
Iter_H.start(max_q_size=16, workers=2)



def SaveCheckpoint():
    torch.save(model_ESRGAN, '{}/checkpoint/v{}/model_ESRGAN_i{:06d}.pth'.format(EXP_NAME, str(VERSION), i))
    # torch.save(model_ESRGAN2, '{}/checkpoint/v{}/model_ESRGAN2_i{:06d}.pth'.format(EXP_NAME, str(VERSION), i))
    # torch.save(model_ESRGAN_x2, '{}/checkpoint/v{}/model_ESRGAN_x2_i{:06d}.pth'.format(EXP_NAME, str(VERSION), i))

    # torch.save(model_FH, '{}/checkpoint/v{}/model_FH_i{:06d}.pth'.format(EXP_NAME, str(VERSION), i))
    torch.save(model_D, '{}/checkpoint/v{}/model_D_i{:06d}.pth'.format(EXP_NAME, str(VERSION), i))
    # torch.save(model_NL, '{}/checkpoint/v{}/model_NL_i{:06d}.pth'.format(EXP_NAME, str(VERSION), i))

    torch.save(opt_G, '{}/checkpoint/v{}/opt_G_i{:06d}.pth'.format(EXP_NAME, str(VERSION), i))
    torch.save(opt_D, '{}/checkpoint/v{}/opt_D_i{:06d}.pth'.format(EXP_NAME, str(VERSION), i))
    print("Checkpoint saved")





accum_samples = 0



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


n_mix = 0

# TRAINING
for i in tqdm(range(START_ITER+1, NB_ITER+1)):

    # # Adjust LR 
    # if i == ((NB_ITER // 4) * 2) + 1:
    #     for param_group in opt_G.param_groups:
    #         param_group['lr'] = 0.00001/2
    #     for param_group in opt_D.param_groups:
    #         param_group['lr'] = 0.00001/2

    # elif i == ((NB_ITER // 4) * 3) + 1:
    #     for param_group in opt_G.param_groups:
    #         param_group['lr'] = 0.00001/2/2
    #     for param_group in opt_D.param_groups:
    #         param_group['lr'] = 0.00001/2/2


    model_ESRGAN.train()
    model_D.train()


    p_mix = i / 100000
    if p_mix > 0.5:
        p_mix = 0.5


    # Data preparing
    # EDVR (vimeo): 7 frames, Matlab downsampling
    st = time.time()
    batch_H = Iter_H.dequeue()  # BxCxTxHxW
    batch_H = Variable(torch.from_numpy(batch_H)).cuda()  # Matlab downsampled
    # batch_H = batch_H[:,:,0]

    batch_L_Matlab = DownSample2DMatlab(batch_H, 1/float(UPSCALE))
    batch_L_Matlab = torch.clamp(batch_L_Matlab, 0, 1)

    # batch_LI_Matlab = DownSample2DMatlab(batch_H, 1/4.0)
    # batch_LI_Matlab = torch.clamp(batch_LI_Matlab, 0, 1)

    batch_H = batch_H[:,:,UPSCALE*2:-UPSCALE*2,UPSCALE*2:-UPSCALE*2]
    batch_L_Matlab = batch_L_Matlab[:,:,2:-2,2:-2]

    # batch_LI_Matlab = batch_LI_Matlab[:,:,8:-8,8:-8]

    dT += time.time() - st



    # TRAIN D
    st = time.time()

    opt_G.zero_grad()
    opt_D.zero_grad()
    
    # RCAN
    batch_S_RCAN = model_ESRGAN(batch_L_Matlab).detach()

    batch_S_RCAN_CutMix = batch_S_RCAN.clone()

    # FH
    e_S, d_S, _, _ = model_D( batch_S_RCAN )
    e_H, d_H, _, _ = model_D( batch_H )

    # Loss
    # D, real/fake
    loss_D_E_S = torch.nn.ReLU()(1.0 + e_S).mean()
    loss_D_E_H = torch.nn.ReLU()(1.0 - e_H).mean()

    loss_D_D_S = torch.nn.ReLU()(1.0 + d_S).mean()
    loss_D_D_H = torch.nn.ReLU()(1.0 - d_H).mean()

    loss_D = loss_D_E_H + loss_D_D_H

    # CutMix
    if torch.rand(1) <= p_mix:
        n_mix += 1

        r_mix = torch.rand(1)   # real/fake ratio

        # rand_index = torch.randperm(batch_S_RCAN.size()[0]).cuda()
        bbx1, bby1, bbx2, bby2 = rand_bbox(batch_S_RCAN.size(), r_mix)
        batch_S_RCAN_CutMix[:, :, bbx1:bbx2, bby1:bby2] = batch_H[:, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        r_mix = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_S_RCAN.size()[-1] * batch_S_RCAN.size()[-2]))


        # tmp = batch_S_RCAN_CutMix.cpu().data.numpy()
        # outs = []
        # for b in range(NB_BATCH):
        #     outs += [tmp[b].transpose((1,2,0))]
            
        # PIL.Image.fromarray(np.around(np.concatenate(outs, 1)*255).astype(np.uint8)).save("r_mix.png".format(r_mix))
        # input(r_mix)


        e_mix, d_mix, _, _ = model_D( batch_S_RCAN_CutMix )

        loss_D_E_S = torch.nn.ReLU()(1.0 + e_mix).mean()
        loss_D_D_S = torch.nn.ReLU()(1.0 + d_mix).mean()

        d_S[:,:,bbx1:bbx2, bby1:bby2] = d_H[:,:,bbx1:bbx2, bby1:bby2]
        loss_D_Cons = F.mse_loss(d_mix, d_S)

        loss_D += loss_D_Cons
        l_accum[5] += torch.mean(loss_D_Cons).item()

    loss_D += loss_D_E_S + loss_D_D_S

    # Update
    loss_D.backward()
    torch.nn.utils.clip_grad_norm_(params_D, 0.1)
    opt_D.step()
    rT += time.time() - st

    l_accum[0] += loss_D.item()

    # for monitoring
    l_accum[1] += torch.mean(e_H).item()
    l_accum[2] += torch.mean(e_S).item()
    l_accum[3] += torch.mean(d_H).item()
    l_accum[4] += torch.mean(d_S).item()





    # Model
    # EDVR (vimeo): 7 frames, Matlab downsampling
    st = time.time()
    batch_H = Iter_H.dequeue()  # BxCxTxHxW
    batch_H = Variable(torch.from_numpy(batch_H)).cuda()  # Matlab downsampled
    # batch_H = batch_H[:,:,0]

    batch_L_Matlab = DownSample2DMatlab(batch_H, 1/float(UPSCALE))
    batch_L_Matlab = torch.clamp(batch_L_Matlab, 0, 1)

    # batch_LI_Matlab = DownSample2DMatlab(batch_H, 1/4.0)
    # batch_LI_Matlab = torch.clamp(batch_LI_Matlab, 0, 1)

    batch_H = batch_H[:,:,UPSCALE*2:-UPSCALE*2,UPSCALE*2:-UPSCALE*2]
    batch_L_Matlab = batch_L_Matlab[:,:,2:-2,2:-2]

    # batch_LI_Matlab = batch_LI_Matlab[:,:,8:-8,8:-8]

    dT += time.time() - st




    st = time.time()

    opt_G.zero_grad()
    opt_D.zero_grad()


    loss_Pixels = []

    batch_S = model_ESRGAN(batch_L_Matlab)



    loss_Pixels = Huber(batch_S, batch_H)
    loss_LPIPSs, _ = model_LPIPS.forward_pair(batch_H*2-1, batch_S*2-1)
    
    # Update
    loss_Pixel = torch.mean(loss_Pixels)
    loss_LPIPS = torch.mean(loss_LPIPSs)*1e-3
    # loss_FM = torch.mean(torch.stack(loss_FMs))

    loss_G = loss_Pixel 


    # FH
    e_S, d_S, e_Ss, d_Ss = model_D( batch_S )
    e_H, d_H, e_Hs, d_Hs = model_D( batch_H )

    loss_FMs = []
    for f in range(6):
        loss_FMs += [Huber(e_Ss[f], e_Hs[f])]
        loss_FMs += [Huber(d_Ss[f], d_Hs[f])]

    loss_FMs += [torch.nn.ReLU()(1.0 - e_S).mean() * 1e-3]
    loss_FMs += [torch.nn.ReLU()(1.0 - d_S).mean() * 1e-3]

    loss_FM = torch.mean(torch.stack(loss_FMs))

    # al = 0

    loss_G += loss_LPIPS + loss_FM
    
    l_accum[8] += loss_FM.item()




    loss_G.backward()
    torch.nn.utils.clip_grad_norm_(params_G, 0.1)
    opt_G.step()


    rT += time.time() - st

    accum_samples += NB_BATCH
    l_accum[6] += loss_Pixel.item()
    l_accum[7] += loss_LPIPS.item()
    l_accum[9] += loss_G.item()



    if i % I_DISPLAY == 0:
        # writer.add_scalar('loss_D', l_accum[0]/I_DISPLAY, i)
        # writer.add_scalar('prob_real', l_accum[1]/I_DISPLAY, i)
        # writer.add_scalar('prob_fake', (l_accum[2]+l_accum[3])/2/I_DISPLAY, i)
        # writer.add_scalar('loss_Pixel', l_accum[0]/I_DISPLAY, i)
        # writer.add_scalar('loss_LPIPS', l_accum[1]/I_DISPLAY, i)
        # writer.add_scalar('loss_G', l_accum[2]/I_DISPLAY, i)

        print("{} {}| Iter:{:6d}, Sample:{:6d}, D:{:.2e}, DE(1/-1):{:.2f}/{:.2f}, DD(1/-1):{:.2f}/{:.2f}, nMix:{:2d}, Dcons:{:.2e}, Pixel:{:.2e}, FM:{:.2e}, Adv:{:.2e}, G:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
            EXP_NAME, VERSION, i, accum_samples, l_accum[0]/I_DISPLAY, l_accum[1]/I_DISPLAY, l_accum[2]/I_DISPLAY, l_accum[3]/I_DISPLAY, l_accum[4]/I_DISPLAY, n_mix, l_accum[5]/(n_mix+1e-12), l_accum[6]/I_DISPLAY, l_accum[7]/I_DISPLAY, l_accum[8]/I_DISPLAY, l_accum[9]/I_DISPLAY, dT/I_DISPLAY, rT/I_DISPLAY))
        l_accum = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        l_accum_n = 0.
        n_mix=0
        dT = 0.
        rT = 0.


    if i % I_SAVE == 0:
        SaveCheckpoint()


    if i % I_VALIDATION == 0:
        with torch.no_grad():
            model_ESRGAN.eval()

            # Test for validation images
            vid4_dir = VAL_DIR
            files = glob.glob(vid4_dir + '/*.png')

            psnrs = []
            lpips = []
            if not isdir('{}/result/v{}'.format(EXP_NAME, str(VERSION))):
                mkdir('{}/result/v{}'.format(EXP_NAME, str(VERSION)))

            for fn in files:
                tmp = _load_img_array(fn)

                val_H = np.asarray(tmp).astype(np.float32) # HxWxC
                val_H_ = val_H
                val_H_ = np.transpose(val_H_, [2, 0, 1]) # CxHxW
                batch_H = val_H_[np.newaxis, ...]

                # # Fixed padding 8,8
                # h_pad = [8, 8]
                # w_pad = [8, 8]
                # batch_H = np.lib.pad(batch_H, pad_width=((0,0),(0,0),(h_pad[0],h_pad[1]),(0,0)), mode = 'reflect')
                # batch_H = np.lib.pad(batch_H, pad_width=((0,0),(0,0),(0,0),(w_pad[0],w_pad[1])), mode = 'reflect')



                h_least_multiple = UPSCALE
                w_least_multiple = UPSCALE

                h_pad = [0, 0]
                w_pad = [0, 0]
                if batch_H.shape[2] % h_least_multiple > 0:
                    t = h_least_multiple - (batch_H.shape[2] % h_least_multiple )
                    h_pad = [t//2, t-t//2]
                    batch_H = np.lib.pad(batch_H, pad_width=((0,0),(0,0),(h_pad[0],h_pad[1]),(0,0)), mode = 'reflect')
                if batch_H.shape[3] % w_least_multiple > 0:
                    t = w_least_multiple - (batch_H.shape[3] % w_least_multiple )
                    w_pad = [t//2, t-t//2]
                    batch_H = np.lib.pad(batch_H, pad_width=((0,0),(0,0),(0,0),(w_pad[0],w_pad[1])), mode = 'reflect')



                # val_H_ = batch_H[np.newaxis, ...]  # BxCxTxHxW
                batch_H = Variable(torch.from_numpy(batch_H), volatile=True).cuda()
                        
                # Down sampling
                xh = DownSample2DMatlab(batch_H, 1/UPSCALE)
                xh = torch.clamp(xh, 0, 1)

                batch_output = model_ESRGAN(xh)

                #
                out_FI2 = (batch_output).cpu().data.numpy()
                out_FI2 = np.clip(out_FI2[0], 0. , 1.) # CxHxW
                out_FI2 = np.transpose(out_FI2, [1, 2, 0])

                if h_pad[0] > 0:
                    out_FI2 = out_FI2[h_pad[0]:,:,:]
                if h_pad[1] > 0:
                    out_FI2 = out_FI2[:-h_pad[1],:,:]
                if w_pad[0] > 0:
                    out_FI2 = out_FI2[:,w_pad[0]:,:]
                if w_pad[1] > 0:
                    out_FI2 = out_FI2[:,:-w_pad[1],:]
                    
                # Save to file
                CROP_S = 16

                img_gt = (val_H*255).astype(np.uint8)
                img_target = ((out_FI2)*255).astype(np.uint8)

                Image.fromarray(np.around(out_FI2*255).astype(np.uint8)).save('{}/result/v{}/{}.png'.format(EXP_NAME, str(VERSION), fn.split('/')[-1]))
                psnrs.append(PSNR(_rgb2ycbcr(img_gt)[CROP_S:-CROP_S,CROP_S:-CROP_S,0], _rgb2ycbcr(img_target)[CROP_S:-CROP_S,CROP_S:-CROP_S,0], 0))

                # LPIPS
                # CROP_T = 2
                img_gt = im2tensor(img_gt) # RGB image from [-1,1]
                img_target = im2tensor(img_target)
                if CROP_S > 0:
                    img_target = img_target[:,:,CROP_S:-CROP_S,CROP_S:-CROP_S]
                    img_gt = img_gt[:,:,CROP_S:-CROP_S,CROP_S:-CROP_S]
                dist = model_LPIPS.forward(img_target, img_gt)
                lpips.append(dist)

        print('AVG PSNR/LPIPS: Set5: {}/{}'.format(np.mean(np.asarray(psnrs)), np.mean(np.asarray(lpips))))


        if i % I_SAVE == 0:
            # writer.add_scalar('set5', np.mean(np.asarray(psnrs)), i)
            # writer.add_scalar('set5_lpips', np.mean(np.asarray(lpips)), i)

            if np.mean(np.asarray(lpips)) < best_avg_lpips:
                best_avg_lpips = np.mean(np.asarray(lpips))
                
                SaveCheckpoint()
