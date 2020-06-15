

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from PIL import Image
import numpy as np
import time
from os import mkdir
from os.path import join, isdir
from tqdm import tqdm
import glob


from utils import PSNR, GeneratorEnqueuer, DirectoryIterator_NTIRE2020, DownSample2DMatlab, Huber, _load_img_array, _rgb2ycbcr, im2tensor



### USER PARAMS ###
EXP_NAME = "PESR"
VERSION = "1"
UPSCALE = 16        # upscaling factor

NB_BATCH = 2        # mini-batch
NB_CROP_FRAME = 2   # crop N patches from every image
PATCH_SIZE = 384    # Training patch size

START_ITER = 0      # Set 0 for from scratch, else will load saved params and trains further
NB_ITER_MSE = 75000 # MSE pretraining iteration, after that the full loss works
NB_ITER = 150000    # Total number of training iterations

I_DISPLAY = 100     # display info every N iteration
I_VALIDATION = 100  # validate every N iteration
I_SAVE = 1000       # save models every N iteration

L_ADV = 1e-3        # Scaling params for the Adv loss
L_FM = 1            # Scaling params for the feature matching loss
L_LPIPS = 1e-3      # Scaling params for the LPIPS loss

TRAIN_DIR = './train/'  # Training images: png files should just locate in the directory (eg ./train/img0001.png ... ./train/img0800.png)
VAL_DIR = './val/'      # Validation images

LR_G = 1e-5         # Learning rate for the generator
LR_D = 1e-5         # Learning rate for the discriminator

best_avg_lpips = 0.4


### Quality mesuare ###
## LPIPS
import LPIPS.models.dist_model as dm
model_LPIPS = dm.DistModel()
model_LPIPS.initialize(model='net-lin',net='alex',use_gpu=True)


### Generator ###
## ESRGAN for x16
import RRDBNet_arch as arch
model_G = arch.RRDBNetx4x4(3, 3, 64, 23, gc=32).cuda()



### U-Net Discriminator ###
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

    # def shortcut(self, x):
    #     if self.preactivation:
    #         if self.learnable_sc:
    #             x = self.conv_sc(x)
    #         if self.downsample:
    #             x = self.downsample(x)
    #     else:
    #         if self.downsample:
    #             x = self.downsample(x)
    #         if self.learnable_sc:
    #             x = self.conv_sc(x)
    #     return x
        
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


## Optimizers
params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
opt_G = optim.Adam(params_G, lr=LR_G)

params_D = list(filter(lambda p: p.requires_grad, model_D.parameters()))
opt_D = optim.Adam(params_D, lr=LR_D)


## Load saved params
if START_ITER > 0:
    lm = torch.load('{}/checkpoint/v{}/model_G_i{:06d}.pth'.format(EXP_NAME, str(VERSION), START_ITER))
    model_G.load_state_dict(lm.state_dict(), strict=True)
    

# Training dataset
Iter_H = GeneratorEnqueuer(DirectoryIterator_NTIRE2020(
                            TRAIN_DIR,
                            out_batch_size = NB_BATCH, 
                            crop_size = PATCH_SIZE + UPSCALE*4,
                            scale_factor = UPSCALE,
                            crop_per_frame = NB_CROP_FRAME,
                            shuffle=True))
Iter_H.start(max_q_size=16, workers=2)



## Prepare directories
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


## Some preparations 
print('===> Training start')
l_accum = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
l_accum_n = 0.
dT = 0.
rT = 0.
n_mix = 0
accum_samples = 0


def SaveCheckpoint(i, best=False):
    str_best = ''
    if best:
        str_best = '_best'

    torch.save(model_G, '{}/checkpoint/v{}/model_G_i{:06d}{}.pth'.format(EXP_NAME, str(VERSION), i, str_best))
    torch.save(model_D, '{}/checkpoint/v{}/model_D_i{:06d}{}.pth'.format(EXP_NAME, str(VERSION), i, str_best))

    torch.save(opt_G, '{}/checkpoint/v{}/opt_G_i{:06d}{}.pth'.format(EXP_NAME, str(VERSION), i, str_best))
    torch.save(opt_D, '{}/checkpoint/v{}/opt_D_i{:06d}{}.pth'.format(EXP_NAME, str(VERSION), i, str_best))
    print("Checkpoint saved")


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



### TRAINING
for i in tqdm(range(START_ITER+1, NB_ITER+1)):

    model_G.train()
    model_D.train()


    if i > NB_ITER_MSE:
        ## TRAIN D
        # Data preparing
        st = time.time()
        batch_H = Iter_H.dequeue()  # BxCxHxW, data range [0, 1]
        batch_H = Variable(torch.from_numpy(batch_H)).cuda()  # Matlab downsampled

        batch_L_Matlab = DownSample2DMatlab(batch_H, 1/float(UPSCALE))
        batch_L_Matlab = torch.clamp(batch_L_Matlab, 0, 1)

        # Crop borders for avoiding errors
        batch_H = batch_H[:,:,UPSCALE*2:-UPSCALE*2,UPSCALE*2:-UPSCALE*2]
        batch_L_Matlab = batch_L_Matlab[:,:,2:-2,2:-2]
        dT += time.time() - st


        st = time.time()
        opt_G.zero_grad()
        opt_D.zero_grad()
        
        # G
        batch_S = model_G(batch_L_Matlab).detach()

        # D
        e_S, d_S, _, _ = model_D( batch_S )
        e_H, d_H, _, _ = model_D( batch_H )

        # D Loss, for encoder end and decoder end
        loss_D_Enc_S = torch.nn.ReLU()(1.0 + e_S).mean()
        loss_D_Enc_H = torch.nn.ReLU()(1.0 - e_H).mean()

        loss_D_Dec_S = torch.nn.ReLU()(1.0 + d_S).mean()
        loss_D_Dec_H = torch.nn.ReLU()(1.0 - d_H).mean()

        loss_D = loss_D_Enc_H + loss_D_Dec_H

        # CutMix for consistency loss
        batch_S_CutMix = batch_S.clone()

        # probability of doing cutmix
        p_mix = i / 100000
        if p_mix > 0.5:
            p_mix = 0.5

        if torch.rand(1) <= p_mix:
            n_mix += 1
            r_mix = torch.rand(1)   # real/fake ratio

            bbx1, bby1, bbx2, bby2 = rand_bbox(batch_S_CutMix.size(), r_mix)
            batch_S_CutMix[:, :, bbx1:bbx2, bby1:bby2] = batch_H[:, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            r_mix = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_S_CutMix.size()[-1] * batch_S_CutMix.size()[-2]))

            e_mix, d_mix, _, _ = model_D( batch_S_CutMix )

            loss_D_Enc_S = torch.nn.ReLU()(1.0 + e_mix).mean()
            loss_D_Dec_S = torch.nn.ReLU()(1.0 + d_mix).mean()

            d_S[:,:,bbx1:bbx2, bby1:bby2] = d_H[:,:,bbx1:bbx2, bby1:bby2]
            loss_D_Cons = F.mse_loss(d_mix, d_S)

            loss_D += loss_D_Cons
            l_accum[5] += torch.mean(loss_D_Cons).item()

        loss_D += loss_D_Enc_S + loss_D_Dec_S

        # Update
        loss_D.backward()
        torch.nn.utils.clip_grad_norm_(params_D, 0.1)
        opt_D.step()
        rT += time.time() - st

        # for monitoring
        l_accum[0] += loss_D.item()
        l_accum[1] += torch.mean(e_H).item()
        l_accum[2] += torch.mean(e_S).item()
        l_accum[3] += torch.mean(d_H).item()
        l_accum[4] += torch.mean(d_S).item()


    ## TRAIN G
    st = time.time()
    batch_H = Iter_H.dequeue()  # BxCxHxW, data range [0, 1]
    batch_H = Variable(torch.from_numpy(batch_H)).cuda()  # Matlab downsampled

    batch_L_Matlab = DownSample2DMatlab(batch_H, 1/float(UPSCALE))
    batch_L_Matlab = torch.clamp(batch_L_Matlab, 0, 1)

    batch_H = batch_H[:,:,UPSCALE*2:-UPSCALE*2,UPSCALE*2:-UPSCALE*2]
    batch_L_Matlab = batch_L_Matlab[:,:,2:-2,2:-2]
    dT += time.time() - st


    st = time.time()
    opt_G.zero_grad()
    opt_D.zero_grad()

    batch_S = model_G(batch_L_Matlab)

    # Pixel loss
    loss_Pixel = Huber(batch_S, batch_H)
    loss_G = loss_Pixel

    if i > NB_ITER_MSE:
        # LPIPS loss
        loss_LPIPS, _ = model_LPIPS.forward_pair(batch_H*2-1, batch_S*2-1)
        loss_LPIPS = torch.mean(loss_LPIPS) * L_LPIPS

        # FM and GAN losses
        e_S, d_S, e_Ss, d_Ss = model_D( batch_S )
        _, _, e_Hs, d_Hs = model_D( batch_H )

        # FM loss
        loss_FMs = []
        for f in range(6):
            loss_FMs += [Huber(e_Ss[f], e_Hs[f])]
            loss_FMs += [Huber(d_Ss[f], d_Hs[f])]
        loss_FM = torch.mean(torch.stack(loss_FMs)) * L_FM

        # GAN loss
        loss_Advs = []
        loss_Advs += [torch.nn.ReLU()(1.0 - e_S).mean() * L_ADV]
        loss_Advs += [torch.nn.ReLU()(1.0 - d_S).mean() * L_ADV]
        loss_Adv = torch.mean(torch.stack(loss_Advs))

        loss_G += loss_LPIPS + loss_FM + loss_Adv

        # For monitoring
        l_accum[7] += loss_LPIPS.item()
        l_accum[8] += loss_FM.item()
        l_accum[9] += loss_Adv.item()
        
    # Update
    loss_G.backward()
    torch.nn.utils.clip_grad_norm_(params_G, 0.1)
    opt_G.step()
    rT += time.time() - st

    # For monitoring
    l_accum[6] += loss_Pixel.item()
    l_accum[10] += loss_G.item()
    accum_samples += NB_BATCH


    ## Show information
    if i % I_DISPLAY == 0:
        print("{} {} | Iter:{:6d}, Sample:{:6d}, D:{:.2e}, DEnc(1/-1):{:.2f}/{:.2f}, DDec(1/-1):{:.2f}/{:.2f}, nMix:{:2d}, Dcons:{:.2e}, GPixel:{:.2e}, GLPIPS:{:.2e}, GFM:{:.2e}, GAdv:{:.2e}, G:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
            EXP_NAME, VERSION, i, accum_samples, l_accum[0]/I_DISPLAY, l_accum[1]/I_DISPLAY, l_accum[2]/I_DISPLAY, l_accum[3]/I_DISPLAY, l_accum[4]/I_DISPLAY, n_mix, l_accum[5]/(n_mix+1e-12), l_accum[6]/I_DISPLAY, l_accum[7]/I_DISPLAY, l_accum[8]/I_DISPLAY, l_accum[9]/I_DISPLAY, l_accum[10]/I_DISPLAY, dT/I_DISPLAY, rT/I_DISPLAY))
        l_accum = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        l_accum_n = 0.
        n_mix = 0
        dT = 0.
        rT = 0.


    ## Save models
    if i % I_SAVE == 0:
        SaveCheckpoint(i)


    ## Validate
    if i % I_VALIDATION == 0:
        with torch.no_grad():
            model_G.eval()

            # Test for validation images
            vid4_dir = VAL_DIR
            files = glob.glob(vid4_dir + '/*.png')

            psnrs = []
            lpips = []

            for fn in files:
                # Load test image
                tmp = _load_img_array(fn)
                val_H = np.asarray(tmp).astype(np.float32) # HxWxC
                val_H_ = val_H
                val_H_ = np.transpose(val_H_, [2, 0, 1]) # CxHxW
                batch_H = val_H_[np.newaxis, ...]

                # Padding
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


                # Data
                batch_H = Variable(torch.from_numpy(batch_H)).cuda()
                        
                batch_L = DownSample2DMatlab(batch_H, 1/UPSCALE)
                batch_L = torch.clamp(torch.round(batch_L*255)/255.0, 0, 1)

                # Forward
                batch_Out = model_G(batch_L)

                # Output
                batch_Out = (batch_Out).cpu().data.numpy()
                batch_Out = np.clip(batch_Out[0], 0. , 1.) # CxHxW
                batch_Out = np.transpose(batch_Out, [1, 2, 0])

                # Crop padding to the original size
                if h_pad[0] > 0:
                    batch_Out = batch_Out[h_pad[0]:,:,:]
                if h_pad[1] > 0:
                    batch_Out = batch_Out[:-h_pad[1],:,:]
                if w_pad[0] > 0:
                    batch_Out = batch_Out[:,w_pad[0]:,:]
                if w_pad[1] > 0:
                    batch_Out = batch_Out[:,:-w_pad[1],:]
                    
                # Save to file
                Image.fromarray(np.around(batch_Out*255).astype(np.uint8)).save('{}/result/v{}/{}.png'.format(EXP_NAME, str(VERSION), fn.split('/')[-1]))

                # PSNR
                img_gt = (val_H*255).astype(np.uint8)
                img_target = ((batch_Out)*255).astype(np.uint8)

                CROP_S = 16
                if CROP_S > 0:
                    img_gt = img_gt[CROP_S:-CROP_S,CROP_S:-CROP_S]
                    img_target = img_target[CROP_S:-CROP_S,CROP_S:-CROP_S]
                
                psnrs.append(PSNR(_rgb2ycbcr(img_gt)[:,:,0], _rgb2ycbcr(img_target)[:,:,0], 0))

                # LPIPS
                dist = model_LPIPS.forward(im2tensor(img_target), im2tensor(img_gt))    # RGB image from [0,255] to [-1,1]
                lpips.append(dist)

        print('AVG PSNR/LPIPS: Validation: {}/{}'.format(np.mean(np.asarray(psnrs)), np.mean(np.asarray(lpips))))

        # Save best model
        if i % I_SAVE == 0:
            if np.mean(np.asarray(lpips)) < best_avg_lpips:
                best_avg_lpips = np.mean(np.asarray(lpips))
                SaveCheckpoint(i, best=True)
