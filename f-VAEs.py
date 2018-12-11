#! -*- coding: utf-8 -*-
# Implement of <f-VAEs: Improve VAEs with Conditional Flows>
# https://arxiv.org/abs/1809.05861

import numpy as np
from scipy import misc
import glob
import imageio
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import Callback
import flow_layers as fl


imgs = glob.glob('CelebA-HQ/train/*.png')
np.random.shuffle(imgs)

height,width = misc.imread(imgs[0]).shape[:2]
center_height = int((height - width) / 2)
img_dim = 128


def imread(f):
    x = misc.imread(f)
    x = x[center_height:center_height+width, :]
    x = misc.imresize(x, (img_dim, img_dim))
    return x.astype(np.float32) / 255 * 2 - 1


def data_generator(batch_size=32):
    X = []
    while True:
        np.random.shuffle(imgs)
        for f in imgs:
            X.append(imread(f))
            if len(X) == batch_size:
                X = np.array(X)
                yield X,None
                X = []


x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in

for i in range(3):
    x = fl.Squeeze()(x)
    for j in range(12):
        x_ = x
        x_ = Conv2D(K.int_shape(x_)[-1],
                    kernel_size=(3, 3),
                    padding='SAME')(x_)
        x_ = BatchNormalization()(x_)
        x_ = Activation('relu')(x_)
        x_ = Conv2D(K.int_shape(x_)[-1],
                    kernel_size=(1, 1),
                    padding='SAME',
                    kernel_initializer='zeros')(x_)
        x = Add()([x, x_])


encoder = Model(x_in, x)
encoder.summary()


z_in = Input(shape=K.int_shape(encoder.output)[1:])
z = z_in

for i in range(3):
    for j in range(12):
        z_ = z
        z_ = Conv2D(K.int_shape(z_)[-1],
                    kernel_size=(3, 3),
                    padding='SAME')(z_)
        z_ = BatchNormalization()(z_)
        z_ = Activation('relu')(z_)
        z_ = Conv2D(K.int_shape(z_)[-1],
                    kernel_size=(1, 1),
                    padding='SAME',
                    kernel_initializer='zeros')(z_)
        z = Add()([z, z_])
    z = fl.UnSqueeze()(z)

z = Activation('tanh')(z)

decoder = Model(z_in, z)
decoder.summary()


u = Lambda(lambda z: K.random_normal(shape=K.shape(z)))(x) # 留着，不能动
z = Reshape(K.int_shape(u)[1:]+(1,))(u)
z = fl.Actnorm(use_shift=False)(z)
z = Reshape(K.int_shape(z)[1:-1])(z)
z = Add()([z, x])

x_recon = decoder(z)
x_recon = Subtract()([x_recon, x_in])
x_recon = Reshape(K.int_shape(x_recon)[1:]+(1,))(x_recon)
x_recon = fl.Actnorm(use_shift=False)(x_recon)
x_recon = Reshape(K.int_shape(x_recon)[1:-1])(x_recon)

recon_loss = 0.5 * K.sum(K.mean(x_recon**2, 0)) + 0.5 * np.log(2*np.pi) * np.prod(K.int_shape(x_recon)[1:])


depth = 12
level = 4

def build_basic_model(in_size, in_channel):
    """基础模型，即耦合层中的模型（basic model for Coupling）
    """
    _in = Input(shape=(None, None, in_channel))
    _ = _in
    hidden_dim = 256
    _ = Conv2D(hidden_dim,
               (3, 3),
               padding='same')(_)
    # _ = fl.Actnorm(add_logdet_to_loss=False)(_)
    _ = Activation('relu')(_)
    _ = Conv2D(hidden_dim,
               (1, 1),
               activation='relu',
               padding='same')(_)
    # _ = fl.Actnorm(add_logdet_to_loss=False)(_)
    _ = Activation('relu')(_)
    _ = Conv2D(in_channel,
               (3, 3),
               kernel_initializer='zeros',
               padding='same')(_)
    return Model(_in, _)


squeeze = fl.Squeeze()
inner_layers = []
outer_layers = []
for i in range(5):
    inner_layers.append([])

for i in range(3):
    outer_layers.append([])


x = z
in_size = K.int_shape(encoder.outputs[0])
x_outs = []

for i in range(level):
    for j in range(depth):
        actnorm = fl.Actnorm()
        permute = fl.Permute(mode='random')
        split = fl.Split()
        couple = fl.CoupleWrapper(build_basic_model(in_size[1]/2**i, in_size[-1]/2*2**i))
        concat = fl.Concat()
        inner_layers[0].append(actnorm)
        inner_layers[1].append(permute)
        inner_layers[2].append(split)
        inner_layers[3].append(couple)
        inner_layers[4].append(concat)
        x = actnorm(x)
        x = permute(x)
        x1, x2 = split(x)
        x1, x2 = couple([x1, x2])
        x = concat([x1, x2])
    if i < level-1:
        split = fl.Split()
        condactnorm = fl.CondActnorm()
        reshape = fl.Reshape()
        outer_layers[0].append(split)
        outer_layers[1].append(condactnorm)
        outer_layers[2].append(reshape)
        x1, x2 = split(x)
        x_out = condactnorm([x2, x1])
        x_out = reshape(x_out)
        x_outs.append(x_out)
        x = x1
        x = squeeze(x)
    else:
        for _ in outer_layers:
            _.append(None)


final_actnorm = fl.Actnorm()
final_concat = fl.Concat()
final_reshape = fl.Reshape()

x = final_actnorm(x)
x = final_reshape(x)
x = final_concat(x_outs+[x])
z = x

z_loss = 0.5 * K.sum(K.mean(z**2, 0)) - 0.5 * K.sum(K.mean(u**2, 0))
vae_loss = recon_loss + z_loss

vae = Model(x_in, [x_recon, z])
vae.add_loss(vae_loss)

for l in vae.layers:
    if hasattr(l, 'logdet'):
        vae.add_loss(l.logdet)

vae.compile(optimizer=Adam(1e-4))

total_encoder = Model(x_in, z)


# 搭建逆模型（生成模型），将所有操作倒过来执行

x_in = Input(shape=K.int_shape(z)[1:])
x = x_in

x = final_concat.inverse()(x)
outputs = x[:-1]
x = x[-1]
x = final_reshape.inverse()(x)
x = final_actnorm.inverse()(x)
x1 = x


for i,(split,condactnorm,reshape) in enumerate(zip(*outer_layers)[::-1]):
    if i > 0:
        x = squeeze.inverse()(x)
        x1 = x
        x_out = outputs[-i]
        x_out = reshape.inverse()(x_out)
        x2 = condactnorm.inverse()([x_out, x1])
        x = split.inverse()([x1, x2])
    for j,(actnorm,permute,split,couple,concat) in enumerate(zip(*inner_layers)[::-1][i*depth: (i+1)*depth]):
        x1, x2 = concat.inverse()(x)
        x1, x2 = couple.inverse()([x1, x2])
        x = split.inverse()([x1, x2])
        x = permute.inverse()(x)
        x = actnorm.inverse()(x)


flow_decoder = Model(x_in, x)
flow_decoder.summary()


def sample(path, std=1):
    n = 9
    figure = np.zeros((img_dim*n, img_dim*n, 3))
    for i in range(n):
        for j in range(n):
            noise_shape = (1,) + K.int_shape(flow_decoder.inputs[0])[1:]
            z_sample = np.array(np.random.randn(*noise_shape)) * std
            x_recon = decoder.predict(flow_decoder.predict(z_sample))
            digit = x_recon[0]
            figure[i*img_dim: (i+1)*img_dim,
                   j*img_dim: (j+1)*img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.clip(figure, 0, 255)
    imageio.imwrite(path, figure)


class Evaluate(Callback):
    def __init__(self):
        import os
        self.lowest = 1e10
        self.losses = []
        if not os.path.exists('samples'):
            os.mkdir('samples')
    def on_epoch_end(self, epoch, logs=None):
        path = 'samples/test_%s.png' % epoch
        sample(path, 1)
        path = 'samples/test_0.8_%s.png' % epoch
        sample(path, 0.8)
        self.losses.append((epoch, logs['loss']))
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            vae.save_weights('./best_flow_vae.weights')


evaluator = Evaluate()

vae.fit_generator(data_generator(),
                  epochs=1000,
                  steps_per_epoch=1000,
                  callbacks=[evaluator])







def encode_decode_sample(path):
    n = 9
    figure = np.zeros((img_dim*n, img_dim*n, 3))
    for i in range(n):
        for j in range(n):
            z_sample = np.array([imread(imgs[np.random.randint(len(imgs))])])
            x_recon = decoder.predict(encoder.predict(z_sample))
            digit = x_recon[0]
            figure[i*img_dim: (i+1)*img_dim,
                   j*img_dim: (j+1)*img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.clip(figure, 0, 255)
    imageio.imwrite(path, figure)


def interpolation_sample_2(path):
    n = 9
    figure = np.zeros((img_dim*n, img_dim*n, 3))
    for i in range(n):
        img1,img2 = np.random.choice(imgs, 2)
        z_sample_1,z_sample_2 = total_encoder.predict(np.array([imread(img1), imread(img2)]))
        for j in range(n):
            z_sample = 1.*j/(n-1) * z_sample_1 + (1-1.*j/(n-1)) * z_sample_2
            x_recon = decoder.predict(flow_decoder.predict(np.array([z_sample])))
            digit = x_recon[0]
            figure[i*img_dim: (i+1)*img_dim,
                   j*img_dim: (j+1)*img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.clip(figure, 0, 255)
    imageio.imwrite(path, figure)


def interpolation_sample_4(path):
    n = 9
    figure = np.zeros((img_dim*n, img_dim*n, 3))
    img1,img2,img3,img4 = [imread(i) for i in np.random.choice(imgs, 4)]
    z1,z2,z3,z4 = total_encoder.predict(np.array([img1, img2, img3, img4]))
    for i in range(n):
        for j in range(n):
            z5 = 1.*j/(n-1) * z1 + (1-1.*j/(n-1)) * z2
            z6 = 1.*j/(n-1) * z3 + (1-1.*j/(n-1)) * z4
            z_sample = 1.*i/(n-1) * z5 + (1-1.*i/(n-1)) * z6
            x_recon = decoder.predict(flow_decoder.predict(np.array([z_sample])))
            digit = x_recon[0]
            figure[i*img_dim: (i+1)*img_dim,
                   j*img_dim: (j+1)*img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.clip(figure, 0, 255)
    imageio.imwrite(path, figure)
