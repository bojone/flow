#! -*- coding: utf-8 -*-
# Keras implement of Glow
# Glow模型的Keras版
# https://blog.openai.com/glow/

from keras.layers import *
from keras.models import Model
from keras.datasets import cifar10
from keras.callbacks import Callback
from keras.optimizers import Adam
from flow_layers import *
import imageio
import numpy as np
from scipy import misc
import glob
import os


if not os.path.exists('samples'):
    os.mkdir('samples')

imgs = glob.glob('img_align_celeba/*.jpg')

height,width = misc.imread(imgs[0]).shape[:2]
center_height = int((height - width) / 2)

img_size = 64  # for a fast try, please use img_size=32
depth = 10  # orginal paper use depth=32
level = 3  # orginal paper use level=6 for 256*256 CelebA HQ


def imread(f):
    x = misc.imread(f)
    x = x[center_height:center_height+width, :]
    x = misc.imresize(x, (img_size, img_size))
    return x.astype(np.float32) / 256 - 0.5


def data_generator(batch_size=32):
    X = []
    while True:
        np.random.shuffle(imgs)
        for f in imgs:
            X.append(imread(f))
            if len(X) == batch_size:
                X = np.array(X)
                yield X,X.reshape((X.shape[0], -1))
                X = []

                
def build_basic_model(in_channel):
    """基础模型，即耦合层中的模型（basic model for Coupling）
    """
    _in = Input(shape=(None, None, in_channel))
    _ = _in
    hidden_dim = 512
    _ = Conv2D(hidden_dim,
               (3, 3),
               padding='same')(_)
    # _ = Actnorm(add_logdet_to_loss=False)(_)
    _ = Activation('relu')(_)
    _ = Conv2D(hidden_dim,
               (1, 1),
               padding='same')(_)
    # _ = Actnorm(add_logdet_to_loss=False)(_)
    _ = Activation('relu')(_)
    _ = Conv2D(in_channel,
               (3, 3),
               kernel_initializer='zeros',
               padding='same')(_)
    return Model(_in, _)


squeeze = Squeeze()
inner_layers = []
outer_layers = []
for i in range(5):
    inner_layers.append([])

for i in range(3):
    outer_layers.append([])


x_in = Input(shape=(img_size, img_size, 3))
x = x_in
x_outs = []

# 给输入加入噪声（add noise into inputs for stability.）
x = Lambda(lambda s: K.in_train_phase(s + 1./256 * K.random_uniform(K.shape(s)), s))(x)

for i in range(level):
    x = squeeze(x)
    for j in range(depth):
        actnorm = Actnorm()
        permute = Permute(mode='random')
        split = Split()
        couple = CoupleWrapper(build_basic_model(3*2**(i+1)))
        concat = Concat()
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
        split = Split()
        condactnorm = CondActnorm()
        reshape = Reshape()
        outer_layers[0].append(split)
        outer_layers[1].append(condactnorm)
        outer_layers[2].append(reshape)
        x1, x2 = split(x)
        x_out = condactnorm([x2, x1])
        x_out = reshape(x_out)
        x_outs.append(x_out)
        x = x1
    else:
        for _ in outer_layers:
            _.append(None)

final_actnorm = Actnorm()
final_concat = Concat()
final_reshape = Reshape()

x = final_actnorm(x)
x = final_reshape(x)
x = final_concat(x_outs+[x])


encoder = Model(x_in, x)
encoder.summary()
encoder.compile(loss=lambda y_true,y_pred: 0.5 * K.sum(y_pred**2, 1) + 0.5 * np.log(2*np.pi) * K.int_shape(y_pred)[1],
                optimizer=Adam(1e-4))


# 搭建逆模型（生成模型），将所有操作倒过来执行

x_in = Input(shape=K.int_shape(encoder.outputs[0])[1:])
x = x_in

x = final_concat.inverse()(x)
outputs = x[:-1]
x = x[-1]
x = final_reshape.inverse()(x)
x = final_actnorm.inverse()(x)
x1 = x


for i,(split,condactnorm,reshape) in enumerate(zip(*outer_layers)[::-1]):
    if i > 0:
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
    x = squeeze.inverse()(x)


decoder = Model(x_in, x)


def sample(std, path):
    """采样查看生成效果（generate samples per epoch）
    """
    n = 9
    figure = np.zeros((img_size * n, img_size * n, 3))
    for i in range(n):
        for j in range(n):
            decoder_input_shape = (1,) + K.int_shape(decoder.inputs[0])[1:]
            z_sample = np.array(np.random.randn(*decoder_input_shape)) * std
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(img_size, img_size, 3)
            figure[i * img_size: (i + 1) * img_size,
                   j * img_size: (j + 1) * img_size] = digit
    figure = np.clip((figure+0.5)*256, 0, 255)
    imageio.imwrite(path, figure)


class Evaluate(Callback):
    def __init__(self):
        self.lowest = 1e10
    def on_epoch_end(self, epoch, logs=None):
        path = 'samples/test_%s.png' % epoch
        sample(1, path)
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            encoder.save_weights('./best_encoder.weights')
        elif logs['loss'] > 0 and epoch > 10:
            """在后面，loss一般为负数，一旦重新变成正数，
            就意味着模型已经崩溃，需要降低学习率。
            In general, loss is less than zero.
            If loss is greater than zero again, it means model has collapsed.
            We need to reload the best model and lower learning rate.
            """
            encoder.load_weights('./best_encoder.weights')
            K.set_value(encoder.optimizer.lr, 1e-5)


evaluator = Evaluate()

encoder.fit_generator(data_generator(),
                      steps_per_epoch=1000,
                      epochs=1000,
                      callbacks=[evaluator])
