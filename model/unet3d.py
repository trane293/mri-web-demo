import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D
from keras.optimizers import Adam
from functools import partial
from keras import backend as K
import logging
from keras.utils import multi_gpu_model

logging.basicConfig(level=logging.INFO)
try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)

K.set_image_data_format("channels_first")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate

def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index, label_name):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_{1}_dice_coef'.format(label_index, label_name))
    return f


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss

def unet_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=3, initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, metrics=dice_coefficient,
                  batch_normalization=False, activation_name="sigmoid", loss_fn=weighted_dice_coefficient_loss, multi_gpu=False):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
    coefficient for each label as metric.
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and n_labels > 1:
        lab_names = {
            0: 'Necrotic',
            1: 'Edema',
            2: 'Enhancing'
        }

        label_wise_dice_metrics = [get_label_dice_coefficient_function(index, name) for index, name in lab_names.iteritems()]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics

    if multi_gpu == True:
        logger.warn('Compiling a MULTI GPU MODEL')
        model = multi_gpu_model(model, gpus=4)
        logger.warn('Done compiling!')

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=loss_fn, metrics=metrics)

    return model


def custom_loss(n_labels=3):
    name = 'weighted_dice_coefficient_loss'
    label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
    metrics_dict = {func.__name__: func for func in label_wise_dice_metrics}

    metrics_dict[name] = weighted_dice_coefficient_loss
    metrics_dict['dice_coefficient'] = dice_coefficient
    return metrics_dict


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    """

    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)


def save_model_with_hyper_and_history(model, history, name=None):
    import cPickle as pickle
    filename = name if name != None else "model"

    if '.h5' not in filename:
        filename_dict = filename + '_hyper_dict.p'
        filename = filename + '.h5'
        filename_history = filename + '_history.p'
    else:
        filename_dict = filename.split('.')[-2] + '_hyper_dict.p'
        filename_history = filename.split('.')[-2] + '_history.p'

    logger.info('Saving trained model with name {}'.format(filename))
    model.save(filename)
    logger.info('Model save successful!')

    logger.info('Saving history object with name {}'.format(filename_dict))
    with open(filename_history, "wb") as f:
        pickle.dump(history.history, f)
    logger.info('Saved history object!')


def open_model_with_hyper_and_history(name=None, custom_obj=None, load_model_only=False):
    import cPickle as pickle
    from keras.models import load_model
    filename = name if name != None else "model"

    if '.h5' not in filename:
        filename_dict = filename + '_hyper_dict.p'
        filename = filename + '.h5'
        filename_history = filename + '_history.p'
    else:
        filename_dict = filename.split('.')[-2] + '_hyper_dict.p'
        filename_history = filename.split('.')[-2] + '_history.p'

    logger.info('Opening trained model with name {}'.format(filename))
    model = load_model(filename, custom_objects=custom_obj)
    logger.info('Model open successful!')
    if load_model_only == False:
        logger.info('Opening history object with name {}'.format(filename_dict))
        history = pickle.load(open(filename_history, "rb"))
        logger.info('Opened history object!')
        return model, history

    return model


def get_model(inp_shape=(4,32,32,32)):
    model = unet_model_3d(input_shape=inp_shape, pool_size=(2, 2, 2), n_labels=3, initial_learning_rate=0.00001,
                          deconvolution=False,
                          depth=3, n_base_filters=32, include_label_wise_dice_coefficients=True,
                          metrics=dice_coefficient,
                          batch_normalization=True, activation_name="sigmoid", loss_fn=dice_coefficient_loss)
    return model

if __name__ == '__main__':
    model = unet_model_3d(input_shape=(4, None, None, None), pool_size=(2, 2, 2), n_labels=3, initial_learning_rate=0.00001, deconvolution=True,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=True, metrics=dice_coefficient,
                  batch_normalization=True, activation_name="sigmoid")
    logger.info('Created the model!')
