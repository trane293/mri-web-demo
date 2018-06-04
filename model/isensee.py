import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D
from keras.optimizers import Adam
from functools import partial
from keras import backend as K
import logging
from keras.utils import multi_gpu_model
from unet3d import create_convolution_block, concatenate
from keras_contrib.layers.normalization import InstanceNormalization

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


create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)


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

def isensee2017_model(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid",
                      include_label_wise_dice_coefficients=True, metrics=dice_coefficient):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf
    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf
    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, create_convolution_block(current_layer, n_filters=n_labels, kernel=(1, 1, 1)))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation(activation_name)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)

    if include_label_wise_dice_coefficients:
        lab_names = {
            0: 'Necrotic',
            1: 'Edema',
            2: 'Enhancing'
        }

        label_wise_dice_metrics = [get_label_dice_coefficient_function(index, name) for index, name in
                                   lab_names.iteritems()]

        metrics = label_wise_dice_metrics


    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function, metrics=metrics)
    return model


def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2

def custom_loss():

    lab_names = {
        0: 'Necrotic',
        1: 'Edema',
        2: 'Enhancing'
    }

    label_wise_dice_metrics = [get_label_dice_coefficient_function(index, name) for index, name in
                               lab_names.iteritems()]
    metrics_dict = {func.__name__: func for func in label_wise_dice_metrics}

    name = 'weighted_dice_coefficient_loss'
    metrics_dict[name] = weighted_dice_coefficient_loss
    metrics_dict['dice_coefficient'] = dice_coefficient

    metrics_dict['InstanceNormalization'] = InstanceNormalization
    return metrics_dict


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
    model = isensee2017_model(input_shape=(4, None, None, None), n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=3, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid")
    return model

if __name__ == '__main__':
    model = isensee2017_model(input_shape=(4, None, None, None), n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=3, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid",
                      include_label_wise_dice_coefficients=True, metrics=dice_coefficient)
    logger.info('Created the model!')
