import numpy as np
import logging
import cPickle as pickle

logging.basicConfig(level=logging.INFO)
try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)

def sum(input, labels=None, index=None):
    count, sum = _stats(input, labels, index)
    return sum


def apply_mean_std(im, mean_var):
    """
    Supercedes the standardize function. Takes the mean/var  file generated during preprocessed data generation and
    applies the normalization step to the patch.
    :param im: patch of size  (num_egs, channels, x, y, z) or (channels, x, y, z)
    :param mean_var: dictionary containing mean/var value calculated in preprocess.py
    :return: normalized patch
    """

    # expects a dictionary of means and VARIANCES, NOT STD
    for m in range(0, 4):
        if len(np.shape(im)) > 4:
            im[:, m, ...] = (im[:, m, ...] - mean_var['mn'][m]) / np.sqrt(mean_var['var'][m])
        else:
            im[m, ...] = (im[m, ...] - mean_var['mn'][m]) / np.sqrt(mean_var['var'][m])

    return im


def standardize(images, findMeanVarOnly=True, saveDump=None, applyToTest=None):
    """
    This function standardizes the input data to zero mean and unit variance. It is capable of calculating the
    mean and std values from the input data, or can also apply user specified mean/std values to the images.

    :param images: numpy ndarray of shape (num_qg, channels, x, y, z) to apply mean/std normalization to
    :param findMeanVarOnly: only find the mean and variance of the input data, do not normalize
    :param saveDump: if True, saves the calculated mean/variance values to the disk in pickle form
    :param applyToTest: apply user specified mean/var values to given images. checkLargestCropSize.ipynb has more info
    :return: standardized images, and vals (if mean/val was calculated by the function
    """

    # takes a dictionary
    if applyToTest != None:
        logger.debug('Applying to test data using provided values')
        images = apply_mean_std(images, applyToTest)
        return images

    logger.info('Calculating mean value..')
    vals = {
        'mn': [],
        'var': []
    }
    for i in range(4):
        vals['mn'].append(np.mean(images[:, i, :, :, :]))

    logger.info('Calculating variance..')
    for i in range(4):
        vals['var'].append(np.var(images[:, i, :, :, :]))

    if findMeanVarOnly == False:
        logger.info('Starting standardization process..')

        for i in range(4):
            images[:, i, :, :, :] = ((images[:, i, :, :, :] - vals['mn'][i]) / float(vals['var'][i]))

        logger.info('Data standardized!')

    if saveDump != None:
        logger.info('Dumping mean and var values to disk..')
        pickle.dump(vals, open(saveDump, 'wb'))
    logger.info('Done!')

    return images, vals