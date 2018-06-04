"""
==========================================================
                Load BRATS 2017 Data
==========================================================
AUTHOR: Anmol Sharma
AFFILIATION: Simon Fraser University
             Burnaby, BC, Canada
PROJECT: Analysis of Brain MRI Scans for Management of
         Malignant Tumors
COLLABORATORS: Anmol Sharma (SFU)
               Prof. Ghassan Hamarneh (SFU)
               Dr. Brian Toyota (VGH)
               Dr. Mostafa Fatehi (VGH)
DESCRIPTION: The script has multiple functions to load,
             preprocess, and standardize the BRATS
             2017 dataset, along with its survival annotations.
             Main function is the loadDataGenerator which loads
             the data using a generator, and doesn't hog memory.

             The loadDataGenerator is capable of applying
             arbitrary preprocessing steps to the data. This can be
             achieved by implementing the function preprocessData.
LICENCE: Proprietary for now.
"""

from __future__ import print_function
import glob as glob
import numpy as np
import pickle
import os
import logging
import SimpleITK as sitk

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def preprocessData(img_obj, process=False):
    """
    Perform preprocessing on the original nibabel object.
    Use this function to:
        1) Resize/Resample the 3D Volume
        2) Crop the brain region
        3) Do (2) then (1).

    When you do preprocessing, especially something that
    changes the spatial size of the volume, make sure you
    update config['spatial_size_for_training'] = (240, 240)
    value in the config file.

    :param img_obj:
    :param process:
    :return:
    """
    if process == False:
        return img_obj
    else:
        maskImage = sitk.OtsuThreshold(img_obj, 0, 1, 200)
        image = sitk.Cast(img_obj, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        numberFilltingLevels = 4
        corrector.SetMaximumNumberOfIterations([4] * numberFilltingLevels)
        output = corrector.Execute(image, maskImage)
        return output


def loadData(data_dir, preprocess=False):
    """
    Main function to load BRATS 2017 dataset.

    :param data_dir: path to the folder where patient data resides, needs individual paths for HGG and LGG
    :param batch_size: size of batch to load  (default=1)
    :param loadSurvival: load survival data (True/False) (default=False)
    :param csvFilePath: If loadSurvival is True, provide path to survival data (default=False)
    :param loadSeg: load segmentations (True/False) (default=True)
    :return:
    """

    num_sequences = 4 # number of sequences in the data. BRATS has 4.
    num_slices = 155
    running_pats = []
    batch_size = 1
    out_shape = (240, 240)

    # create placeholders, currently only supports theano type convention (num_eg, channels, x, y, z)
    images = np.empty((batch_size, num_sequences, out_shape[0], out_shape[1], num_slices)).astype(np.float32)

    logger.info('starting to load images..')
    for imagefile in glob.glob(data_dir + '/*'):
        if 't1.' in imagefile:
            i = 0
            seq_name = 't1'
        elif 't2.' in imagefile:
            i = 1
            seq_name = 't2'
        elif 't1ce.' in imagefile:
            i = 2
            seq_name = 't1ce'
        elif 'flair.' in imagefile:
            i = 3
            seq_name = 'flair'

        img_obj = sitk.ReadImage(imagefile)
        if preprocess == True:
            logger.debug('performing N4ITK Bias Field Correction on {} modality'.format(seq_name))
        img_obj = preprocessData(img_obj, process=preprocess)

        pix_data = sitk.GetArrayViewFromImage(img_obj)

        pix_data_swapped = np.swapaxes(pix_data, 0, 1)
        pix_data_swapped = np.swapaxes(pix_data_swapped, 1, 2)

        images[0, i, :, :, :] = pix_data_swapped

    return images

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
        logger.info('Applying to test data using provided values')
        from training_helpers import apply_mean_std
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


if __name__ == "__main__":
    """
    Only for testing purpose, DO NOT ATTEMPT TO RUN THIS SCRIPT. ONLY IMPORT AS MODULE
    """
    data_dir = '/local-scratch/cedar-rm/scratch/asa224/Datasets/BRATS2017/MICCAI_BraTS17_Data_Training/HGG/'
    images, segmasks = loadDataGenerator(data_dir, batch_size=2, loadSurvival=False,
                 csvFilePath=None, loadSeg=True)
