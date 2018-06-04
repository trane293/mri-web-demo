import  os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import modules.dataloader as dl
import numpy as np
from model.isensee import open_model_with_hyper_and_history, custom_loss
import tensorflow as tf
import cPickle as pickle
from modules.training_helpers import standardize
from skimage import io, exposure, img_as_uint, img_as_float
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

io.use_plugin('freeimage')

UPLOAD_FOLDER = './upload'
SAMPLE_FOLDER = './sample'
NUM_SLICES = 155
ALLOWED_EXTENSIONS = set(['nii', 'nii.gz', 'gz'])
MODEL = None
graph = None
app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SAMPLE_FOLDER'] = SAMPLE_FOLDER
app.config['MODEL_PATH'] = './model/main_model.h5'
app.config['PREDICTION_PATH']= './predictions'

app.secret_key = "super secret key"

def clean_jpeg_dir():
    os.system('rm -rf ./predictions/edema/*')
    os.system('rm -rf ./predictions/enhancing/*')
    os.system('rm -rf ./predictions/necrotic/*')
    os.system('rm -rf ./upload/*')

def get_model():
    global MODEL
    global graph
    custom_obj = custom_loss()
    MODEL = open_model_with_hyper_and_history(app.config['MODEL_PATH'],
                                              custom_obj=custom_obj,
                                              load_model_only=True)

    # workarounds for the bug where the last layer of the graph is unconnected
    MODEL._make_predict_function()
    graph = tf.get_default_graph()

def save_predictions(predictions):
    # remove first dimension from the shape
    predictions[np.where(predictions < 0.1)] = 0
    predictions = predictions[0]

    nec = predictions[0,]
    ede = predictions[1,]
    enh = predictions[2,]

    arr_dict = {'necrotic': nec, 'edema': ede, 'enhancing': enh}

    # now we have 3 3D volumes. We save it as jpg files

    # save necrotic as JPG files
    for key, npa in arr_dict.items():
        for i in range(0, predictions.shape[-1]):
            im = npa[:,:,i]
            im = exposure.rescale_intensity(im, out_range='float')
            im = img_as_uint(im)

            if key == 'necrotic':
                filepath = os.path.join(app.config['PREDICTION_PATH'], 'necrotic', "nec_{}.png".format(i))
            elif key == 'edema':
                filepath = os.path.join(app.config['PREDICTION_PATH'], 'edema', "ede_{}.png".format(i))
            elif key == 'enhancing':
                filepath = os.path.join(app.config['PREDICTION_PATH'], 'enhancing', "enh_{}.png".format(i))
            io.imsave(filepath, im)


def get_predictions(MODEL, images):
    logger.debug('Starting prediction process!')
    # predict using the whole volume
    pred = MODEL.predict(images)
    logger.debug('Prediction done!')
    # get back the main volume and strip the padding
    pred = pred[:, :, :, :, 0:155]

    assert pred.shape == (1, 3, 240, 240, 155)

    return pred


def pad_image(images):
    '''
    Pad the image to nearest multiple of two for easy processing by UNet type models
    This function pads the z direction and makes it = 160.
    :param images:
    :return:
    '''
    tmp = list(np.shape(images))
    tmp[-1] = 160 # change it to 160
    padded_image = np.zeros(tmp)
    padded_image[:,:,:,:,0:155] = images
    return padded_image

def prepare_data():
    '''
    Prepare data for prediction. Performs the following tasks:
        1. Load data and perform N4ITK
        2. Pads image to nearest multiple of 2 (160).
        3. Loads mean/var file.
        4. Apply standardization
    :return:
    '''
    logger.debug('loading data..')
    images = dl.loadData('./upload', preprocess=True)

    # open mean/var file
    logger.debug('loading mean/var file..')
    mean_var = pickle.load(open('./model/BRATS2018_HDF5_Datasetstraining_data_hgg_mean_var.p', 'rb'))

    # standardize the images
    logger.debug('standardizing data..')
    images = standardize(images, applyToTest=mean_var)

    logger.debug('padding images..')
    padded_im = pad_image(images)
    return padded_im


def check_extension(filename):
    '''
    Check if the uploaded file  has a legal extension

    :param filename: filename of the upload
    :return: Boolean
    '''
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/show/')
def show_results():
    return render_template('final.html')


@app.route('/predict/')
def predict():
    global MODEL
    logger.debug('Starting loading data..')
    images = prepare_data()

    logger.debug('Saving this as sample..')
    np.save('./sample/sample.npy', images)
    logger.debug('Sample saved!')

    logger.debug('Initialize prediction process..')
    predictions = get_predictions(MODEL, images)

    logger.debug('Saving predictions to disk..')
    save_predictions(predictions)
    logger.debug('Predictions saved!')

    return redirect(url_for('show_results'))


@app.route('/demo', methods=['POST'])
def demo():
    global MODEL
    logger.debug('Loading demo data..')

    images = np.load(open('./sample/sample.npy', 'rb'))
    logger.debug('Sample loaded!')

    logger.debug('Initialize prediction process..')
    predictions = get_predictions(MODEL, images)

    logger.debug('Saving predictions to disk..')
    save_predictions(predictions)
    logger.debug('Predictions saved!')

    return redirect(url_for('show_results'))

@app.route('/uploads/')
def uploaded_file():
    return redirect(url_for('predict'))


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')

    elif request.method == 'POST':
        # check of the post request has the file  part
        if 't1' not in request.files or 't2' not in request.files or \
                't2flair' not in request.files or 't1ce' not in request.files:
            flash('No file part!')
            return redirect(request.url)

        flash(request.files.keys())
        # get all the files
        t1_file = request.files['t1']
        t2_file = request.files['t2']
        t2flair_file = request.files['t2flair']
        t1ce_file = request.files['t1ce']

        if t1_file.filename == '' or \
                t2_file.filename == '' or \
                t1ce_file.filename == '' or \
                t2flair_file.filename == '':
            flash('No selected file!')
            return redirect(request.url)

        if (t1_file and check_extension(t1_file.filename)) and \
                t2_file and check_extension(t2_file.filename) and \
                t1ce_file and check_extension(t1ce_file.filename) and \
                t2flair_file and check_extension(t2flair_file.filename):
            t1_filename = secure_filename(t1_file.filename)
            t1_file.save(os.path.join(app.config['UPLOAD_FOLDER'], t1_filename))

            t2_filename = secure_filename(t2_file.filename)
            t2_file.save(os.path.join(app.config['UPLOAD_FOLDER'], t2_filename))

            t1ce_filename = secure_filename(t1ce_file.filename)
            t1ce_file.save(os.path.join(app.config['UPLOAD_FOLDER'], t1ce_filename))

            t2flair_filename = secure_filename(t2flair_file.filename)
            t2flair_file.save(os.path.join(app.config['UPLOAD_FOLDER'], t2flair_filename))

            return redirect(url_for('uploaded_file'))

    return render_template('index.html')

if __name__ == '__main__':
    clean_jpeg_dir()
    get_model()
    app.run(debug=True)