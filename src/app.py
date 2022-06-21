# pylint: disable=E0401, W0611
import logging
from flask import Flask, jsonify, request, render_template
from waitress import serve
from werkzeug.utils import secure_filename
import os

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten


class Inference():
    def __init__(self):
        self.FOODS = [
            'chilli_crab',
            'curry_puff',
            'dim_sum',
            'ice_kacang',
            'kaya_toast',
            'nasi_ayam',
            'popiah',
            'roti_prata',
            'sambal_stingray',
            'satay',
            'tau_huay',
            'wanton_noodle']
        self.img_width = 224
        self.img_height = 224
        self.pretrained_weights = None
        self.model = None

    def init_model(self):
        """
        Downloads Vgg16 model using keras. Removes top 3 layers and add
        2 dense layers. Init with trained weights stored on polyaxon.

        Returns
        model
        """
        self.model = tf.keras.applications.vgg16.VGG16(
            include_top=False,
            weights=self.pretrained_weights,
            input_shape=(self.img_height, self.img_width, 3))

        # Create additional layers for training
        x = self.model.output
        x = Dense(100, activation='relu')(x)
        x = Flatten()(x)
        pred_proba = Dense(12, activation='softmax')(x)
        self.model = Model(inputs=self.model.input, outputs=pred_proba)

        if self.model is None:
            logger.error("Failed to init model")

        self.model.load_weights('./model/tensorfood.h5')

        return self.model

    def process_image(self, filename):
        """
        Converts an image to array. Reshapes into (-1, img_height, img_width, 3)
        and normalises the values of the array.

        Arguments:
            filename {image format} -- filename of image user uploads
        Returns:
            input data
            A normalised array of shape (-1, img_height, img_width, 3)
        """
        img = image.img_to_array(
            image.load_img(
                filename,
                target_size=(self.img_height, self.img_width))) / 255.
        input_data = img.reshape((-1, self.img_height, self.img_width, 3))

        logger.info(f"Uploaded file: {filename}")

        if input_data is None:
            logger.error("Failed to convert required image input")

        return input_data

    def predict_image(self, model, input_data):
        """
        Predicts the image uploaded.

        Arguments:
            model {model} -- model initiated in app
            input_data {array}
            A normalised array of shape (-1, img_height, img_width, 3)

        Returns:
            pred_class -- {str} predicted class
            pred_proba - {float} probability of predicted class
        """
        pred = self.model.predict(input_data)
        y_classes = pred.argmax(axis=-1)
        pred_class = self.FOODS[y_classes[0]]
        pred_proba = int(pred[0][y_classes[0]]*100)/100

        # Guard cases where pred proba is too low or predictions may
        # not make sense
        if pred_proba < 0.80:
            pred_class = "I'm not certain what food this is. Sorry =p"

        print('Prediction')
        print('I think there\'s a {}% chance this is {}. Yummy!'.format(
            pred_proba*100,
            pred_class))

        logger.info("food: {}, probability: {}".format(pred_class, pred_proba))

        return pred_class, pred_proba


# Add logger steps
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

UPLOAD_FOLDER = './uploads'

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs('./uploads')

inf = Inference()
model = inf.init_model()


@app.route('/')
def index():
    return render_template('client.html')


@app.route('/info')
def short_description():

    return jsonify({'model': model.name,
                    'input-size': model.input_shape[1:],
                    'num-classes': model.output_shape[1],
                    'pretrained-on': "Imagenet"})


@app.route('/doc')
def readme():
    return render_template('README.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        input_data = inf.process_image(filepath)
        pred_class, pred_proba = inf.predict_image(model, input_data)

    return jsonify({'food': pred_class, 'probability': pred_proba})


if __name__ == "__main__":
    # app.run(host="0.0.0.0", debug=True, port=8000)
    # For production mode, comment the line above and uncomment below
    serve(app, host="0.0.0.0", port=8000)
