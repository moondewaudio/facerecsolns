"""
server.py

ECE196 Face Recognition Project
Author: Will Chen, Simon Fong

What this script should do:
1. Load a model with saved weights.
2. Create a webserver.
3. Handle classification requests:
    3.1 Save the image from the request.
    3.2 Load the image and classify it.
    3.3 Send the label and confidence back to requester(Pi).

Installation:
    pip install numpy keras tensorflow h5py flask flask-cors
"""
import cv2
import numpy as np
from keras.models import load_model
from flask import Flask, request, jsonify, g, abort, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)                               # Allow CORS (Cross Origin Requests)


# Read saved weights and name it model
def get_model():
    model = getattr(g, 'model', None)
    if model is None:
        model = g.model = load_model('face_recognition_weights.h5')
    return model

def classify(file_path):
    """
    classify a face image
    :param file_path: path of face image
    :return: classification results label and confidence
    """
    
    # Image dimensions that the model expects
    img_height, img_width, num_channel = 224, 224, 3
    mean_pixel = np.array([104., 117., 123.]).reshape((1, 1, 3))

    # TODO: Use opencv to read and resize image to standard dimensions
    img = cv2.imread(file_path)
    img = cv2.resize(img, (224,224))

    # TODO: Subtract mean_pixel from the image store the new image in 
    # a variable called 'normalized_image'
    normalized_image = img - mean_pixel
    
    # Turns shape of (2,) to (1,2)
    expanded_image = np.expand_dims(normalized_image, axis=0)

    # TODO: Use network to predict x, get label and confidence of prediction
    # Label is a number, which corresponds to the same number you give to 
    # the folder when you organized data
    
    # TODO: Get the model.
    model = get_model()
    
    # TODO: Use network to predict the 'image_to_be_classified' and
    # get an array of prediction values
    # Note: model.predict() returns an array of arrays ie. [[classes]]
    predictions = model.predict(expanded_image)[0]
    
    # TODO: Get the predicted label which is defined as follows:
    # Label = the index of the largest value in the prediction array
    # This label is a number, which corresponds to the same number you 
    # give to the folder when you organized data
    # Hint: np.argmax
    label = np.argmax(predictions)
    
    # TODO: Calculate confidence according to the following metric:
    # Confidence = prediction_value / sum(all_prediction_values)
    # Be sure to call your confidence value 'conf'
    # Hint: np.sum()
    label_value = predictions[label]
    total = np.sum(predictions)
    conf = label_value/total
    
    labels_to_names = {17: 'Simon',
                       18: 'Tyler'}

    if(label in labels_to_names):
        name = labels_to_names[label]
    else:
        name = "Unknown"
    
    prediction = {'label': name,
                  'confidence': float(conf)}   #Convert to be JSON serializable

    return prediction

@app.route('/')
def index():
    """
    Handles sending the webcam tool.
    """
    return send_from_directory('.','index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    """Receives an image, classifies the image, and responds with the label."""
    
    image = None
    
    # This extracts the image data from the request 
    if(request.method == 'POST'):
        
        if('image' not in request.form):
            print(request.form)
            abort(400)    
        image = request.form['image']
        data = {'data':'foo'}
        
    starter = image.find(',')
    image_data = image[starter+1:]
    
    # Path where the image will be saved
    temp_image_name = 'image.jpg'
    
    # Decodes the image data and saves the image to disk 
    with open(temp_image_name, 'wb') as fh:
        fh.write(image_data.decode('base64'))
        
    # TODO: Call classify to predict the image and save the result to a 
    # variable called 'prediction'
    prediction = classify(temp_image_name)
    
    # Converts python dictionary into JSON format
    prediction_json = jsonify(prediction)
    
    # Respond to the request (Send prediction back to Pi)    
    return prediction_json
        
def main():
    app.run(host='0.0.0.0', port=8080, threaded=False, debug=True)


if(__name__ == "__main__"):
    main()
