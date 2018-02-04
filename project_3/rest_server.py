import cv2
import numpy as np
from keras.models import load_model
from flask import Flask, request, jsonify

app = Flask(__name__)

# Read saved weights and name it model
model = load_model('side_hoe_number_2.h5')

def classify(file_path):
    """
    classify a face image
    :param file_path: path of face image
    :param model: model to use
    :return: classification results label and confidence
    """
    img_height, img_width, num_channel = 224, 224, 3
    mean_pixel = np.array([104., 117., 123.]).reshape((1, 1, 3))

    # Use opencv to read and resize image to standard dimensions
    
    img = cv2.imread(file_path)
    img = cv2.resize(img, (224,224))

    # TODO: subtract mean_pixel from that image, name the final image as new_img
    #CHECK
    new_img = img - mean_pixel
    
    # Turns shape of (2,) to (1,2)
    x = np.expand_dims(new_img, axis=0)

    # TODO: use network to predict x, get label and confidence of prediction
    # TODO: label is a number, which correspond to the same number you give to the folder when you organized data
    # CHECK
    predictions = model.predict(x)[0]
    label = np.argmax(predictions)
    
    
    maxValue = predictions[label]
    total = np.sum(predictions)
    conf = maxValue/total
    
    labels_to_names = {17: 'Simon',
              18: 'Tyler'}

    if label in labels_to_names:
        name = labels_to_names[label]
    else:
        name = "Unknown"
    
    prediction = {'label': name,
                  'confidence': conf}

    return prediction

@app.route('/predict')
def save_image():
    image = request.args.get('file')
    starter = image.find(',')
    image_data = image[starter+1:]
    temp_image_name = 'image.jpg'
    with open(temp_image_name, 'wb') as fh:
        fh.write(image_data.decode('base64'))
        
    return jsonify(classify(temp_image_name))
        
def main():
    app.run(host='0.0.0.0', port=8196)


if __name__ == "__main__":
    main()
