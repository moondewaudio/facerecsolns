import time,cv2, base64, requests

    
def request_from_server():
    URL = "http://52.36.100.249:8080/predict"
    temp_image_name = 'test_image.jpg'

    # Reopen image and encode in base64
    image = open(temp_image_name, 'rb') #open binary file in read mode
    image_read = image.read()
    image_64_encode = base64.encodestring(image_read)
     
    # Defining a params dict for the parameters to be sent to the API
    PARAMS = {'file':image_64_encode}
     
    # Dending get request and saving the response as response object
    r = requests.get(url = URL, params = PARAMS)
     
    # Print response
    data = r.json()

    return data


def main():
    response = request_from_server()
    print(response)
if __name__ == '__main__':
    main()
