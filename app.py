from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
# from skimage import io
import os
from PIL import Image
import sys
from ocr_model import decode_batch,getOCRModel,getImageData


app = Flask(__name__)

MODEL_PATH = 'model_artif-serv/model_artif/'
def weight_variable(name, shape):   
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name, initializer=initial)

def bias_variable(name, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, initializer=initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 

class Model(object):
    xxx = 0

def build_model(pixel):
    x_placeholder = tf.placeholder(tf.float32, shape=[None, pixel])
    # y_placeholder = tf.placeholder(tf.float32, shape=[None, LABEL_COUNT])

    x_image = tf.reshape(x_placeholder, [-1, 64, 128, 1])
    # Convolution Layer 1
    W_conv1 = weight_variable("w1", [3, 3, 1, 32])
    b_conv1 = bias_variable("b1", [32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # Convolution Layer 2
    W_conv2 = weight_variable("w2", [2, 2, 32, 64])
    b_conv2 = bias_variable("b2", [64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # Convolution Layer 3
    W_conv3 = weight_variable("w3", [2, 2, 64, 128])
    b_conv3 = bias_variable("b3", [128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    # Dense layer 1
    h_pool3_flat = tf.reshape(h_pool3, [-1, 8*16*128])
    W_fc1 = weight_variable("w4", [8*16*128, 500])
    b_fc1 = bias_variable("b4", [500])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)   
    # Dense layer 2
    W_fc2 = weight_variable("w5", [500, 500])
    b_fc2 = bias_variable("b5", [500])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)  
    # Output layer
    W_out = weight_variable("w6", [500, 4])
    b_out = bias_variable("b6", [4])
    
    output = tf.matmul(h_fc2, W_out) + b_out
    
    model = Model()
    model.x_placeholder = x_placeholder
    # model.y_placeholder = y_placeholder
    model.output = output
    
    return model

def load_model(pixel):
    global model,session
    model = build_model(8192)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, os.path.join(os.getcwd() + "\\" + MODEL_PATH, "model"))
    # return model


    # g = tf.Graph()
    # with g.as_default():
    #     session = tf.InteractiveSession()
    #     saver = tf.train.Saver()
    #     saver.restore(session, os.path.join(os.getcwd() + "\\" + MODEL_PATH, "model"))
        
    #     return model
        # session.close()
        # predictions = model.output.eval(session=session, feed_dict={model.x_placeholder: X2_test[ids]})
        # 


# you can then reference this model object in evaluate function/handler


def LoadImage(path):
    return io.imread(path)[:,:] / 255.

def LoadData(paths):
    xs = []
    xs.append(LoadImage(paths))
    
    return np.array(xs)




# The request method is POST (this method enables your to send arbitrary data to the endpoint in the request body, including images, JSON, encoded-data, etc.)
@app.route('/getLicenseImage', methods=["POST"])
def evaluate():
    # """"Preprocessing the data and evaluate the model""""
    # TODO: data/input preprocessing
    # eg: request.files.get('file')
    # eg: request.args.get('style')
    # eg: request.form.get('model_name')
    global model,session
    load_model(8192)
    Imagefile = request.files['file']
    fileName = os.path.join(os.getcwd() + "\\" + Imagefile.filename)
    imageFormat = Imagefile.filename
    Imagefile.save(fileName)
    basewidth = 128 # MNIST image width
    img1 = Image.open(fileName)
    hsize = 64
    img = img1.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save("x" + imageFormat)
    X_test = LoadData("x" + imageFormat)
    pixel = X_test.shape[1]*X_test.shape[2]
    
    if len(X_test.shape) > 3 : 
        X_test = np.dot(X_test[...,:3], [0.299, 0.587, 0.114])
    X2_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
    # TODO: model evaluation
    # eg: prediction = model.eval()
    predictions = model.output.eval(session=session, feed_dict={model.x_placeholder: X2_test})
    session.close()
    # TODO: return prediction
    # eg: return jsonify({'score': 0.95})
    
    predictions = (predictions+1) * (img1.size[0]/2, img1.size[1]/2, img1.size[0]/2, img1.size[1]/2) 
    predictions = predictions.tolist()
    image = Image.open(fileName)
    for box in predictions:
        box1= tuple(box)
        ic = image.crop(box1)
        ic.save("test" + imageFormat)
    ocrModel,ocrSession = getOCRModel()
    net_inp = ocrModel.get_layer(name='the_input').input
    net_out = ocrModel.get_layer(name='softmax').output
    img = getImageData("test" + imageFormat)
    img = np.expand_dims(img, 0)
    img = img.T
    X_data = np.ones([1, 128, 64,1])
    X_data[0] = img
    net_out_value = ocrSession.run(net_out, feed_dict={net_inp:X_data})
    pred_texts = decode_batch(net_out_value)
    print(pred_texts)

    return jsonify({'number': pred_texts})

# The following is for running command `python app.py` in local development, not required for serving on FloydHub.


if __name__ == "__main__":
    print("* Starting web server... please wait until server has fully started")
    port = int(os.getenv("PORT", 5000)) 
    # app.run(host="0.0.0.0",port=port) #run app in debug mode on port 5000
    # app.run(port=port)
    if len(sys.argv)>=2:
        app.run(host=sys.argv[1], port=port)
    else:
        app.run(host='localhost', port=port)
    
   