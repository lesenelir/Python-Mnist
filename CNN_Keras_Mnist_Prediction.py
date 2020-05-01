import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as processimage


model = load_model('./CNN_Own_Mnist.h5')

class MainPredictImg(object):
    def __init__(self):
        pass


    def pred(self, filename):
        # numpy array
        pred_img = processimage.imread(filename)  # read image
        pred_img = np.array(pred_img)  # transfer to array np
        pred_img = pred_img.reshape(-1, 28, 28, 1)  # reshape into network needed shape
        prediction = model.predict(pred_img)


def main():
    Predict = MainPredictImg()
    res = Predict.pred('./picture/2.jpg')
    print("your number is: " + res)
