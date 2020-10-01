# DarkNet-Obj-Training

Colab file copied from Roboflow and modified to suit.

Vision.py is the Python file that uses the tflite file downloaded in the Colab outputs. This then performs vision checking on a image (can be got from camera or dropped into a directory. The file resizes andpads the input file to the required size that the tflite model was trained at. Default is 512x512.

Used on a Rpi with Tensorflow 2.3 installed. Will need other dependicies installed.

Need to use the github repo from https://github.com/hunglc007/tensorflow-yolov4-tflite.git Run the Vision.py inside this directory as it uses other code in this github.

Big thanks to all repo's I've used to get this working :)



