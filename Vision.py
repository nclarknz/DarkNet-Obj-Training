import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
import time
import os
import glob
import datetime
from socketIO_client_nexus import SocketIO, LoggingNamespace
from picamera import PiCamera
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from statistics import mode, StatisticsError
from requests.exceptions import ConnectionError
from zipfile import ZipFile


# Setup where the sockets go
hostn = '192.168.0.1'
hostp = '3000'
cardid = ""
# Setup the TF model locations and scoring for results
weights = '/code/vision/tensorflow-yolov4-tflite/checkpoints/yolov4-tiny-512v2.tflite'
score = 0.4
iou = 0.45
labels = '/code/vision/tensorflow-yolov4-tflite/obj.names'
model = 'yolo4'
imagedir = '/code/vision/testimgs/'
zipdir = '/code/vision/zipped/'
tiny = True
size = '512x512'
saveresults = True


# Response from SocketIO server
def on_connect():
    print('connect')

def on_disconnect():
    print('disconnect')

def on_reconnect():
    print('reconnect')
	
def drgStartrec(*args):
    print('drg Start Rec')
    #take_photos(camera)
	# Take photos
	# Save them in the correct folder
	# Then process the images
  
    start_time = time.monotonic()
    labellist = LoadLabels(labels)
  
    # print('Input Details')
    # print(input_details)
    # print('Output Details')
    # print(output_details)
	# Now process all images in the directory and return result for each
    for filename in glob.glob(imagedir + '/*.jpg'):
        start_timesing = time.monotonic()
        # print('Process Image - ' + filename)
        result = ProcessImages(filename,interpreter, input_width, input_height,input_details,output_details,labellist,saveresults)
        elapsed_mssing = round((time.monotonic() - start_timesing) ,3)
        print('---------  Elapsed Time ' + str(elapsed_mssing) + " secs")
    #zippedfile = ZipFiles(result)
    #DelFiles()
    #distance2 = 0
    #resultstr = result + '#' + zippedfile + '#' + str(distance2)
    #try:
      #socketIO = SocketIO('192.168.0.1', 3000, LoggingNamespace)
      #socketIO.emit('result', resultstr) 
      #socketIO.wait(seconds=1)
    #except ConnectionError:
    #     print('The server is down. Try again later.')
    #except:
    #     print('Error With connection to server sending result')
    elapsed_ms = round((time.monotonic() - start_time) ,3)
    print('Elapsed Time ' + str(elapsed_ms) + " secs")
	
def take_photos(camera):
    for i in range(2):
        camera.capture('/code/testimgs/image{0:04d}.jpg'.format(i))
    print('Taken Photos')

#Load the labels into array to say which boxtype is present
def LoadLabels(path):
    lines=[]
    with open(path, 'r') as fh:
        for line in fh:
           line = line.strip()
           lines.append(line)
    return lines

def ZipFiles(result):
    t = (datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
    t = t.replace(':','')
    t = t.replace('.','')
    zipname = (str(result) + '_' + t + '.zip')
    with ZipFile(zipdir + '/' + zipname,'w') as zipObj:
        for filename in glob.glob(imagedir + '/*.jpg'):
                zipObj.write(filename)
    return zipname
	
def DelFiles():
    for filename in glob.glob(imagedir + '/*.jpg'):
        os.remove(filename)
    print('Deleted')
	
def resizeAndPad(img, size, padColor=0):
    h, w = img.shape[:2]
    sh, sw = size
    if h > sh or w > sw: # shrinking image
      interp = cv2.INTER_AREA

    else: # stretching image 
      interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = float(w)/h 
    saspect = float(sw)/sh

    if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
       new_h = sh
       new_w = np.round(new_h * aspect).astype(int)
       pad_horz = float(sw - new_w) / 2
       pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
       pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
       new_w = sw
       new_h = np.round(float(new_w) / aspect).astype(int)
       pad_vert = float(sh - new_h) / 2
       pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
       pad_left, pad_right = 0, 0

# set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img
	
def ProcessImages(filename,interpreter,input_width,input_height,input_details,output_details,labellist,saveresults):
    start_timing = time.monotonic()
    original_image = cv2.imread(filename)
    #original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
    #image_data = cv2.resize(original_image, (input_width, input_height))
    image_data = resizeAndPad(original_image, (512,512),0)
    imgresize = image_data
    cv2.imwrite('resize.jpg',image_data)
    image_data = image_data / 255.
    # image_data = image_data[np.newaxis, ...].astype(np.float32)
    images_data = []
    for i in range(1):
           images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], images_data)
    interpreter.invoke()

    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_height, input_width]))
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=150,
        max_total_size=150,
        iou_threshold=iou,
        score_threshold=score
    )
  
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

    classlist = []
    class_u8 = tf.cast(classes, dtype=tf.uint8)
    scorelist = scores.numpy().tolist()[0]
    class_list = class_u8.numpy().tolist()[0]
	   
    for index in class_list:
        if scorelist[index] > score:
            classlist.append(class_list[index])
            classscore = round(scorelist[index] * 100,1)
    if(len(classlist) > 0):
        try:
            finalresult = mode(classlist)
            finalboxtype = labellist[finalresult]
            print('Image is type ' + str(finalboxtype) + ' box type for ' + filename)
        except StatisticsError:
            finalboxtype = 'More1'
            print('More then 1 box type found for ' + filename)
    else:
        finalboxtype = 'NoBox'
        print('No Object Detected in ' + filename)
    if saveresults:
       imageres = utils.draw_bbox(imgresize, pred_bbox)
       #imageres = Image.fromarray(imageres.astype(np.uint8))
       #imageres = cv2.cvtColor(np.array(imageres), cv2.COLOR_BGR2RGB)
       head, tail = os.path.split(filename)
       cv2.imwrite(imagedir + '/savedresults/' + finalboxtype + '_' +  tail, imageres)
    
    return finalboxtype

#try:
#Start the socket client to listen
   #socketIO = SocketIO('192.168.0.1', 3000, LoggingNamespace)
   #socketIO.on('connect', on_connect)
   #socketIO.on('disconnect', on_disconnect)
   #socketIO.on('reconnect', on_reconnect)
   # Socket Listen for control box sending command that start button used
   #socketIO.on('start', drgStartrec)
   #socketIO.wait(seconds=1)
#except ConnectionError:
#   print('The server is down. Try again later.')
#except:
#   print('Error With connection to server startup')
   
# Start the camera
#camera = PiCamera()
#camera.resolution = (1024, 768)
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(tiny)
input_width, input_height = utils.input_size(size)

interpreter = tf.lite.Interpreter(model_path=weights)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()

def main(_argv):
    drgStartrec()
   

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
