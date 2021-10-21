##############################  Video Processing ####################################################


import cv2 
import imutils
import numpy as np
from google.colab.patches import cv2_imshow


def predict(imag):
  from keras.preprocessing import image 
  imag = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
  imag = cv2.resize(imag, (224, 224))
  x = image.img_to_array(imag)
  prediction = model.predict(np.expand_dims(x, axis=0))
  x1 = (prediction[1] + prediction[0])/2
  x1 = np.asscalar(x1)
  return x1


def preprocess(image):
  # resize it to have a maximum width of 400 pixels
  image = imutils.resize(image, width=400)
  blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
  net.setInput(blob)
  detections = net.forward()
  for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the prediction
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence threshold
    if confidence > 0.5:
      # compute the (x, y)-coordinates of the bounding box for the object
      box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
      (startX, startY, endX, endY) = box.astype("int")
      # draw the bounding box of the face along with the associated probability
      clone = image.copy()
      crop_img = clone[startY:endY, startX:endX]
      probability = predict(crop_img)
      text = "TrustWorth - {:.2f}%".format(probability * 100)
      y = startY - 10 if startY - 10 > 10 else startY + 10
      cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
      cv2.putText(image, text, (startX, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
  
  return image

def annotate_video():
  print("[INFO] loading model...")
  prototxt = 'deploy.prototxt'
  modelcv = 'res10_300x300_ssd_iter_140000.caffemodel'
  net = cv2.dnn.readNetFromCaffe(prototxt, modelcv)

  Video_in = "/content/test.mp4"
  video_out = "/content/annotatedtestfinal2.mp4"

  cap = cv2.VideoCapture(Video_in)
  writer = None
  while(cap.read()):
    ret, frame = cap.read()
    if not ret:
      break
    img = preprocess(frame)
    if writer is None:
      fourcc = cv2.VideoWriter_fourcc(*"XVID")
      writer = cv2.VideoWriter(video_out, fourcc, 30, (img.shape[1], img.shape[0]), True)
    writer.write(img)
  print("[INFO] cleaning up......")
  writer.release()



