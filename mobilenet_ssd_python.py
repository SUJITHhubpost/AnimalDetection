#Import the neccesary libraries
import argparse
from flask import jsonify
import cv2
import numpy as np
import pandas as pd

# Classes
df_class = pd.read_csv('class.csv')

classes = df_class['class'].to_dict()

def main(video):



    # construct the argument parse 
    # parser = argparse.ArgumentParser(
    #     description='Script to run MobileNet-SSD object detection network ')
    # parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
    # parser.add_argument("--prototxt", default="graph.pbtxt",
    #                                 help='Path to text network file: '
    #                                     'MobileNetSSD_deploy.prototxt for Caffe model or '
    #                                     )
    # parser.add_argument("--weights", default="frozen_inference_graph.pb",
    #                                 help='Path to weights: '
    #                                     'MobileNetSSD_deploy.caffemodel for Caffe model or '
    #                                     )
    # parser.add_argument("--thr", default=0.4, type=float, help="confidence threshold to filter out weak detections")
    # args = parser.parse_args()

    # Labels of Network.
    # classNames = classes

    # Open video file or capture device. 
    # if args.video:
    #     cap = cv2.VideoCapture(args.video)
    # else:
    #     cap = cv2.VideoCapture(0)

    cap = cv2.VideoCapture(video)
    #Load the Caffe model 
    net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction

        frame, labels = singleDetection(frame)
                
        
                
        # Show the image with a rectagle surrounding the detected objects 
        # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        # cv2.imshow("frame", frame)
        # if cv2.waitKey(1) >= 0:  # Break with ESC 
        #     break
        
    # cap.release()
    # cv2.destroyAllWindows()
    
    return labels

def update(image):
    	# if the background model is None, initialize it
		
		bg = image.copy().astype("float")
  
		return


def singleDetection(frame):
    
    labels = []
    
    net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')
    
    net.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))
    # MobileNet requires fixed dimensions for input image(s)
    # so we have to ensure that it is resized to 300x300 pixels.
    # set a scale factor to image because network the objects has differents size. 
    # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
    # after executing this command our "blob" now has the shape:
    # (1, 3, 300, 300)
    # blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    # #Set to network the input blob 
    # net.setInput(blob)
    # #Prediction of network
    detections = net.forward()
    rows, cols, channels = frame.shape
    #Size of frame resize (300x300)
    # cols = frame_resized.shape[1] 
    # rows = frame_resized.shape[0]
    
    for detection in detections[0,0]:
        
        score = float(detection[2])
        
        if score > 0.45:
            
            
            class_name = classes[detection[1]-1]
            
            label = str(class_name) + " : " + str(round(score * 100, 2)) + "%"
            print(label)

            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] *  rows

            #draw a red rectangle around detected objects
            
            
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)

                
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (36,255,12), 4)
            
            labelSize=cv2.getTextSize(label,cv2.FONT_HERSHEY_COMPLEX,1,2)
            
            _x1 = int(left)
            _y1 = int(top)#+int(labelSize[0][1]/2)
            _x2 = int(left)+labelSize[0][0]
            _y2 = int(top)-int(labelSize[0][1])
            cv2.rectangle(frame,(_x1,_y1),(_x2,_y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,label,(int(left),int(top)),cv2.FONT_HERSHEY_COMPLEX,1.1,(255, 255, 255),2)
            
            cv2.imwrite(r'static/out/out.jpg',frame)
            
            labels.append(label)
            
            
            
    return frame, labels
        
if __name__ == "__main__":
    main(0)
