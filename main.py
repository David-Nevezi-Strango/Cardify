import cv2
import imutils
import numpy as np
import tensorflow as tf
from scipy import ndimage
import datetime
# from model import trainModel
# document_img = cv2.imread("id/id5.jpg")
# document_img = imutils.resize(document_img, width=460)

def rotateImg(img):
    #preprocess image for hough lines
    src = imutils.resize(img, width=460)
    gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.float32)/25
    gray = cv2.filter2D(gray,-1,kernel)
    edges = cv2.Canny(gray,400,600,apertureSize = 5)
    # cv2.imshow('image',edges)
    # cv2.waitKey(0)

    lines = cv2.HoughLines(edges,1,np.pi/180,15)
    angles = list()
    #iterate through lines to get a number of angles
    for i in range(8):
        for rho,theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)
            # cv2.line(src, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Rotate image to be horizontal
    # Using median angle due to inconsistency of dominant line detection
    median_angle = np.median(angles)
    rotated_img = ndimage.rotate(src, median_angle)

    # for some reason crop breaks the live detection
    # # Crop image to retain original resolution
    # h_rot, w_rot = rotated_img.shape[:2]
    # h, w = src.shape[:2]
    # dist_w, dist_h = (w // 2, h // 2)
    # center_rot = (w_rot // 2, h_rot // 2)
    # cropped_img = rotated_img[center_rot[0] - dist_h: center_rot[0] + dist_h,
    #               center_rot[1] - dist_w: center_rot[1] + dist_w]

    # # Display original image with the hough lines drawn
    # cv2.imshow("Detected lines", src)
    # # cv2.imwrite("detected.jpg", src)
    # cv2.waitKey(0)
    # # Display the rotated image
    # cv2.imshow("Rotated Image", rotated_img)
    # # cv2.imwrite("rotated.jpg", rotated_img)
    # cv2.waitKey(0)
    # # Display the cropped image
    # cv2.imshow("Cropped Image", cropped_img)
    # # cv2.imwrite("cropped.jpg", cropped_img)
    # cv2.waitKey(0)

    return rotated_img


def evaluate():
    #initialize app
    cap = cv2.VideoCapture(0)
    model = tf.keras.models.load_model("mobilenetv2ID.keras")
    while True:
        #get the current time for predictions saves
        currentTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ret, src = cap.read()
        #process the input image
        img_rot = rotateImg(src)
        img = cv2.resize(img_rot, (224,224))
        #give the image to the NN for inference
        imgpred = np.expand_dims(img, axis=0)  # Add a batch dimension
        pred = model.predict(imgpred).flatten()
        pred = tf.nn.sigmoid(pred)
        pred = tf.where(pred < 0.5, 0, 1)
        cv2.imshow("Camera", src)
        # print("pred: ", pred[0])
        if pred[0] == 1:
            #if the prediction says it is an ID card, save it
            print("ID Card detected! Saving it to PC!")
            try:
                #create prediction folder if it does not exist already
                os.mkdir("pred")
            except:
                pass
            filename = "pred/idpred_{date}.jpg".format(date=currentTime)
            # print(filename)
            #save the rotated image
            cv2.imwrite(filename, img_rot)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #if q button was pressed, exit
            cap.release()
            cv2.destroyAllWindows()
            # exit()
            break


if __name__ == "__main__":
    # rotateImg(document_img)
    evaluate()