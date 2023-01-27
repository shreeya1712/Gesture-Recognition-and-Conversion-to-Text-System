
# -----------------import necessary modules and datasets--------------

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
from os import listdir
from tkinter import *
from numpy import zeros
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from cv2 import *
from PIL import Image
import os


win = Tk()
win.geometry("700x700")
win.title("Gesture Recognizer System")
gesturebg = PhotoImage( file = "gesturebg.png")
label1 = Label(win, image=gesturebg)
label1.place(x=250, y=150)

label = Label(win , text = "Gesture Recognizer System" ,  bg = "white" , fg = "white")
label.config(font=('Helvetica bold', 40))
label.config(bg = "#202A44")
label.place(x = 700, y = 700)
label.pack(pady = 40)

btn = Button(win, text='Exit', command= win.destroy)

btn.place(x=350, y=450)


#--------------------------image acquisition------------------------------

# background image acquisition
vid = cv2.VideoCapture(0)
i = 0
while i < 2:

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    # Display the resulting frame
    # cv2.imshow('Capturing Background', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    i = i+1
    cv2.imwrite('bgimg.jpg', frame)

# After the loop release the cap object
vid.release()
# Destroy all the windows
bgpicture = Image.open('bgimg.jpg')
bg = bgpicture.crop((100,100,600,600))


# gesture image acquisition
vid = cv2.VideoCapture(0)
i = 0
while True:

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    cv2.putText(frame, 'insert gesture here', (50, 50), 1, 2, cv2.LINE_4)
    cv2.rectangle(frame, (100, 100), (600, 600), (0,255,0), 2)

    # Display the resulting frame

    cv2.imshow('Gesture Capturer', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    i = i+1
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('img.jpg', frame)
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()

cv2.destroyAllWindows()
picture = Image.open('img.jpg')
picture = picture.crop((100,100,600,600))

# final gesture image acquired
bg.show()
picture.show()
# final background image acquired
pc = np.asarray(picture)
bg = np.asarray(bg)

# ---------------------Background removal function--------------------


def bgremove(bg ,pc):
    ret, bg = cv2.threshold(bg, 127, 255, cv2.THRESH_BINARY)
    ret, pc = cv2.threshold(pc, 127, 255, cv2.THRESH_BINARY)
    outimg = np.zeros((500,500))
    tol = 0.03
    for ch in range(500):
        for cw in range(500):
            impix = pc[ch,cw]
            bgpix = bg[ch,cw]
            comparison1 = (impix == bgpix)
            comparison2 = (bgpix > impix) & (bgpix <= impix + tol)
            comparison3 = (bgpix < impix) & (bgpix >= impix - tol)
            if comparison1.all():

                outimg[ch, cw] = 0
            elif comparison2.all():
                outimg[ch, cw] = 0

            elif comparison3.all():
                outimg[ch, cw] = 0

            else:
                outimg[ch,cw] = 255

#
#
    return outimg
#
# -----------------------------------------------------------


outimg = bgremove(bg,pc)
outimg = np.asarray(outimg, dtype = np.int32)
out_image = Image.fromarray(outimg)
out_image.show()

# ----------------Loading the datasets------------------------
train_data_path = '/Users/shreeyarao/Desktop/internship_a1logic/data/train'
test_data_path = '/Users/shreeyarao/Desktop/internship_a1logic/data/test'
# path to the image folder

# ------------function definitions-----------------------------------


def load_images_from_folder(folder):

    # function that reads images from our image folder and converts them into matrix format and stores them into a list
    img_data = []
    # will store vectorized images
    img_label = []
    # will store the labels
    for num in os.listdir(folder):

        # enter the train folder
        subdir = folder+'/'+num
        for img in os.listdir(subdir):
            # entered the 0,1,2,3 folder
            im = cv2.imread(subdir+'/'+img)  # read image with filename
            im = cv2.resize(im, (50,50))
            ret, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
            img_data.append(im)
            img_label.append(num)
            # we now have two lists, the first list contains
            # all the images in their pixel format and the second list contains the corresponding image labels
    return img_data, img_label

# ------------------------data pre processing and transformations-------------------

# loading the training and testing data


train_img_data, train_img_label = load_images_from_folder(train_data_path)
test_img_data, test_img_label = load_images_from_folder(test_data_path)

# converting into array and changing the data type to float
train_img_data = np.asarray(train_img_data, dtype=np.float64)
train_img_label = np.asarray(train_img_label, dtype=np.float64)
test_img_data = np.asarray(test_img_data, dtype=np.float64)
test_img_label = np.asarray(test_img_label, dtype=np.float64)

# normalizing the data
train_img_data /= 255
test_img_data /= 255

# reshaping and flattening the training and testing image data
train_img_data = train_img_data.reshape((5400, (50*50*3) * 1))
test_img_data = test_img_data.reshape((1800, (50*50*3) * 1))
# initially had a 50x50x3 image and converted it into array with each value having 50*50*3 rows and 1 column

# standardize the data using standard scaler. The idea is to get the idea to values such that the mean
# # and standard deviation of the data are 0 and 1 respectively.


# we now have the data with reduced dimensions and data such that mean and s.d are 0 and 1
# we can now start training different models


scaler = StandardScaler()
scaler.fit(train_img_data)
scaler.fit(test_img_data)

X_train = scaler.transform(train_img_data)
X_test = scaler.transform(test_img_data)

# ---------------------PERFORMING PCA------------------------------------------
#
# PCA is performed. This is used to perform dimensionality reduction in the data
pca = PCA(n_components=1)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)


# ---------------training and testing a Decision Tree model--------------------

clf = RandomForestClassifier(max_depth=1000, random_state=42)
clf.fit(X_train, train_img_label)
y_predict = clf.predict(X_test)
train_accuracy = clf.score(X_train, train_img_label)
print('train accuracy is ' + str(train_accuracy))
test_accuracy = clf.score(X_test, test_img_label)
print('test accuracy is ' + str(test_accuracy))


# ----------------------------------------------------------------------------

outimg = np.asarray(outimg, dtype=np.float64)
outimg /= 255
scaler.fit(outimg)
outimg = scaler.transform(outimg)
outimg= pca.fit_transform(outimg)
y_predic = clf.predict(outimg)
y_predic = np.asarray(y_predic, dtype = np.int32)
counts = np.bincount(y_predic)
print(counts)
ans = np.argmax(counts)




# -------------_Displaying result---------------
print('the number is ')
print(ans)

if ans == 1:
    p = Image.open('sign1.jpg')
    p.show()

if ans == 2:
    p = Image.open('sign2.jpg')
    p.show()

if ans == 2:
    p = Image.open('sign2.jpg')
    p.show()


if ans == 3:
    p = Image.open('sign3.jpg')
    p.show()

if ans == 4:
    p = Image.open('sign4.jpg')
    p.show()


if ans == 5:
    p = Image.open('sign5.jpg')
    p.show()


if ans == 0:
    p = Image.open('sign0.jpg')
    p.show()

win.mainloop()

