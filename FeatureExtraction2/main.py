import cv2
import numpy as np
import scipy
import _pickle as pickle
import random
import os
from os import listdir
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl




#Oriented FAST and Rotated BRIEF (ORB) Feature Extraction
def image_detect_and_compute(detector, img_name):
    """Detect and compute interest points and their descriptors."""
    img = cv2.imread(os.path.join(folder + img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(img, None)
    all_kp.append([kp])
    return img, kp, des

#Oriented FAST and Rotated BRIEF (ORB) Feature Extraction
def image_detect_and_compute2(img_name):
    """Detect and compute interest points and their descriptors."""
    sift = cv2.xfeatures2d.SIFT_create()
    img = cv2.imread(os.path.join(folder + img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(img, None)
    all_kp.append([kp])
    return img, kp, des

all_kp = []
folder = 'Pictures/'

#Image Preprocessing (Resize)
"""
Bilder müssen alle in die gleiche Shape gebracht werden (photo = load_img(folder + file, target_size=(200, 200)))
tbc: Feature extraction mit numpy arrays?
"""



for file in listdir(folder):
    print(file)
    orb = cv2.ORB_create()
    image_detect_and_compute(orb,file)
    image_detect_and_compute2(file)

df = pd.DataFrame(all_kp)
df.to_excel('Features.xlsx', sheet_name='new_sheet_name')

