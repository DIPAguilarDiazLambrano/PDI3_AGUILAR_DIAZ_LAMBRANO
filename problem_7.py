import cv2
import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
#-------------------------------------------------------------------------------
def plot_image(image, title):
    plt.imshow(image, cv2.COLOR_HSV2RGB)
    plt.title(title)
    plt.axis('off')
#-------------------------------------------------------------------------------
def process(img, img_bkg):   
    # take dimentions from Hiro's image
    heigh, width = img.shape[:2] 
    # resize dimentions to datacenter image 
    img_bkg = cv2.resize(img_bkg,(width, heigh), interpolation = cv2.INTER_CUBIC)
    # Converts an image from one color space to HSV format.
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_bkg_hsv = cv2.cvtColor(img_bkg, cv2.COLOR_BGR2HSV)
    # define range of green color in HSV
    lower_green = np.array([50, 10, 50]) 
    upper_green = np.array([60, 255, 255]) 
    # Threshold the HSV image to get only green colors 
    green_mask = cv2.inRange(img_hsv, lower_green, upper_green) #
    # Threshold the HSV image to get only green colors
    img_fg = cv2.bitwise_and(img_hsv, img_hsv, mask= 255 - green_mask)
    # Threshold the HSV image to get only colors differents to green 
    img_bg = cv2.bitwise_and(img_bkg_hsv, img_bkg_hsv, mask = green_mask)
    # Add the images 
    img_out = cv2.add(img_fg, img_bg)
    # Show orignal image
    plt.figure(figsize=(10, 10))
    plt.subplot(2,1,1)
    plt.imshow(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)); 
    plt.axis('off');
    plt.title('Original Image')
    # display background image
    plt.subplot(2,1,2)
    plt.imshow(cv2.cvtColor(img_bkg, cv2.COLOR_BGR2RGB)); 
    plt.axis('off');
    plt.title('Background Image')
    # Show mask
    plt.figure(figsize=(10, 10))
    plt.subplot(2,1,1)
    plt.imshow(green_mask, cmap = 'gray');
    plt.axis('off')
    plt.title('Mask Frame')
    # Display final result
    plt.subplot(2,1,2)
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_HSV2RGB)); 
    plt.axis('off')
    plt.title('Final Result')
    plt.show()
#-------------------------------------------------------------------------------    
def main7():
    # load hiro and datecenter image 
    img = cv2.imread('media/hiro.jpg',cv2.IMREAD_COLOR)
    img_bkg = cv2.imread('media/datacenter.jpg',cv2.IMREAD_COLOR)
    process(img, img_bkg)
#-------------------------------------------------------------------------------
main7() 