import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
#%matplotlib inline
#-------------------------------------------------------------------------------
def false_color(image,T):
    (height, width) = image.shape
    out=np.zeros((height, width, 3), np.uint8)
    blues = np.where(image < T, 255, 0)
    yellows = 255 - blues
    out[:,:,0] = blues; out[:,:,1] = yellows; out[:,:,2] = yellows;
    return out
#-------------------------------------------------------------------------------
def main6(): 
    img = cv2.imread("media/weld_x-ray.jpg", cv2.IMREAD_GRAYSCALE)
    img_out = false_color(img,250)
    plt.figure(figsize = (12, 12))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Original Image')
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('False-color image')
#-------------------------------------------------------------------------------
main6()