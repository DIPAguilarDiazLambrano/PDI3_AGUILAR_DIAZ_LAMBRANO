# problem_1.py
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
#%matplotlib inline
#-------------------------------------------------------------------------------
def plot_image(image, title):
    plt.imshow(image, cmap = 'gray')
    plt.title(title)
    plt.axis('off')
#-------------------------------------------------------------------------------
def log_image(image):
    return 20 * np.log(np.abs(image + 1))
#-------------------------------------------------------------------------------
def fourier_image(image):
    fft_image = np.fft.fft2(image)
    shift_fft = np.fft.fftshift(fft_image)
    return shift_fft
#-------------------------------------------------------------------------------
def process_1(filename):
    # define a figure
    fig = plt.figure(figsize = (12,12))
    # load image
    image = cv2.imread('media/' + filename, cv2.cv2.IMREAD_GRAYSCALE)
    # display original image 
    plt.subplot(1, 2, 1)
    plot_image(image, 'Original image')
    # compute fourier image
    fft_image = fourier_image(image)
    # use a logaritmic transformation
    log_correction = log_image(fft_image)
    # display gamma image 
    plt.subplot(1, 2, 2)
    plot_image(log_correction, 'Spectrum image')
    # display title on figure
    #plt.suptitle('Image: ' + filename, fontsize = 16)   
#-------------------------------------------------------------------------------
def main1():
    image_names = ['face.png', 'blown_ic.png', 'test_pattern_blurring_orig.png', 
                   'translated_rectangle.png', 'rectangle.png']
    for filename in image_names:
        process_1(filename)
    plt.show()
#-------------------------------------------------------------------------------
main1()