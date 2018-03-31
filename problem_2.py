# problem_2.py
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
#%matplotlib inline
#-------------------------------------------------------------------------------
def dist(x1, y1, x2, y2):
    return np.power(np.power(x1 - x2, 2) + np.power(y1 - y2, 2),0.5)
#-------------------------------------------------------------------------------
def log_image(image):
    return 20 * np.log(np.abs(image + 1))
#-------------------------------------------------------------------------------
def fourier_image(image):
    (height, width) = image.shape
    fft_image = np.fft.fft2(image, [height*2,width*2])
    #shift_fft = np.fft.fftshift(fft_image)
    return fft_image
#-------------------------------------------------------------------------------
def butter_low_pass(image, n, percent):
    # get dimentions from image
    (height, width) = image.shape
    # cut 
    D0 = 0.1 * height #D0 = 2.0 * 0.05 * height
    hhp = np.zeros((2 * height, 2 * width), dtype = np.float)
    # define filters values
    for i in range(2*height):
        for j in range(2*width):
            if(i != height and j != width):
                hhp[i][j] = 1/(1+(D0/dist(i - height, j - width, 0, 0))**(2*n))
    return hhp 
#-------------------------------------------------------------------------------
def butter_high_pass(hhp, a, b):
    hfe = a + b * hhp
    return hfe
#-------------------------------------------------------------------------------
def plot_image(image, title):
    plt.imshow(image, cmap = 'gray')
    plt.title(title)
    plt.axis('off')
#-------------------------------------------------------------------------------
def stretch_intensity(image):
    (h,w) = np.shape(image)
    matrix_image = image[:int(h/2),:int(w/2)]
    #new_image = matrix_image/255 
    min_value = np.min(matrix_image)
    max_value = np.max(matrix_image)
    new_image = abs((matrix_image - min_value)*255/max_value)
    #print(new_image)
    return new_image
#-------------------------------------------------------------------------------
def process_2(image):
    # Create a high-pass Butterworth filter with a frequency domain 
    # dimension of 2 times the size of the input image.
    hhp = butter_low_pass(image, 2, 0.05)
    # Create a high-frequency emphasis filter based on the 
    # Butterworth low pass filter
    hfe = butter_high_pass(hhp, 0.5, 2)
    # Get fft from original image
    fft_image = fourier_image(image)
    shift_fft = np.fft.fftshift(fft_image)
    # Multiply hfe filter and fft image 
    filtered_fft = np.fft.ifft2(np.multiply(hfe,fft_image))
    # resize and strech image_values
    stretch_image = stretch_intensity(filtered_fft)
    # Equalized histogram image
    equalized_img = cv2.equalizeHist(np.array(stretch_image,dtype=np.uint8))
    # Display original image 
    plt.figure(figsize = (12,12))
    plt.subplot(1,2,1)
    plt.imshow(image, cmap = 'gray')
    plt.title('Original image')
    plt.axis('off')
    # Display FFT image
    plt.subplot(1,2,2)
    #use a logaritmoc transformation to see better the fourier image
    plt.imshow(log_image(shift_fft), cmap = 'gray')
    plt.title('FFT image')
    plt.axis('off')
    # display filter 
    plt.figure(figsize = (12,12))
    plt.subplot(1,2,1)
    plt.imshow(hhp, cmap = 'gray')
    plt.title('Butterworth High Pass Filter')
    plt.axis('off')
    #
    plt.subplot(1,2,2)
    plt.imshow(hfe, cmap = 'gray')
    plt.title('High Emphasis Frequency Filter')
    plt.axis('off')
    # Filtered image 
    plt.figure(figsize = (12,12))
    plt.subplot(1,2,1)
    plt.imshow(stretch_image, cmap = 'gray')
    plt.title('Filtered Image')
    plt.axis('off')
    # Equalized-Filtered image 
    plt.subplot(1,2,2)
    plt.imshow(equalized_img, cmap = 'gray')
    plt.title('Equalized image')
    plt.axis('off')    
    # show all 
    plt.show() 
    
#------------------------------------------------------------------------------
def main2():
    # define 
    # load image 
    filename = 'chest.jpg'
    image = cv2.imread('media/' + filename, cv2.IMREAD_GRAYSCALE)
    process_2(image)
#------------------------------------------------------------------------------
main2()