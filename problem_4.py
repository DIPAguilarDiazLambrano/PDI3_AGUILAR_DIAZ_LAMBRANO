import numpy as np
import cv2
import matplotlib.pyplot as plt 
#%matplotlib inline
#-------------------------------------------------------------------------------
def angle(R, G, B):
    if (R == G and G == B):
        ang = 0
    else: 
        num = 0.5*(2 * R - G - B)
        den = np.sqrt(np.power((R - G),2) + (R - B)*(G - B))
        nd = num/den
        if (nd > 1):
            ang = 0
        elif (nd < -1):
            ang = 180
        else: 
            ang = np.degrees(np.arccos(nd))
    return ang
#-------------------------------------------------------------------------------
def rgb2hsi(pixelRGB):
    # Normalize pixels values 
    [R, G, B] = np.array(pixelRGB/255.0)
    # Get H value
    H = angle(R,G,B)
    if(B > G):
        H = 360 - H 
    # Get S value 
    pixel_sum = np.sum([R, G, B])
    if pixel_sum != 0:
        S = 1 - 3 * np.min([R, G, B])/pixel_sum
    else:
        S = 0
    # get I value
    I = pixel_sum/3
    return np.array([H, S, I])
#-------------------------------------------------------------------------------
def imageRGB2HSI(image):
    h, w = image.shape[:2]
    out_image = np.zeros(image.shape)
    for i in range(h):
        for j in range(w):
            out_image[i][j] = rgb2hsi(image[i][j])
    return out_image
#-------------------------------------------------------------------------------
def imageHSI2RGB(image):
    h, w = image.shape[:2]
    out_image = np.zeros(image.shape)
    for i in range(h):
        for j in range(w):
            out_image[i][j]  = hsi2rgb(image[i][j])
    return out_image
#-------------------------------------------------------------------------------
def hsi2rgb(pixelHSI):
    [H, S, I] = pixelHSI
    if (H >= 0 and H < 120):
        B = I*(1 - S)
        R = I * (1 + S * np.cos(np.radians(H))/np.cos(np.radians(60 - H)))
        G = 3*I - (R + B)
    elif (H >= 120 and H < 240):
        H = H - 120
        R = I * (1 - S)
        G = I * (1 + S * np.cos(np.radians(H))/np.cos(np.radians(60 - H)))
        B = 3*I - (R + G)
    else:
        H = H - 240
        G = I * (1 - S)
        B = I * (1 + S * np.cos(np.radians(H))/np.cos(np.radians(60 - H)))
        R = 3 * I - (G + B)
    pixelRGB = np.array([R, G, B])
    return pixelRGB
#-------------------------------------------------------------------------------
def equ_hist(image):
    #print(type(img))
    img = image.astype('uint8')
    # https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[img]
    return img2
#-------------------------------------------------------------------------------
def equ_intensity(image):
    hsi_image = imageRGB2HSI(image)
    intensity = 255 * hsi_image[:,:,2]
    hsi_image[:,:,2] = equ_hist(intensity)/255.0
    return imageHSI2RGB(hsi_image)
#-------------------------------------------------------------------------------    
def main4():
    filename = 'media/dark_fountain.jpg'
    img = cv2.imread(filename, 1)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    equalized = equ_intensity(image)
    #
    plt.figure()
    plt.imshow(image)
    plt.title('Original')
    plt.axis('off')
    #
    plt.figure()
    plt.imshow(equalized)
    plt.title('Equalized image')
    plt.axis('off')
    plt.show()
#-------------------------------------------------------------------------------
main4()