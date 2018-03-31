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
def blurred_hue(image):
    hsi_image = imageRGB2HSI(image)
    hue = hsi_image[0]
    h,w = hue.shape
    hsi_image[0] = cv2.blur(hue,(25, 25))    
    return imageHSI2RGB(hsi_image)
#-------------------------------------------------------------------------------
def blurred_sat(image):
    hsi_image = imageRGB2HSI(image)
    hue = hsi_image[1]
    hsi_image[1] = cv2.blur(hue,(25, 25)) 
    return imageHSI2RGB(hsi_image)
#-------------------------------------------------------------------------------
def main5():
    filename = 'media/squares.jpg'
    image = cv2.cvtColor(cv2.imread(filename, 1), cv2.COLOR_BGR2RGB)
    blur_hue = blurred_hue(image)
    blur_sat = blurred_sat(image)
    
    plt.figure()
    plt.imshow(image)
    plt.title('Original')
    plt.axis('off')
    #
    plt.figure()
    plt.imshow(blur_hue)
    plt.title('Blurred Hue image')
    plt.axis('off')
    #
    plt.figure()
    plt.imshow(blur_sat)
    plt.title('Blurred Saturation image')
    plt.axis('off')
    
    plt.show()
#-------------------------------------------------------------------------------
main5()