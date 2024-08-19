import numpy as np
from scipy import ndimage
# DEFINE THE GUASSIAN KERNEL
def guassian_kernel(size , sigma) : 
    size = int(size) // 2 
    x , y = np.mgrid[-size:size+1 , -size:size+1]
    g = (1/(2.0*np.pi*sigma**2))*np.exp(-(x**2 + y**2)/(2.0*sigma**2))
    return g


# GRADIENT CALCULATION 
def sobel_filter(img) : 
    kx ,ky = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) , np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    ix , iy = ndimage.filters.convolve(img , kx) , ndimage.filters.convolve(img , ky)
    G = np.hypot(ix, iy)
    G = G / G.max() * 255
    theta = np.arctan2(iy, ix)
    
    return (G , theta)


# DEFINE NON MAXIMUM SUPRESSION 
def non_max_sup(img , D) : 
    m , n = img.shape 
    z = np.zeros_like(img)
    angle = 180*D/np.pi
    angle[angle < 0] += 180
    for i in range(1 , m-1) : 
        for j in range(1 , n-1 ) : 
            try : 
                q = 255 
                r = 255
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]
                    
                    
                if (img[i,j] >= q) and (img[i,j] >= r):
                    z[i,j] = img[i,j]
                else:
                    z[i,j] = 0
            except IndexError as e:
                print('Index error')
    
    return z 

# DEFINE THRESHOLD 
def threshold(img , lowThresholdRatio = 0.05 , hightThresholdRatio=0.1) : 
    highThreshold = img.max() * hightThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    
    m , n = img.shape 
    res = np.zeros_like(img)
    weak = np.int32(25)
    strong = np.int32(255)
    for i in range(m) : 
        for j in range(n): 
            if img[i][j] > highThreshold : 
                res[i][j] = strong
            elif img[i][j] >= lowThreshold and img[i][j] <= highThreshold : 
                img[i][j] = weak
    
    return (res , weak , strong)
# DEFINE Edge Tracking by Hysteresis  
def hysteresis(img, weak =25, strong=255):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img
    
