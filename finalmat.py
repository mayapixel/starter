import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- 1. HISTOGRAM ----------------
def compute_histogram(img):
    h, w = img.shape
    hist = np.zeros(256, dtype=int)
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1
    print(f"Histogram computed: {hist}")
    return hist

def plot_histogram(hist, title="Histogram"):
    plt.figure()
    plt.title(title)
    plt.xlabel("Gray Level")
    plt.ylabel("Frequency")
    plt.bar(np.arange(256), hist, width=1.0)
    plt.show()
    print(f"{title} plotted.")

# ---------------- 2. GRAY LEVEL MODIFICATION ----------------
def negative(img):
    out = 255 - img
    print(f"Original Image:\n{img}\nNegative Image:\n{out}")
    return out

# ---------------- 3. EDGE DETECTION ----------------
def sobel_edge(img):
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    h, w = img.shape
    Gx = np.zeros_like(img, dtype=float)
    Gy = np.zeros_like(img, dtype=float)
    for i in range(1,h-1):
        for j in range(1,w-1):
            region = img[i-1:i+2, j-1:j+2]
            Gx[i,j] = np.sum(region*Kx)
            Gy[i,j] = np.sum(region*Ky)
    G = np.sqrt(Gx**2 + Gy**2)
    G = np.uint8(np.clip(G, 0, 255))
    print(f"Sobel Gx:\n{Gx}\nSobel Gy:\n{Gy}\nGradient Magnitude:\n{G}")
    return G

def canny_edge(img):
    out = cv2.Canny(img, 100, 200)
    print(f"Canny Edge Image:\n{out}")
    return out

# ---------------- 4. BINARY IMAGE ANALYSIS ----------------
def binary_threshold(img, thresh):
    out = np.zeros_like(img)
    out[img>thresh] = 255
    print(f"Binary Threshold ({thresh}) applied:\n{out}")
    return out

def otsu_threshold(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    _, th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print(f"Otsu Threshold applied:\n{th}")
    return th

def multiple_threshold(img, t1, t2):
    out = np.zeros_like(img)
    out[(img>t1) & (img<=t2)] = 127
    out[img>t2] = 255
    print(f"Multiple Threshold (t1={t1}, t2={t2}) applied:\n{out}")
    return out

def connected_components(binary):
    num_labels, labels = cv2.connectedComponents(binary)
    scaled_labels = labels * (255 // num_labels)
    print(f"Connected Components (num_labels={num_labels}):\n{scaled_labels}")
    return scaled_labels.astype(np.uint8)

# ---------------- 5. ARITHMETIC / LOGICAL OPS ----------------
def add_images(img1, img2):
    out = cv2.add(img1, img2)
    print(f"Added Image:\n{out}")
    return out

def and_images(img1, img2):
    out = cv2.bitwise_and(img1, img2)
    print(f"AND Image:\n{out}")
    return out

def or_images(img1, img2):
    out = cv2.bitwise_or(img1, img2)
    print(f"OR Image:\n{out}")
    return out

# ---------------- 6. SAMPLING & QUANTIZATION ----------------
def sampling(img, factor):
    out = img[::factor, ::factor]
    print(f"Sampled Image (factor={factor}):\n{out}")
    return out

def quantization(img, levels):
    step = 256//levels
    out = (img//step)*step
    print(f"Quantized Image to {levels} levels:\n{out}")
    return out

# ---------------- 7. INTERPOLATION ----------------
def zero_order_hold(img, scale):
    h, w = img.shape
    out = np.zeros((h*scale, w*scale), dtype=img.dtype)
    for i in range(h):
        for j in range(w):
            out[i*scale:(i+1)*scale, j*scale:(j+1)*scale] = img[i,j]
    print(f"Zero Order Hold (scale={scale}) applied:\n{out}")
    return out

def first_order_hold(img, scale):
    out = cv2.resize(img, (img.shape[1]*scale, img.shape[0]*scale), interpolation=cv2.INTER_LINEAR)
    print(f"First Order Hold (linear, scale={scale}) applied:\n{out}")
    return out

# ---------------- 8. BASIC OPS ----------------
def basic_ops(img):
    flipped = cv2.flip(img, 0)  # vertical flip
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cropped = img[1:4,1:4]  # simple crop for demo
    print(f"Flipped Image:\n{flipped}\nRotated Image:\n{rotated}\nCropped Image:\n{cropped}")
    return flipped, rotated, cropped

# ---------------- 9. HISTOGRAM EQUALIZATION ----------------
def histogram_equalization(img):
    h, w = img.shape
    hist = compute_histogram(img)
    cdf = hist.cumsum()
    cdf_norm = (cdf - cdf.min())*255/(cdf.max()-cdf.min())
    out = np.interp(img.flatten(), np.arange(256), cdf_norm)
    out = out.reshape((h,w)).astype('uint8')
    print(f"Histogram Equalized Image:\n{out}")
    return out

# ---------------- MAIN ----------------
if __name__ == "__main__":
    img = np.array([
        [0, 50, 100, 150, 200],
        [10, 60, 110, 160, 210],
        [20, 70, 120, 170, 220],
        [30, 80, 130, 180, 230],
        [40, 90, 140, 190, 240]
    ], dtype=np.uint8)
    
    img2 = np.copy(img)
    
    print("----- STEP 1: Histogram -----")
    hist = compute_histogram(img)
    plot_histogram(hist, "Original Histogram")
    
    print("\n----- STEP 2: Negative -----")
    neg = negative(img)
    
    print("\n----- STEP 3: Edge Detection -----")
    sobel = sobel_edge(img)
    canny = canny_edge(img)
    
    print("\n----- STEP 4: Binary Thresholds -----")
    binary = binary_threshold(img, 127)
    otsu = otsu_threshold(img)
    multi = multiple_threshold(img, 85, 170)
    labels = connected_components(binary)
    
    print("\n----- STEP 5: Arithmetic / Logical -----")
    added = add_images(img, img2)
    anded = and_images(img, binary)
    ored = or_images(img, binary)
    
    print("\n----- STEP 6: Sampling & Quantization -----")
    sampled = sampling(img, 2)
    quantized = quantization(img, 4)
    
    print("\n----- STEP 7: Interpolation -----")
    zoh = zero_order_hold(img, 2)
    foh = first_order_hold(img, 2)
    
    print("\n----- STEP 8: Basic Operations -----")
    flipped, rotated, cropped = basic_ops(img)
    
    print("\n----- STEP 9: Histogram Equalization -----")
    eq = histogram_equalization(img)
