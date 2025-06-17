import cv2
import numpy as np
import os

#Image-Filters

def apply_sepia(image):
    """Applies a sepia tone effect to the image."""
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia = cv2.transform(image, sepia_filter)
    return np.clip(sepia, 0, 255).astype(np.uint8)

def add_grain(image, intensity=0.05):
    """Adds film grain (noise) to the image."""
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    return cv2.addWeighted(image, 1 - intensity, noise, intensity, 0)

def add_vignette(image, strength=200):
    """Applies a vignette (dark corners) effect to the image."""
    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, strength)
    kernel_y = cv2.getGaussianKernel(rows, strength)
    mask = kernel_y * kernel_x.T
    mask = mask / np.max(mask)
    vignette = np.zeros_like(image)
    for i in range(3):  # Apply mask to each channel
        vignette[:, :, i] = image[:, :, i] * mask
    return vignette.astype(np.uint8)

def pencil_sketch(image):
    """Creates a pencil sketch effect from the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def grayscale(image):
    """Converts the image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def duotone(image, dark_color=(10, 20, 90), light_color=(250, 180, 200)):
    """Applies a duotone effect using two color tones."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = gray / 255.0
    duotone_img = np.zeros_like(image)
    for i in range(3):
        duotone_img[:, :, i] = (dark_color[i] * (1 - normalized) + light_color[i] * normalized)
    return duotone_img.astype(np.uint8)

def thermal(image):
    """Applies a thermal imaging effect using color mapping."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thermal_base = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    blurred = cv2.GaussianBlur(thermal_base, (9, 9), 0)
    thermal_mixed = cv2.addWeighted(thermal_base, 0.7, blurred, 0.3, 0)
    return thermal_mixed

def glitch_effect(image):
    """Applies a simple RGB shift glitch effect."""
    shifted = image.copy()
    rows, cols, _ = image.shape
    dx = np.random.randint(5, 20)
    shifted[:, :, 0] = np.roll(shifted[:, :, 0], dx, axis=1)    # Shift Blue channel
    shifted[:, :, 2] = np.roll(shifted[:, :, 2], -dx, axis=1)   # Shift Red channel
    return shifted

def cartoon_effect(image):
    """Applies a cartoon effect using edge detection and smoothing."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  blockSize=9,
                                  C=2)
    color = cv2.bilateralFilter(image, d=9, sigmaColor=300, sigmaSpace=300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def pop_art(image):
    """Creates a pop art style by altering hue and increasing contrast."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, 0] = (hsv[:, 0] + 90) % 180  # Shift hue
    pop = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.convertScaleAbs(pop, alpha=1.5, beta=30)

def hdr_enhance(image):
    """Applies detail enhancement for an HDR-like effect."""
    return cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)

def solarize(image, threshold=128):
    """Applies a solarize effect by inverting pixels above a threshold."""
    solarized = image.copy()
    for i in range(3):
        mask = solarized[:, :, i] > threshold
        solarized[:, :, i][mask] = 255 - solarized[:, :, i][mask]
    return cv2.convertScaleAbs(solarized, alpha=1.2, beta=15)

def negative_filter(image):
    """Inverts all colors (negative effect)."""
    return cv2.bitwise_not(image)

def edge_detect(image):
    """Applies edge detection and returns a colored edge map."""
    edges = cv2.Canny(image, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

#Filter-Function-Calling

def convert_image(image, mode):
    """Applies the selected image filter based on mode string."""
    if mode == 'retro':
        image = apply_sepia(image)
        image = add_grain(image)
        image = add_vignette(image)
    elif mode == 'sketch':
        image = pencil_sketch(image)
    elif mode == 'gray':
        image = grayscale(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert back for consistency
    elif mode == 'duotone':
        image = duotone(image)
    elif mode == 'thermal':
        image = thermal(image)
    elif mode == 'glitch':
        image = glitch_effect(image)
    elif mode == 'cartoon':
        image = cartoon_effect(image)
    elif mode == 'pop':
        image = pop_art(image)
    elif mode == 'hdr':
        image = hdr_enhance(image)
    elif mode == 'solarize':
        image = solarize(image)
    elif mode == 'negative':
        image = negative_filter(image)
    elif mode == 'edges':
        image = edge_detect(image)
    return image

#Testing

if __name__ == '__main__':
    test_image_path = 'WIN_20241107_13_03_02_Pro.jpg'  # Path to input image
    mode = 'cartoon'  # Choose from: 'retro', 'sketch', 'gray', 'duotone', 'thermal', 'glitch', 'cartoon', 'pop', 'hdr', 'solarize', 'negative', 'edges'

    # Load and validate image
    image = cv2.imread(test_image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {test_image_path}")

    # Resize for display and processing convenience
    image = cv2.resize(image, (600, 400))

    # Apply the selected filter
    result = convert_image(image, mode)

    # Save the processed image
    output_path = f'{mode}_output.jpg'
    cv2.imwrite(output_path, result)
    print(f"Saved processed image as: {output_path}")
 