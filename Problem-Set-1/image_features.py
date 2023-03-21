import cv2
import numpy as np

def crop_image(image_path):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    crop_height = int(height / 2)
    crop_width = int(width / 2)
    x = int((width - crop_width) / 2)
    y = int((height - crop_height) / 2)
    cropped_img = img[y:y+crop_height, x:x+crop_width]
    return cropped_img

def extract_red_channel(image_path):
    img = cv2.imread(image_path)
    red_channel = img[:,:,2]
    return red_channel

def grayscale_image(image_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def sobel_filter(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    gradient_magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    gradient_orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180
    return gradient_magnitude, gradient_orientation


def laplacian_of_gaussian(image_path, sigma_values):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Iterate over the sigma values
    for sigma in sigma_values:
        ksize = int(2 * np.ceil(3 * sigma) + 1)
        log = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
        log = cv2.Laplacian(log, cv2.CV_64F)
        log = cv2.normalize(log, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)
        cv2.imwrite(f'output/log_sigma_{sigma}.png', 255*log)


cropped_image = crop_image("input/image.jpg")
cv2.imwrite("output/cropped_image.png", cropped_image)

red_channel_image = extract_red_channel("input/image.jpg")
cv2.imwrite("output/red_channel_image.png", red_channel_image)

gray_image = grayscale_image("input/image.jpg")
cv2.imwrite("output/gray_image.png", gray_image)

gradient_magnitude_image, gradient_orientation_image = sobel_filter("input/image.jpg")
cv2.imwrite("output/gradient_magnitude_image.png", gradient_magnitude_image)
cv2.imwrite("output/gradient_orientation_image.png", gradient_orientation_image)

laplacian_of_gaussian("input/image.jpg", [1, 2, 3, 4, 5, 6])
