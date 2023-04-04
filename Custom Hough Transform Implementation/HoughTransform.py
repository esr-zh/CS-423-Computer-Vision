import cv2
import numpy as np
import math

class HoughTransform:
    def __init__(self, theta_resolution=1, rho_resolution=1):
        self.theta_resolution = theta_resolution
        self.rho_resolution = rho_resolution

    def hough_lines_acc(self, edge_image):
        width, height = edge_image.shape
        rho_max = int(np.sqrt(width**2 + height**2))
        rhos = np.arange(-rho_max, rho_max, self.rho_resolution)
        thetas = np.deg2rad(np.arange(-180, 180, self.theta_resolution))

        H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
        edge_points = np.where(edge_image == 255)

        for i in range(len(edge_points[0])):
            y, x = edge_points[0][i], edge_points[1][i]
            for j in range(len(thetas)):
                rho = x * np.cos(thetas[j]) + y * np.sin(thetas[j])
                rho_index = int((rho + rho_max) / self.rho_resolution)
                H[rho_index, j] += 1
                
        return H, thetas, rhos

    def hough_peaks(self, H, thetas, rhos, threshold, num_peaks):
        peaks = []
        H_copy = H.copy()
        for _ in range(num_peaks):
            max_val = H_copy.max()
            if max_val > threshold:
                rho_index, theta_index = np.where(H_copy == max_val)
                peaks.append((thetas[theta_index[0]], rhos[rho_index[0]]))
                H_copy[rho_index[0], theta_index[0]] = 0
            else:
                break
                
        return peaks

    def hough_lines_draw(self, img, peaks):
        img_lines = img.copy()
        for peak in peaks:
            theta, rho = peak
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        return img_lines

def hough_lines_opencv(img, rho_res, theta_res, threshold):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 50, 200)
    lines = cv2.HoughLinesP(edges, rho_res, np.deg2rad(theta_res), threshold, minLineLength=10, maxLineGap=250)
    img_lines = img.copy()

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img_lines


def main():

    print('running...')

    # Read an input image
    img = cv2.imread('input/image2.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 50, 200)

    # Hough Transform
    hough_transform = HoughTransform()
    H, thetas, rhos = hough_transform.hough_lines_acc(edges)
    peaks = hough_transform.hough_peaks(H, thetas, rhos, threshold=150, num_peaks=100)
    img_lines = hough_transform.hough_lines_draw(img, peaks)

    # Display the images
    #cv2.imshow('Input Image', img)
    #cv2.imshow('Edge Image', edges)
    #cv2.imshow('Hough Lines', img_lines)

    cv2.imwrite('output/Edge_Image.jpg', edges)
    cv2.imwrite('output/Custom_Hough_Lines.jpg', img_lines)

    img_lines_opencv = hough_lines_opencv(img, 1, 1, 150)
    #cv2.imshow('OpenCV Hough Lines', img_lines_opencv)
    cv2.imwrite('output/OpenCV_Hough_Lines.jpg', img_lines_opencv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('ended... check results in output folder')

if __name__ == '__main__':
    main()
