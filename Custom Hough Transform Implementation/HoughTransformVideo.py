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


def main():

    print('running... press q to exit')

    hough_transform = HoughTransform(theta_resolution=10, rho_resolution=10)
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output/custom_hough_output.mp4', fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, 50, 200)
        H, thetas, rhos = hough_transform.hough_lines_acc(edges)
        peaks = hough_transform.hough_peaks(H, thetas, rhos, threshold=150, num_peaks=100)
        frame_with_lines = hough_transform.hough_lines_draw(frame, peaks)

        out.write(frame_with_lines)
        
        cv2.imshow('Live Video with Hough Lines', frame_with_lines)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print('ended... video recording saved in output folder')

if __name__ == '__main__':
    main()
