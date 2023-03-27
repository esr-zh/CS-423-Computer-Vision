import cv2
import numpy as np
import time

vid = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/output.mp4', fourcc, 20.0, (1280, 480))

start_time = cv2.getTickCount()
record_duration = 15.0  

while True:
    ret, frame = vid.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gX = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=5)
    gY = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    
    gradient_magnitude_rgb = cv2.cvtColor(
        cv2.convertScaleAbs(gradient_magnitude),
        cv2.COLOR_GRAY2BGR
    )

    concatenated_frame = np.concatenate(
        (frame, gradient_magnitude_rgb), axis=1
    )

    fps = vid.get(cv2.CAP_PROP_FPS)
    cv2.putText(concatenated_frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('video', concatenated_frame)
    out.write(concatenated_frame)

    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    if elapsed_time >= record_duration:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
vid.release()
out.release()
cv2.destroyAllWindows()
