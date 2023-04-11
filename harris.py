import numpy as np

Ix = np.array([[2, 1, 0],
               [3, 1, 1],
               [2, 2, 1]])

Iy = np.array([[1, -1, 0],
               [0, 1, 1],
               [1, 0, 3]])

Ix2 = Ix * Ix
Iy2 = Iy * Iy
IxIy = Ix * Iy

sum_Ix2 = np.sum(Ix2)
sum_Iy2 = np.sum(Iy2)
sum_IxIy = np.sum(IxIy)

M = np.array([[sum_Ix2, sum_IxIy],
              [sum_IxIy, sum_Iy2]])

det_M = np.linalg.det(M)
trace_M = np.trace(M)

# Calculate the Harris response
k = 0.04
R = det_M - k * trace_M**2

print("Harris response at the center pixel: ", R)
