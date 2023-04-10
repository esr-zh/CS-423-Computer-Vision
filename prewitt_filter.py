import numpy as np

# the input matrix
matrix = np.array([[5, 9, 10, 11, 12],
                   [6, 8, 7, 9, 13],
                   [7, 7, 20, 10, 13],
                   [8, 4, 5, 11, 12],
                   [2, 3, 3, 8, 11]])

matrix = np.array([[7,8,9],
                    [6,14,10],
                    [4,12,11]])


# Define Prewitt filter kernels
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

prewitt_y = np.array([[-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]])

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# pad the matrix using wrap around technique
padded_matrix = np.pad(matrix, pad_width=1, mode='wrap')

def apply_filter(matrix, kernel):
    filtered_matrix = np.zeros(matrix.shape)
    for i in range(1, matrix.shape[0]-1):
        for j in range(1, matrix.shape[1]-1):
            filtered_matrix[i, j] = np.sum(matrix[i-1:i+2, j-1:j+2] * kernel)
    return filtered_matrix[1:-1, 1:-1]

# Apply required filter
prewitt_filtered_x = apply_filter(padded_matrix, prewitt_x)
prewitt_filtered_y = apply_filter(padded_matrix, prewitt_y)

# Calculate the gradient magnitude
prewitt_gradient = np.sqrt(prewitt_filtered_x**2 + prewitt_filtered_y**2)

print("Prewitt Filtered X:\n", prewitt_filtered_x)
print("Prewitt Filtered Y:\n", prewitt_filtered_y)
print("Prewitt Gradient Magnitude:\n", prewitt_gradient)
