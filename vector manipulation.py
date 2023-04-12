import numpy as np

def transform_vector(vector, translation, rotation_angle_degrees):
    # Convert angle to radians
    rotation_angle_radians = np.deg2rad(rotation_angle_degrees)

    # Create translation and rotation matrices
    translation_matrix = np.array([
        [1, 0, 0, -translation[0]],
        [0, 1, 0, -translation[1]],
        [0, 0, 1, -translation[2]],
        [0, 0, 0, 1]
    ])

    rotation_matrix = np.array([
        [np.cos(rotation_angle_radians), -np.sin(rotation_angle_radians), 0, 0],
        [np.sin(rotation_angle_radians), np.cos(rotation_angle_radians), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Apply translation and rotation
    vector = np.append(vector, 1)  # Convert vector to homogeneous coordinates
    translated_vector = np.dot(translation_matrix, vector)
    transformed_vector = np.dot(rotation_matrix, translated_vector)

    return transformed_vector[:-1]  # Convert back to 3D coordinates

vector = np.array([7, 3, 2])
translation = np.array([4, -3, 7])
rotation_angle_degrees = 90

transformed_vector = transform_vector(vector, translation, rotation_angle_degrees)
print(transformed_vector)
