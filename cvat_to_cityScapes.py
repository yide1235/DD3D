from collections import defaultdict
from bs4 import BeautifulSoup
from pathlib import Path
import numpy as np
import json
from pyquaternion import Quaternion
import os

def parse_annotations(path_to_file):
    
    """
    Converts cvat xml files into kitti format.

    Parameters:
    
        path_to_file (str): The path of xml file.
        attribute_separator (char) : Charachter to separate attribute values in kitti_with_attributes file.

    """
    # Loading the file on to the variable

    soup = BeautifulSoup(open(path_to_file), 'xml')

    image_annotations = defaultdict(dict)
    class_counts = defaultdict(int)
    for cuboid in soup.find_all('cuboid'):
        img = cuboid.parent
        img_id = img['id']
        img_width = img['width']
        img_height = img['height']
        
        label = cuboid['label'].lower()
        class_counts[label] += 1
        # Order: top left, top right, bottom right, bottom left
        # Front
        P_f1_x = cuboid['xtl1']
        P_f1_y = cuboid['ytl1']

        P_f2_x = cuboid['xtr1']
        P_f2_y = cuboid['ytr1']

        P_f3_x = cuboid['xbr1']
        P_f3_y = cuboid['ybr1']

        P_f4_x = cuboid['xbl1']
        P_f4_y = cuboid['ybl1']

        # Back
        P_b1_x = cuboid['xtr2']
        P_b1_y = cuboid['ytr2']

        P_b2_x = cuboid['xtl2']
        P_b2_y = cuboid['ytl2']

        P_b3_x = cuboid['xbl2']
        P_b3_y = cuboid['ybl2']

        P_b4_x = cuboid['xbr2']
        P_b4_y = cuboid['ybr2']
        
        cuboid_points = np.zeros((8, 2))
        # front face
        cuboid_points[0] = [P_f1_x, P_f1_y]
        cuboid_points[1] = [P_f2_x, P_f2_y]
        cuboid_points[2] = [P_f3_x, P_f3_y]
        cuboid_points[3] = [P_f4_x, P_f4_y]
        
        # back face
        cuboid_points[4] = [P_b1_x, P_b1_y]
        cuboid_points[5] = [P_b2_x, P_b2_y]
        cuboid_points[6] = [P_b3_x, P_b3_y]
        cuboid_points[7] = [P_b4_x, P_b4_y]   

        world_coordinates = pixel_coords_to_world_coords(cuboid_points, projection_matrix)

        object = {
            "3d": {
                "center": get_center(world_coordinates).tolist(),
                "dimensions": get_dimensions(world_coordinates),
                "rotation": get_rotation(world_coordinates).tolist()
            },
            "label": label
        }
        if img_id not in image_annotations:
            sensor_data = {
                    "fx": 1.696492386995381139e+03,
                    "fy": 1.744145207786859601e+03,
                    "u0": 6.791408672800436079e+02,
                    "v0": 5.033422145735390245e+02
            }
            image_metadata = {
                "imgWidth": img_width,
                "imgHeight": img_height,
                "sensor": sensor_data,
                "objects": []
            }
            image_annotations[img_id] = image_metadata
        
        image_annotations[img_id]['objects'].append(object)
    
    return image_annotations, class_counts
        
    

def get_dimensions(points) -> list:
    """
    Returns [length, width, height] of box
    """
    length = np.sqrt((points[0][0] - points[4][0])**2 + (points[0][1] - points[4][1])**2 + (points[0][2] - points[4][2])**2)
    width = np.sqrt((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2 + (points[0][2] - points[1][2])**2)
    height = np.sqrt((points[0][0] - points[3][0])**2 + (points[0][1] - points[3][1])**2 + (points[0][2] - points[3][2])**2)
    return [length, width, height]
    


def get_center(points) -> list:
    """
    Returns [x, y, z] of center
    np.mean(points, axis = 0)
    """
    return np.average(points, axis=0)


def get_rotation(points) -> list:
    """
    Returns [q1, q2, q3, q4] of rotation in
    quaternions
    """
    P_b4 = points[7]

    P_b3 = points[6] # x axis
    P_b1 = points[4] # y axis
    P_f4 = points[3] # z axis
    
    v1 = P_b4 - P_f4 # back to front vector
    v2 = P_b4 - P_b4 # down to up vector

    # Quaternion q;
    a = np.cross(v1, v2)
    q_w = np.sqrt((np.linalg.norm(v1)) * (np.linalg.norm(v2))) + np.dot(v1, v2)
    
    q = Quaternion(q_w, a[0], a[1], a[2])
    # print(q)
    return np.zeros(4)


def pixel_coords_to_world_coords(pixel_coords : np.ndarray, 
                                projection_matrix  : np.ndarray) ->  np.ndarray:
    pixel_coords = np.concatenate([pixel_coords, np.ones((pixel_coords.shape[0],1 ))], axis=1)
    world_coords = np.matmul(pixel_coords, projection_matrix)
    world_coords = world_coords / np.expand_dims(world_coords[:,3], axis=1)
    return world_coords[:,:-1]


def readNpy(target):
    try:
        with(open(target, 'r')) as f:
            return np.loadtxt(f)
    except:
        return None

if __name__ == "__main__":
    rotationmatrix_path = "/Users/amir/projects/matt3r/DD3D/calibration/extrinsic/front/R_matrix.txt"
    translationvector_path = "/Users/amir/projects/matt3r/DD3D/calibration/extrinsic/front/T_vector.txt"

    R_matrix = readNpy(rotationmatrix_path)
    T_vector = readNpy(translationvector_path).reshape((3, 1))
    projection_matrix = np.concatenate([R_matrix, T_vector], axis=1)
    parsed_annotations, class_counts = parse_annotations("annotations.xml")
    print("Class Counts: ", class_counts)
    for filename, data in parsed_annotations.items():
        with open(f"{filename}.json", 'w') as f:
            json.dump(data, f)
