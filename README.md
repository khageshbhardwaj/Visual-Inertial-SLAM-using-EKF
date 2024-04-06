# Visual-Inertial-SLAM-using-EKF
The project entails developing a Visual-Inertial SLAM algorithm by integrating IMU data and visual feature observations 
from stereo-camera images.
It comprises three main parts.

Project Description: https://drive.google.com/file/d/1dwGYWkjS6Tm9KZfhfIMY-vru9jwgw431/view?usp=sharing
Data: https://drive.google.com/drive/folders/1xBpuDP9Ee_nHvVB7zLbqkfuq4LWW3Ffj?usp=sharing

# Part a
Implementing an EKF prediction step using IMU measurements to estimate the IMU's pose over time. This part of the program 
generates a plot for an estimated pose trajectory over time.

# Part b
Incorporating an EKF update step for landmark mapping with visual features extracted from the camera images. This part of the code
initializes the landmark features using the feature point data. Following that, it updates the landmark locations by using EKF update
step. It also generates a 2D point cloud for landmark locations.

# Part c
and integrating these steps to create a cohesive algorithm for real-time pose estimation and landmark mapping. This integration aims to improve the accuracy and robustness of localization and mapping in dynamic environments, with potential applications in robotics, autonomous navigation, and augmented reality.


# To run the code
1. main.py contains all the code for parts a, b and c.
2. pr3_utils.py contains all the required functions referenced from main.py.

a. Run main.py, and it will run through and generate the required plots. 
b. Keep the code in the code folder or provide the appropriate path to access the data.
c. Change the dataset number to access another dataset.

# Dependencies
***for main.py:
import numpy as np
from tqdm import tqdm
from pr3_utils import *

***for pr3_utils.py:
import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler

