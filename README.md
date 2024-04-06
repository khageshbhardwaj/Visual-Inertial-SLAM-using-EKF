# Visual-Inertial-SLAM-using-EKF
The project entails developing a Visual-Inertial SLAM algorithm by integrating IMU data and visual feature observations 
from stereo-camera images.
It comprises three main parts.
Data: https://drive.google.com/drive/folders/1xBpuDP9Ee_nHvVB7zLbqkfuq4LWW3Ffj?usp=sharing
Project Description: https://drive.google.com/file/d/1dwGYWkjS6Tm9KZfhfIMY-vru9jwgw431/view?usp=sharing

# part 1
Implementing an EKF prediction step using IMU measurements to estimate the IMU's pose over time. This part of the program 
generates a plot for an estimated pose trajectory over time.

# part 2
Incorporating an EKF update step for landmark mapping with visual features extracted from the camera images.


# part 3
and integrating these steps to create a cohesive algorithm for real-time pose estimation and landmark mapping. This integration aims to improve the accuracy and robustness of localization and mapping in dynamic environments, with potential applications in robotics, autonomous navigation, and augmented reality.


# To run the code
we need only two files given in code folder, main.py and pr3utils.py

1. main.py contains all the code for part a, b and c.
2. pr3_utils.py contains all the requied functions referenced from main.py.

a. Run main.py and it will run through and generate the required plots. 
b. Keep the code in the code folder, else provide approapriate path.
c. Change the dataset number to access another dataset.

# import them before running the code
***for main.py

import numpy as np
from tqdm import tqdm
from pr3_utils import *

***for pr3_utils.py

import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler

