
import numpy as np
from tqdm import tqdm
from pr3_utils import *

filename = "../data/10.npz"
t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)

# (a) IMU Localization via EKF Prediction
#  Calculating incremental timestamp
ts_diff = np.array(t[0,1:] - t[0,:-1])

# forming the se(3) matrix using the linear and angular velocity
v = linear_velocity
w = angular_velocity
zeta = np.vstack((v,w))
zeta1 = zeta[:,:-1]    # drop the last element of zeta to broadcast it with tau
ts_diff_reshaped = ts_diff.reshape(1, -1)    # Reshape ts_diff to broadcast along the columns
tau_zeta = zeta1 * ts_diff_reshaped    # Multiply zeta with ts_diff element-wise
twist_se3 = column2se3(tau_zeta)   # calculate zeta hat
twist_pose = twist2pose(twist_se3)   # from se(3) to SE(3)

world_T_imu = np.zeros((3026, 4, 4))  # initial IMU matrix
# world_T_imu[0, :, :] = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
world_T_imu[0, :, :] = np.eye(4)

# save the result
np.save('world_T_imu.npy', world_T_imu)

#  generate the odometry pose
for i in range(twist_pose.shape[0]):
    world_T_imu[i+1, :, :] = world_T_imu[i, :, :] @ twist_pose[i, :, :]

# Plot x vs y trajectory
plt.plot(world_T_imu[:,0,3], world_T_imu[:,1,3], color = 'red', label='IMU Trajectory part a')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs Y Trajectory')
plt.legend()
plt.show()
plt.savefig('imu_trajectory')

# # Visualize the robot pose over time
# visualize_trajectory_2d(np.transpose(world_T_imu, axes=(1, 2, 0)), show_ori = False)

# (b) Landmark Mapping via EKF Update
# formulate intrinsic parameters
fsu = K[0,0]
fsv = K[1,1]
cu = K[0,2]
cv = K[1,2]

Ks = np.zeros((4,4))
Ks = np.array([[fsu, 0, cu, 0], [0, fsv, cv, 0], [fsu, 0, cu, -fsu*b], [0, fsv, cv, 0]])

# initializing the map locations
processed_features = set()
m_bar_world = np.zeros((4, features.shape[1]))    # initializing the m_bar_world
k_inv = np.linalg.pinv(K)   # calculate inverse of the extrinsic matrix

for i in tqdm(range(features.shape[2])):
    # Pick columns where most of the values are not equal to -1
    visible_feature_boolean = np.any(features[:,:,i] != -1, axis=0)
    visible_features = np.where(visible_feature_boolean)[0]

    # Filter out the features that have already been processed
    new_visible_features = [f for f in visible_features if f not in processed_features]

    for feature_index in new_visible_features:
        ul = features[0,feature_index,i]
        vl = features[1,feature_index,i]
        ur = features[2,feature_index,i]
        vr = features[3,feature_index,i]

        zl = k_inv @ [ul, vl, 1]
        zr = k_inv @ [ur, vr, 1]

        m_bar_cam_feature = b/(zl[0] - zr[0]) * zl

        m_bar_cam_h = np.hstack((m_bar_cam_feature, 1))

        m_bar_imu_h = imu_T_cam @ m_bar_cam_h
        m_bar_world[:, feature_index] =  world_T_imu[i, :, :] @ m_bar_imu_h

        # Add the feature index to the set of processed features
        processed_features.add(feature_index)


# save the result
np.save('m_bar_world.npy', m_bar_world)

# Plot x vs y trajectory
plt.scatter(m_bar_world[0,:], m_bar_world[1,:], s=1, color = 'green', label='World Points initialized')
plt.plot(world_T_imu[:,0,3], world_T_imu[:,1,3], color = 'red', label='IMU Trajectory part a')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs Y Trajectory')
plt.legend()
plt.show()
plt.savefig('m_initialized')

# initializing the map location & co-variances
Sigma = 2 * np.eye(3*features.shape[1],dtype=np.float32)

mu_bar= m_bar_world
mu_flatten = m_bar_world[:-1,:].T.flatten()[:, None]
sigma_map = Sigma

inv_world_T_imu = inversePose(world_T_imu)
inv_imu_T_cam = np.linalg.pinv(imu_T_cam)

# z_tilda = np.zeros((4, features.shape[1]))
# H = np.zeros((4*features.shape[1], 3*features.shape[1]))

P = np.hstack((np.eye(3), np.zeros((3, 1))))

for i in tqdm(range(features.shape[2]-1)):
# for i in range(1):
    visible_feature_boolean = np.any(features[:,:,i] != -1, axis=0)
    visible_features = np.where(visible_feature_boolean)[0]

    #################################################################################################################
    mu_t_bar = mu_bar[:, visible_features]
    m_bar_imu_update = inv_world_T_imu[i+1, :, :] @ mu_t_bar
    m_bar_cam_update = inv_imu_T_cam @ m_bar_imu_update
    m_projection = projection(m_bar_cam_update.T)
    z_tilda= Ks @ m_projection.T

    ###################################################################################################################
    invT_PT = inv_world_T_imu[i+1, :, :] @ P.T
    H3 = inv_imu_T_cam @ invT_PT
    jacobian_m = projectionJacobian(m_bar_cam_update.T)
    jacobian_m_reshaped = jacobian_m.reshape(-1, 4)

    H2_reshaped = jacobian_m_reshaped @ H3
    H2 = H2_reshaped.reshape(jacobian_m.shape[0], 4, 3)

    # Expand dimensions of matrix_4x4 to make it 3D of shape (1, 4, 4)
    expanded_Ks = np.expand_dims(Ks, axis=0)
    matrix_nx4x3 = expanded_Ks @ H2

    H = np.zeros((len(visible_features)* 4, len(visible_features) * 3))
    # Place each 4x3 matrix along the diagonal
    for j in range(len(visible_features)):
        H[j * 4:(j +1) * 4 ,j * 3:(j +1) * 3] = matrix_nx4x3[j]

    ####################################################################################################################
    # Calculate the Kalman Gain
    V = 4 * np.eye(4*visible_features.shape[0])
    sigma_t_map = extract_sigma(sigma_map, visible_features)

    prod_SH = sigma_t_map @ H.T
    prod_HSH = H @ prod_SH
    inv_HSH = np.linalg.pinv(prod_HSH + V)
    prod_H_invHSH = H.T @ inv_HSH
    K_gain = sigma_t_map @ prod_H_invHSH

    #####################################################################################################################
    # update mean (mu)
    ind = visible_features
    good_ind = np.array([np.arange(3 * int(xx), 3 * int(xx) + 3) for xx in ind[0:int(ind.shape[0])]]).reshape(-1,1)
    mu_t_flatten_final = mu_flatten[good_ind[:,0],:]
    error_flatten = (features[:, visible_features, i] - z_tilda).T.flatten()[:, None]
    mu_t_flatten_final = mu_t_flatten_final + ( K_gain @ error_flatten)

    # #update in the complete matrix
    mu_flatten[good_ind[:,0],:] = mu_t_flatten_final
    mu = mu_flatten.reshape(-1, 3).T
    mu_bar = np.vstack((mu, np.ones((1, mu.shape[1]))))

    ###################################################################################################################
    # update co-variance matrix
    sigma_t_map = (np.eye(K_gain.shape[0]) - (K_gain @ H)) @ sigma_t_map

    # Plug back the modified blocks into the original sigma matrix
    sigma_map = plug_back_sigma(sigma_map, sigma_t_map, visible_features)


# save the result
np.save('mu_bar.npy', mu_bar)

plt.scatter(mu_bar[0,:], mu_bar[1,:], s=1, color = 'green',label='World Points Updated part b')
plt.plot(world_T_imu[:,0,3], world_T_imu[:,1,3], color = 'red', label='IMU Trajectory part a')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs Y Trajectory')
plt.legend()
plt.show()
plt.savefig('imu_a_point_b')

# retrieve the landmark location from mapping
m_landmark = mu
m_landmark_bar = mu_bar
m_landmark_flatten = m_landmark_bar[:-1,:].T.flatten()[:, None]
sigma_slam = 2 * np.eye(3*features.shape[1] + 6, dtype=np.float32)

sigma_slam_final = sigma_slam
mu_iTw = inv_world_T_imu
wTi_slam = world_T_imu

# define F
tau_zeta = zeta1 * -ts_diff_reshaped
twist_se3 = column2se3(tau_zeta)
twist_pose = twist2pose(twist_se3)
F = pose2adpose(twist_pose)

F1 = F
# W = np.diag([0.01] * 3 + [0.001] * 3)
# W = np.diag([1e-1,1e-2,1e-2,1e-5,1e-5,1e-4])
W = np.diag([1e-3,1e-3,1e-3,1e-5,1e-5,1e-5])
sigma_LL = sigma_slam_final[:3*features.shape[1], :3*features.shape[1]]
sigma_LR = sigma_slam_final[:3*features.shape[1], 3*features.shape[1]:]
sigma_RL = sigma_slam_final[3*features.shape[1]:, :3*features.shape[1]]
# sigma_RR = sigma_slam_final[3*features.shape[1]:, 3*features.shape[1]:]

# sigma_RR = np.diag([1e-3,1e-3,1e-3,1e-4,1e-4,1e-4])
sigma_RR = np.diag([1e-2,1e-2,1e-2,1e-3,1e-3,1e-3])

####################################################################################################################
for i in tqdm(range(features.shape[2]-1)):
# for i in range(1):
    visible_feature_boolean = np.any(features[:,:,i] != -1, axis=0)
    visible_features = np.where(visible_feature_boolean)[0]

    sigma_slam_final[:3*features.shape[1], 3*features.shape[1]:] = sigma_LR @ F1[i, :, :].T
    sigma_slam_final[3*features.shape[1]:, :3*features.shape[1]] = F1[i, :, :] @ sigma_RL
    sigma_slam_final[3*features.shape[1]:, 3*features.shape[1]:] = F1[i, :, :] @ sigma_RR @ F1[i, :, :].T + W

    # sigma_LL = sigma_slam_final[:3*features.shape[1], :3*features.shape[1]]
    sigma_LR = sigma_slam_final[:3*features.shape[1], 3*features.shape[1]:]
    sigma_RL = sigma_slam_final[3*features.shape[1]:, :3*features.shape[1]]
    sigma_RR = sigma_slam_final[3*features.shape[1]:, 3*features.shape[1]:]

    #################################################################################################################
    # predicted observation from localization
    m_landmark_t_bar = m_landmark_bar[:, visible_features]    #dim: 4*n
    proj_arg = inv_imu_T_cam @ mu_iTw[i+1, :, :] @ m_landmark_t_bar
    m_proj_slam = projection(proj_arg.T)
    z_til_slam= Ks @ m_proj_slam.T

    ################################################################################################################
    ############# recalculating the H_landmarks
    invT_PT = mu_iTw[i+1, :, :] @ P.T
    H3 = inv_imu_T_cam @ invT_PT

    jacobian_m_map = projectionJacobian(proj_arg.T)
    jacobian_m_map_reshaped = jacobian_m_map.reshape(-1, 4)

    H2_reshaped_map = jacobian_m_map_reshaped @ H3
    H2 = H2_reshaped_map.reshape(jacobian_m_map.shape[0], 4, 3)

    # Expand dimensions of matrix_4x4 to make it 3D of shape (1, 4, 4)
    expanded_Ks = np.expand_dims(Ks, axis=0)
    matrix_nx4x3 = expanded_Ks @ H2

    H_landmark = np.zeros((len(visible_features)* 4, len(visible_features) * 3))

    # Place each 4x3 matrix along the diagonal
    for j in range(len(visible_features)):
        H_landmark[j * 4:(j +1) * 4 ,j * 3:(j +1) * 3] = matrix_nx4x3[j]

    ######################################################
    # H-Robot
    # dot operation
    dot_arg = mu_iTw[i+1, :, :] @ m_landmark_t_bar
    opr_dot_arg = dotoperation(dot_arg.T) # n*4*6
    # opr_dot_arg_T = np.transpose(opr_dot_arg, axes=(1, 2, 0)) # dim 4*6*n

    # jacobian PI part
    jacob_m_map = projectionJacobian(proj_arg.T)  # dim: n*4*4
    # jacob_m__map_reshaped = jacob_m_map.reshape(-1, 4)      # dim: 4n*4

    H_slam = np.zeros((len(visible_features)* 4, len(visible_features) * 3 + 6))  # dim: 4n X (3n+6)

    H_robot = np.zeros((len(visible_features)* 4, 6))

    # Place each 4x3 matrix along the diagonal
    for j in range(len(visible_features)):
        H0 = inv_imu_T_cam @ opr_dot_arg[j, :, :]
        H1 = jacob_m_map[j, :, :] @ H0
        H2 = -Ks @ H1
        H_robot[j * 4:(j + 1) * 4, :] = H2


    H_slam[:,:len(visible_features)* 3] = H_landmark
    H_slam[:,len(visible_features)* 3 :] = H_robot

    ##################################################################################################################
    # Calculate the Kalman Gain
    num = visible_features.shape[0]

    sigma_t_slam = np.zeros((num*3+6, num*3+6))
    sigma_t_slam[:3*num, :3*num] = extract_sigma(sigma_LL, visible_features)
    sigma_t_slam[:3*num, 3*num:] = extract_sigma_LR(sigma_LR, visible_features)
    sigma_t_slam[3*num:, :3*num] = extract_sigma_RL(sigma_RL, visible_features)
    sigma_t_slam[3*num:, 3*num:] = sigma_RR

    prod_SH_slam = sigma_t_slam @ H_slam.T
    prod_HSH_slam = H_slam @ prod_SH_slam

    V = 4 * np.eye(4*visible_features.shape[0])
    inv_HSH_slam = np.linalg.pinv(prod_HSH_slam + V)
    prod_H_invHSH_slam = H_slam.T @ inv_HSH_slam
    K_gain_slam = sigma_t_slam @ prod_H_invHSH_slam   # dim: (3*n + 6) X 4n


    Kgain_landmark = K_gain_slam[:3*num, :]
    Kgain_robot = K_gain_slam[3*num:,:]

    # #################################################################################################################
    # update co-variance matrix
    # landmark covariance
    sigma_t_slam[:3*num, :3*num] = (np.eye(Kgain_landmark.shape[0]) - (Kgain_landmark @ H_landmark)) @ sigma_t_slam[:3*num, :3*num]

    # # Plug back the modified blocks into the original sigma matrix
    sigma_slam_final = plug_back_sigma(sigma_slam_final, sigma_t_slam, visible_features)

    ###########################################################
    # robot covariance
    sigma_RR = (np.eye(Kgain_robot.shape[0]) - (Kgain_robot @ H_robot)) @ sigma_t_slam[3*num:, 3*num:]
    sigma_slam_final[3*features.shape[1]:, 3*features.shape[1]:] = sigma_RR

    #####################################################################################################################
    # landmark updates
    ind_map = visible_features
    good_ind_map = np.array([np.arange(3 * int(xx), 3 * int(xx) + 3) for xx in ind_map[0:int(ind_map.shape[0])]]).reshape(-1,1)
    m_landmark_t_flatten = m_landmark_flatten[good_ind_map[:,0],:]
    error_flatten_lm = (features[:, visible_features, i] - z_til_slam).T.flatten()[:, None]
    m_landmark_t_flatten = m_landmark_t_flatten + ( Kgain_landmark @ error_flatten_lm)

    # update in the complete matrix
    m_landmark_flatten[good_ind_map[:,0],:] = m_landmark_t_flatten    #flatten x,y,z
    m_landmark = m_landmark_flatten.reshape(-1, 3).T
    m_landmark_bar = np.vstack((m_landmark, np.ones((1, m_landmark.shape[1]))))

    ###########################################################
    # trajectory pose update
    error_flatten_imu = (features[:, visible_features, i] - z_til_slam).T.flatten()[:, None]
    axis_angle_term = Kgain_robot @ error_flatten_imu
    wTi_slam[i+1, :, :] = wTi_slam[i+1, :, :] @ twist2pose(axangle2twist(axis_angle_term.T))

np.save('wTi_slam.npy', wTi_slam)
np.save('m_landmark.npy', m_landmark)

plt.scatter(m_landmark[0,:], m_landmark[1,:], s=1, color = 'green', label='World Points Updated part c')
plt.plot(wTi_slam[:,0,3], wTi_slam[:,1,3], color = 'red', label='IMU Trajectory Updated part c')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs Y Trajectory')
plt.legend()
plt.show()
plt.savefig('partc_imu_m')

plt.scatter(mu_bar[0,:], mu_bar[1,:], s=1, color = 'blue',label='World Points Updated part b')
plt.scatter(m_landmark[0,:], m_landmark[1,:], s=1, color = 'green', label='World Points updated part c')
plt.plot(wTi_slam[:,0,3], wTi_slam[:,1,3], color = 'red', label='IMU Trajectory Updated part c')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs Y Trajectory')
plt.legend()
plt.show()
plt.savefig('compare_b_c')