############################################################
## BRUFACE: Brussels Faculty of Engineering 
############################################################
## Master Thesis " Autonmous Grabbing with RGB-D"  
## July 2022 Manar Mahmalji      
## Version (1)     
############################################################
## This code serves for performing the eye-to-hand calibration
## in the 5 different methods proposed by OpenCV 


import pyrealsense2 as rs
import numpy as np
import cv2.aruco as aruco
import cv2
import math
import pandas as pd 
import time 


# Useful functions 

# Go from rvec and tvec to T matrix
def matrix_from_rtvec(rvec, tvec):
    (R, jac) = cv2.Rodrigues(rvec) # ignore the jacobian
    M = np.eye(4)
    M[0:3, 0:3] = R
    M[0:3, 3] = tvec.squeeze() # 1-D vector, row vector, column vector, whatever
    return M

# Go from T matrix to rvec and tvec 
def rtvec_from_matrix(M):
    (rvec, jac) = cv2.Rodrigues(M[0:3, 0:3]) # ignore the jacobian
    tvec = M[0:3, 3]
    return (rvec, tvec)

# Go from T matrix to rotation matrix R and translation vector t 
def rt_from_matrix(M):
    R= M[0:3, 0:3] 
    t=  M[0:3, 3]
    return (R, t)

# Go from rotation matrix R and translation vector t to T matrix
def matrix_from_rt(R,t):
    M = np.eye(4)
    M[0:3, 0:3] = R
    M[0:3, 3] = t.squeeze()
    return (M)



def readSamples( targetPoseFilePath, gripperPoseFilePath, NumOfSamples):
        
    # read from xlsx files the target (calibration board) pose wrt camera 
    # and gripper pose wrt robot base. The pose of each sample is stored in
    # the form of a T matrix

    target_R_list=[]
    target_t_list=[]
    gripper_R_list=[]
    gripper_t_list=[]

    # for each sample, extract the rotation matrix and transaltion vector from the T matrix 
    # then add them to a list 
    for j in range(NumOfSamples):

        # Traget pose 
        data = pd.read_excel(targetPoseFilePath, sheet_name='Sheet1'+str(j+1))
        T= data.head() 
        T= T.to_numpy()
        (R,t)= rt_from_matrix(T)
        target_R_list.append(R)
        target_t_list.append(t)

        # Gripper pose 
        data = pd.read_excel(gripperPoseFilePath, sheet_name='Sheet1'+str(j+1))
        T= data.head()
        T= T.to_numpy()
        (R,t)= rt_from_matrix(T)
        t=t*1000 # go from meter to milimeter
        gripper_R_list.append(R)
        gripper_t_list.append(t)
        
    print("Reading is finished!")
    return target_R_list,target_t_list, gripper_R_list, gripper_t_list



## Eye to hand calibration 

# Read samples pose data 
 ##################CHANGE_IF_NEEDED##################
N= 25 # Number of samples
targetPose_filepath= 'Calibration Data/target2cam.xlsx'
gripperPose_filepath= 'Calibration Data/gripper2base.xlsx'
cameraPose_filepath = 'Calibration Data/cam2base_test.xlsx'
 ##################CHANGE_IF_NEEDED##################
(R_target2cam,t_target2cam, R_gripper2base, t_gripper2base)=readSamples(targetPose_filepath,
 gripperPose_filepath,N)

# Methods for eye-to-hand calibration 
Tsai= cv2.CALIB_HAND_EYE_TSAI
Park= cv2.CALIB_HAND_EYE_PARK
Horaud= cv2.CALIB_HAND_EYE_HORAUD
Andreff= cv2.CALIB_HAND_EYE_ANDREFF
Daniilidis= cv2.CALIB_HAND_EYE_DANIILIDIS
methods= [Tsai, Park, Horaud,Andreff, Daniilidis]

# since eye-to-hand configuration, change coordinates from gripper2base to base2gripper
R_base2gripper, t_base2gripper = [], []
for R, t in zip(R_gripper2base, t_gripper2base):
    R_b2g = R.T
    t_b2g = -R_b2g @ t
    R_base2gripper.append(R_b2g)
    t_base2gripper.append(t_b2g)


# setup xlsx file
writer = pd.ExcelWriter(cameraPose_filepath, mode="w", engine="openpyxl")

# for each method perform the eye-to-hand calibration and 
# save the obtained transforamtion from camera to robot base
# frame in an excel sheet
for i in range(5):
    R_cam2base, t_cam2base= cv2.calibrateHandEye(
            R_gripper2base=R_base2gripper,
            t_gripper2base=t_base2gripper,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,method=methods[i]
        )
    # save data in excel file 
    T= matrix_from_rt(R_cam2base, t_cam2base)
    df = pd.DataFrame (T)
    sheet_name= 'method'+str(i+1)
    df.to_excel(writer, index=False, sheet_name=sheet_name)

writer.save()
print("Camera pose wrt base is saved!")






