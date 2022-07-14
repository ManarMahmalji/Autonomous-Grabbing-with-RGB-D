############################################################
## BRUFACE: Brussels Faculty of Engineering 
############################################################
## Master Thesis " Autonmous Grabbing with RGB-D"  
## July 2022 Manar Mahmalji      
## Version (1)     
############################################################
##This code serves to see the effect of adding more samples on 
## the obtained pose. This is referred to as anconvergence study. 
## A graphical visaulization of this study is done by a matlab script 

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


# Check if a matrix is a valid rotation matrix
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    if n>1e-3:
        print("Error! Not a rotation matrix")
    return n<1e-3

# Transform rotation matrix to euler angles: same as MATLAB
def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    x=math.degrees(x)
    y=math.degrees(y)
    z=math.degrees(z)
    return np.array([z, y, x])
  

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
        
    return target_R_list,target_t_list, gripper_R_list, gripper_t_list



##################CHANGE_IF_NEEDED##################
N= 25 # number of samples
samples=range(8,N) # range of samples over which convergence is studied
# for some reason, below 8, the calibration function does not work 
convStudy_filepath = 'MATLAB/convergence_calib.xlsx'
targetPose_filepath= 'Calibration Data/target2cam.xlsx'
gripperPose_filepath= 'Calibration Data/gripper2base.xlsx'
 ##################CHANGE_IF_NEEDED##################

# setup xlsx file
# First, we create a file, then we close it
# /!\ if the file already exists, it will be overwritten 
writer = pd.ExcelWriter(convStudy_filepath, engine="xlsxwriter")
writer.save()
# Then, we open the same file in append mode 
writer = pd.ExcelWriter(convStudy_filepath, mode="a", engine="openpyxl")


# set of methods for eye-to-hand calibration 
Tsai= cv2.CALIB_HAND_EYE_TSAI
Park= cv2.CALIB_HAND_EYE_PARK
Horaud= cv2.CALIB_HAND_EYE_HORAUD
Andreff= cv2.CALIB_HAND_EYE_ANDREFF
Daniilidis= cv2.CALIB_HAND_EYE_DANIILIDIS
methods= [Tsai, Park, Horaud,Andreff, Daniilidis]

# perfom eye-to-hand calibration, each time increasing the number of taken samples
for j in samples:
    # Read sampels pose data
    (R_target2cam,t_target2cam, R_gripper2base, t_gripper2base)=readSamples(targetPose_filepath,
        gripperPose_filepath,j)

    # since eye-to-hand configuration, change coordinates from gripper2base to base2gripper
    R_base2gripper, t_base2gripper = [], []
    for R, t in zip(R_gripper2base, t_gripper2base):
        R_b2g = R.T
        t_b2g = -R_b2g @ t
        R_base2gripper.append(R_b2g)
        t_base2gripper.append(t_b2g)
    
    # for each method perform the eye-to-hand calibration and 
    # save the obtained pose in the form of 3D position and 
    # euler angles in a list
    pose_list=[]
    for i in range(5):
        R_cam2base,t_cam2base= cv2.calibrateHandEye(
                R_gripper2base=R_base2gripper,
                t_gripper2base=t_base2gripper,
                R_target2cam=R_target2cam,
                t_target2cam=t_target2cam,method=methods[i]
            )

        EulerAng= rotationMatrixToEulerAngles(R_cam2base)
        a=np.ravel(t_cam2base.transpose())
        b=EulerAng
        pose=np.concatenate((a, b), axis=None)
        pose_list.append(pose)
        
    # separate poses corresponding to each method 
    Tsai=pose_list[0]
    Park=pose_list[1]
    Horaud=pose_list[2]
    Andreff=pose_list[3] 
    Daniilidis=pose_list[4]

    # for one number of samples, save the obtained poses in an excel sheet
    df= pd.DataFrame([Tsai, Park, Horaud, Andreff, Daniilidis],
                     index=['Tsai', 'Park','Horaud','Andreff', 'Daniilidis'],
                     columns=['x','y','z','roll','pitch','yaw'])

    sheet_name= str(j)+' samples'
    df.to_excel(writer, sheet_name=sheet_name)
    writer.save()


print("Convergence study is finished!")






