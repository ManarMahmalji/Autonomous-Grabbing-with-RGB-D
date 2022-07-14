############################################################
## BRUFACE: Brussels Faculty of Engineering 
############################################################
## Master Thesis " Autonmous Grabbing with RGB-D"  
## July 2022 Manar Mahmalji      
## Version (1)     
############################################################
# This code serves for saving, in a txt file the grabbing 
# pose of the block only after it becomes static 

import pyrealsense2 as rs
import numpy as np
import cv2.aruco as aruco
import cv2
import math
import pandas as pd 
import time 
import os 

# Go from T matrix to rotation matrix R and translation vector t 
def rt_from_matrix(M):
    r= M[0:3, 0:3] 
    t=  M[0:3, 3]
    return (r, t)

# Go from rvec and tvec to T matrix
def matrix_from_rtvec(rvec, tvec):
    (R, jac) = cv2.Rodrigues(rvec) # ignore the jacobian
    M = np.eye(4)
    M[0:3, 0:3] = R
    M[0:3, 3] = tvec.squeeze() # 1-D vector, row vector, column vector, whatever
    return M

# Check if a matrix is a valid rotation matrix
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    if n>1e-3:
        print("Error! Not a rotation matrix")
    return n<1e-3

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB 
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

# Rotation matrix around z 
def Rotz(gamma):
    R= np.array([[math.cos(gamma), -math.sin(gamma), 0],
                [math.sin(gamma),  math.cos(gamma), 0],
                [0, 0, 1]])  
    return R
# Rotation matrix around y
def Roty(betta):
    R= np.array([[math.cos(betta), 0, math.sin(betta)],
                [0, 1, 0],
                [-math.sin(betta), 0, math.cos(betta)]]) 
    return R
# Rotation matrix around x
def Rotx(alpha):
    R= np.array([[1, 0, 0],
                [0, math.cos(alpha), -math.sin(alpha)],
                [0, math.sin(alpha), math.cos(alpha)]]) 
    return R


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# Configure resolution of depth and color sensors
##################CHANGE_IF_NEEDED##################
# Lower FPS, higher resoultion 
config.enable_stream(rs.stream.color, 1280 , 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Higher FPS, lower resolution 
# config.enable_stream(rs.stream.color, 848 ,480, rs.format.bgr8, 60)
# config.enable_stream(rs.stream.depth, 848 ,480, rs.format.z16, 60)

# Higher FPS, lower resolution 
# config.enable_stream(rs.stream.color, 640 ,360, rs.format.bgr8, 90)
# config.enable_stream(rs.stream.depth, 640 ,360, rs.format.z16, 90)
##################CHANGE_IF_NEEDED##################

if device_product_line == 'D400':
    print('Hello, I am D455')
    
# Create an align object
# rs.align allows us to perform alignment of a stream to other streams
# In this case, depth stream is aligned to color stream since the depth
# stream is centerd at the left camera whereas the color stream is 
# centered at the RGB camera/sensor 
align_to = rs.stream.color
align = rs.align(align_to)

# Start streaming
profile = pipeline.start(config)

# Information to be saved 
N= 5000#number of samples
pose= np.zeros((N,15))
k=0


# Reading cam2base pose 

##################CHANGE_IF_NEEDED##################
chosen_method= 2 # Park
##################CHANGE_IF_NEEDED##################
# get parent directory
path = os.getcwd()
parent= os.path.abspath(os.path.join(path, os.pardir)) 
##################CHANGE_IF_NEEDED##################
cameraPose_filepath= parent+'/Eye-to-hand Calibration/Calibration Data/cam2base.xlsx'
##################CHANGE_IF_NEEDED##################
data = pd.read_excel(cameraPose_filepath, sheet_name='method'+str(chosen_method))
T= data.head()
T= T.to_numpy()
(R_cam2base,t_cam2base)= rt_from_matrix(T)



try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get  frames
        color_frame = frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

       
       
        if  not color_frame or not depth_frame:
            continue

        # Convert image to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        

        # Get intrinsics for RGB sensor
        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics


        # Extract camera matrix and distortion coefficients
        mat = [color_intrinsics.fx, 0,color_intrinsics.ppx,0 ,color_intrinsics.fy , color_intrinsics.ppy, 0, 0, 1]
        mat = np.array(mat)
        camera_matrix= mat.reshape((3, 3))
        dist_coeffs= np.array(color_intrinsics.coeffs)


        ## Detection of ArUco Board

        # Detection Parameters 
        arucoParams = aruco.DetectorParameters_create()

        # Tuned paramters. If commented, the paramters are set to default values
        arucoParams.adaptiveThreshWinSizeMin=5
        arucoParams.adaptiveThreshWinSizeMax=20
        arucoParams.adaptiveThreshWinSizeStep=5
        arucoParams.cornerRefinementMethod= aruco.CORNER_REFINE_SUBPIX
        arucoParams.cornerRefinementWinSize=3
        arucoParams.cornerRefinementMinAccuracy=0.01
        arucoParams.cornerRefinementMaxIterations=50
        
        # Board's Parameters 
        ##################CHANGE_IF_NEEDED##################
        aruco_dict = aruco.Dictionary_get( aruco.DICT_5X5_1000 )
        markersX = 3
        markersY = 2
        markerLength = 120 # in mm 
        markerSeparation = 10 # in mm 
        ##################CHANGE_IF_NEEDED##################
        board = aruco.GridBoard_create(markersX, markersY, float(markerLength),
                                                    float(markerSeparation), aruco_dict)


        # Detection: we start by detecting all markers 
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image, aruco_dict, parameters=arucoParams)  # First, detect markers
        aruco.refineDetectedMarkers(gray_image, board, corners, ids, rejectedImgPoints)

        if len(corners) > 0: # if there is at least one marker detected
            # Estimate board pose 
            markNum, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs, rvec=np.float32([0,0,0]), tvec=np.float32([0,0,0]))  
            # tvec(in mm) is the 3D postion of the board frame wrt camera frame 
            # rvec(in rad) is the the rotaion vector of the board frame wrt camera frame, expressed in axis-angle representation
                        
            if markNum != 0:
                # Replace tvec with 3D coordinates of  deprojected pixel corresponding to any point of the board
                # Here, we take the center of the marker of ID 0 or 3. Note that we take these
                # IDs because we assume that there will always be one of them detected 
                found_ID= False
                ##################CHANGE_IF_NEEDED##################
                id1=0
                id2=3
                ##################CHANGE_IF_NEEDED##################
                # look first for maker with ID= id1
                for j in range(len(ids)): 
                    if ids[j][0]==id1:
                        found_ID= True
                        break
                # If ID 0 not found, look for ID= id2    
                if found_ID==False:
                    for j in range(len(ids)): 
                        if ids[j][0]==id2:
                            break
                
                if not found_ID:
                    continue
                # Get pixel coordinates of the center of chosen marker 
                x=0 
                y=0
                for i in range(4):
                    x=x+corners[j][0][i][0]
                    y=y+corners[j][0][i][1]

                x= int(x/4)
                y= int(y/4)


                 # Deprojection 
                depth= depth_frame.get_distance(x,y)
                deprojected_point = rs.rs2_deproject_pixel_to_point(color_intrinsics,[x,y], depth)
                # new tvec
                tvec=np.array(deprojected_point)*1000# in mm
               
                # Now, you have a board frame with the following transformation wrt camera frame 
                (R_board2cam, jac) = cv2.Rodrigues(rvec) # go to rotation matrix representation 
                t_board2cam = tvec


                # Get rotation matrix of board frame wrt base 
                R_board2base= np.dot(R_cam2base,R_board2cam)
                # Apply rotation to align the board frame to gripper frame 
                # rotation around y by 180 degrees
                betta= 3*(np.pi/2)
                R_board2base=np.dot(R_board2base,Roty(betta))
                # rotation around z by 90 degrees
                gamma= 3*(np.pi/2)
                R_board2base=np.dot(R_board2base,Rotz(gamma))
                (rvec, jac) = cv2.Rodrigues(R_board2base)
                # Now, we have the orientation of the desired grabbing frame
                # we need to get the 3D coordiates of its center


                # We can transform a point in board frame to base frame 
                # This point is where we desire to grab the block.
                # The coordinates of desired point in board frame(in mm) are: 
                id= ids[j][0]

                # This is the final grabbing point
                 ##################CHANGE_IF_NEEDED##################
                if id==id1: 
                    # Done for id1=0 
                    pt_wrt_board2= np.array([ 95+30+240+20+60 ,-60 ,-55]) 
                if id==id2:
                    # Done for id2=3 
                    pt_wrt_board2= np.array([ 95+30+240+20+60 , 60+10 ,-55]) 
                
                # This is an intermediate grabbing point
                # This is the same point displaced 300 mm in the y direction in the board frame 
                if id==id1:
                    # Done for id1=0 
                    pt_wrt_board1= np.array([ 95+30+240+20+60,-60+300 ,-55]) 
                if id==id2:
                    # Done for id2=3
                    pt_wrt_board1= np.array([ 95+30+240+20+60 , 60+10+300 ,-55]) 
                 ##################CHANGE_IF_NEEDED##################
                ## Transforming of point 1 from board frame to camera frame 

                # Rotation 
                rotated_point= np.dot(R_board2cam,pt_wrt_board1)
                rotated_point= np.ravel(rotated_point, order='C') # make it one dimensional numpy.ndarray
                # Translation 
                a= t_board2cam[0]+rotated_point[0]
                b= t_board2cam[1]+rotated_point[1]
                c= t_board2cam[2]+rotated_point[2]
                pt_wrt_cam= [a,b,c]       

                ## Transforming of point 1 from camera frame to base frame 

                # Rotation 
                rotated_point= np.dot(R_cam2base,pt_wrt_cam)
                rotated_point= np.ravel(rotated_point, order='C') # make it one dimensional numpy.ndarray

                # Translation 
                a= t_cam2base[0]+rotated_point[0]
                b= t_cam2base[1]+rotated_point[1]
                c= t_cam2base[2]+rotated_point[2]

                pt_wrt_base1= [a,b,c] #intermediate grabbing point wrt base frame 


                ## Transforming of point 2 from board frame to camera frame 

                # Rotation 
                rotated_point= np.dot(R_board2cam,pt_wrt_board2)
                rotated_point= np.ravel(rotated_point, order='C') # make it one dimensional numpy.ndarray

                # Translation 
                a= t_board2cam[0]+rotated_point[0]
                b= t_board2cam[1]+rotated_point[1]
                c= t_board2cam[2]+rotated_point[2]
                pt_wrt_cam= [a,b,c]       

                ## Transforming of point 2 from camera frame to base frame 
                # Rotation 
                rotated_point= np.dot(R_cam2base,pt_wrt_cam)
                rotated_point= np.ravel(rotated_point, order='C') # make it one dimensional numpy.ndarray

                # Translation 
                a= t_cam2base[0]+rotated_point[0]
                b= t_cam2base[1]+rotated_point[1]
                c= t_cam2base[2]+rotated_point[2]
                pt_wrt_base2= [a,b,c] # final grabbing point wrt base frame 
                
                # Get Euler angles of board frame to camera frame 
                EulerAng= rotationMatrixToEulerAngles(R_board2cam)
                
                ## Information to be saved
                
                # We need these for the static block criteria
                pose[k][0]= t_board2cam[0]
                pose[k][1]= t_board2cam[1]
                pose[k][2]= t_board2cam[2]
                pose[k][3]= EulerAng[0]
                pose[k][4]= EulerAng[1]
                pose[k][5]= EulerAng[2]

                # We need theses to construct the  T matrix of the final and itermediate grabbing poses
                pose[k][6]= rvec[0]
                pose[k][7]= rvec[1]
                pose[k][8]= rvec[2]
                pose[k][9]= pt_wrt_base2[0]
                pose[k][10]= pt_wrt_base2[1]
                pose[k][11]= pt_wrt_base2[2]
                pose[k][12]= pt_wrt_base1[0]
                pose[k][13]= pt_wrt_base1[1]
                pose[k][14]= pt_wrt_base1[2]


                # Static block criteria
                ##################CHANGE_IF_NEEDED##################
                Buffer_size= 20
                ##################CHANGE_IF_NEEDED##################
                # These limits are obtained from a staric block test( mentioned in section 5.1 in the report )
                # They are increased a bit to not wait a lot of time for block to settle 
                ##################CHANGE_IF_NEEDED##################
                limit_rot=1.5 # degrees
                limit_pos= 30 # mm 
                ##################CHANGE_IF_NEEDED##################
                if k>Buffer_size:
                    buffer_x= pose[k-Buffer_size:k,0]
                    buffer_y=pose[k-Buffer_size:k,1]
                    buffer_z= pose[k-Buffer_size:k,2]
                    buffer_roll= pose[k-Buffer_size:k,3]
                    buffer_pitch=pose[k-Buffer_size:k,4]
                    buffer_yaw= pose[k-Buffer_size:k,5]

                    # we apply a moving standard deviation on the previous pose elements 
                    rot_cond= np.std(buffer_roll)< limit_rot and np.std(buffer_pitch)< limit_rot and np.std(buffer_yaw)< limit_rot # degrees
                    pos_cond= np.std(buffer_x)< limit_pos and np.std(buffer_y)< limit_pos and np.std(buffer_z)< limit_pos # mm
                    
                    if pos_cond and rot_cond:  #block is static 
                        print("Block is static")
                        buffer_x1= pose[k-Buffer_size:k,12]
                        buffer_y1=pose[k-Buffer_size:k,13]
                        buffer_z1= pose[k-Buffer_size:k,14]
                        buffer_x2= pose[k-Buffer_size:k,9]
                        buffer_y2=pose[k-Buffer_size:k,10]
                        buffer_z2= pose[k-Buffer_size:k,11]
                        buffer_rvec1= pose[k-Buffer_size:k,6]
                        buffer_rvec2=pose[k-Buffer_size:k,7]
                        buffer_rvec3= pose[k-Buffer_size:k,8]
                         # Take the mean of the buffer instead of one value ( more accurate)
                        rvec=np.array([np.mean(buffer_rvec1),np.mean(buffer_rvec2),np.mean(buffer_rvec3)])
                        tvec[0]= np.mean(buffer_x1)*0.001 # in m
                        tvec[1]= np.mean(buffer_y1)*0.001# in m
                        tvec[2]= np.mean(buffer_z1)*0.001# in m

                        T= matrix_from_rtvec(rvec,tvec)
                         # save pose in txt file: intermediate grabbing point 
                        df = pd.DataFrame(T)
                        ##################CHANGE_IF_NEEDED##################
                        df.to_csv(r'T_mat_interm.txt', header=None, index=None, sep='\t', mode='w')
                       ##################CHANGE_IF_NEEDED##################
                        # Take the mean of the buffer instead of one value ( more accurate)
                        tvec[0]= np.mean(buffer_x2)*0.001# in m
                        tvec[1]= np.mean(buffer_y2)*0.001# in m
                        tvec[2]= np.mean(buffer_z2)*0.001# in m
                        T= matrix_from_rtvec(rvec,tvec)
                        # save pose in txt file: final grabbing point 
                        df = pd.DataFrame(T)
                        ##################CHANGE_IF_NEEDED##################
                        df.to_csv(r'T_mat_final.txt', header=None, index=None, sep='\t', mode='w')
                        ##################CHANGE_IF_NEEDED##################
                        print("Final are intermediate grabbing poses are saved!")
                        break

                
                k=k+1

              
                    
                   


finally:

    # Stop streaming
    pipeline.stop()







