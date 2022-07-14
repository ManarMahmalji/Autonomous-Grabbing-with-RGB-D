#!/usr/bin/env python3
############################################################
## BRUFACE: Brussels Faculty of Engineering 
############################################################
## Master Thesis " Autonmous Grabbing with RGB-D"  
## July 2022 Manar Mahmalji      
## Version (1)     
############################################################
# This ROS node continously sends the grabbing pose of a 
# block for the sake of tracking its movement 


import rospy
from std_msgs.msg  import Float64MultiArray
import numpy as np
import pandas as pd   
import pyrealsense2 as rs
import cv2.aruco as aruco
import cv2
import math
import time 

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


def talker():
    # publisher setup 
    ##################CHANGE_IF_NEEDED##################
    pub = rospy.Publisher('ArUcoPose', Float64MultiArray, queue_size=1)
    rospy.init_node('IntelD455', anonymous=True)
    ##################CHANGE_IF_NEEDED##################

    ## camera setup 

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
    # config.enable_stream(rs.stream.color, 1280 , 720, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    # Higher FPS, lower resolution 
    # config.enable_stream(rs.stream.color, 848 ,480, rs.format.bgr8, 60)
    # config.enable_stream(rs.stream.depth, 848 ,480, rs.format.z16, 60)

    # Higher FPS, lower resolution 
    config.enable_stream(rs.stream.color, 640 ,360, rs.format.bgr8, 90)
    config.enable_stream(rs.stream.depth, 640 ,360, rs.format.z16, 90)
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
    pose= np.zeros((N,6))
    k=0


    # Reading cam2base pose 
    ##################CHANGE_IF_NEEDED##################
    chosen_method= 2 # Park
    ##################CHANGE_IF_NEEDED##################
    cameraPose_filepath= '/home/manar/Master Thesis/Eye-to-hand Calibration/Calibration Data/cam2base.xlsx'
    ##################CHANGE_IF_NEEDED##################
    data = pd.read_excel(cameraPose_filepath, sheet_name='method'+str(chosen_method))
    T= data.head()
    T= T.to_numpy()
    (R_cam2base,t_cam2base)= rt_from_matrix(T)



    T_mat= Float64MultiArray() # instantiate an object
    ##################CHANGE_IF_NEEDED##################
    rate = rospy.Rate(40) # Hz 
    ##################CHANGE_IF_NEEDED##################
    

    while not rospy.is_shutdown():
        start= time.time()
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
 
        # Default pose , -1 is used in the beginning to indicate that no pose is taken from the camera 
        T_mat.data= np.array([-1, 0,0,0,0,0,0,0,0,0,0,0,0])
        if len(corners)>0: # if there is at least one marker detected
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
                    pt_wrt_board= np.array([ 95+30+240+20+60 ,-60 ,-55]) 
                if id==id2:
                    # Done for id2=3 
                    pt_wrt_board= np.array([ 95+30+240+20+60 , 60+10 ,-55]) 
        
                 ##################CHANGE_IF_NEEDED##################
    

                ## Transforming of grabbing point from board frame to camera frame 

                # Rotation 
                rotated_point= np.dot(R_board2cam,pt_wrt_board)
                rotated_point= np.ravel(rotated_point, order='C') # make it one dimensional numpy.ndarray

                # Translation 
                a= t_board2cam[0]+rotated_point[0]
                b= t_board2cam[1]+rotated_point[1]
                c= t_board2cam[2]+rotated_point[2]
                pt_wrt_cam= [a,b,c]       

                ## Transforming of grabbing point from camera frame to base frame 
                # Rotation 
                rotated_point= np.dot(R_cam2base,pt_wrt_cam)
                rotated_point= np.ravel(rotated_point, order='C') # make it one dimensional numpy.ndarray

                # Translation 
                a= t_cam2base[0]+rotated_point[0]
                b= t_cam2base[1]+rotated_point[1]
                c= t_cam2base[2]+rotated_point[2]
                pt_wrt_base= [a,b,c] # final grabbing point wrt base frame 
                
                
                # so you have got R_board2base and pt_wrt_base
                pose[k][0]= pt_wrt_base[0]
                pose[k][1]= pt_wrt_base[1]
                pose[k][2]= pt_wrt_base[2]
                pose[k][3]= rvec[0]
                pose[k][4]= rvec[1]
                pose[k][5]= rvec[2]


                derv_pos_limit= 20 #mm/s
                derv_rot_limit= 20*math.pi/180 #rad/s
                mov_avg_size= 20 # moving average filter window size 

                # detect spikes: if any, do not send pose 
                if k>0: 
                    derv_cond_pos= abs(pose[k][0]-pose[k-1][0]) > derv_pos_limit or abs(pose[k][1]-pose[k-1][1])> derv_pos_limit or  abs(pose[k][2]-pose[k-1][2])> derv_pos_limit 
                    derv_cond_rot= abs(pose[k][3]-pose[k-1][3]) > derv_rot_limit or abs(pose[k][4]-pose[k-1][4])> derv_rot_limit or  abs(pose[k][5]-pose[k-1][5])>derv_rot_limit
                    derv_cond= derv_cond_pos or derv_cond_rot

                else:
                    derv_cond= False
 
                if not derv_cond and k>mov_avg_size: 
                    # moving aveverage on tranlsation vector
                    t1=  np.mean(pose[k-mov_avg_size:k,0])*0.001
                    t2=  np.mean(pose[k-mov_avg_size:k,1])*0.001
                    t3=  np.mean(pose[k-mov_avg_size:k,2])*0.001
                    # moving average on rotation vector 
                    rvec= np.array([np.mean(pose[k-mov_avg_size:k,3]),np.mean(pose[k-mov_avg_size:k,4]),np.mean(pose[k-mov_avg_size:k,5])])
                    # Rotation matrix elements 
                    (R_board2base, jac) = cv2.Rodrigues(rvec) 
                    r11= R_board2base[0][0]
                    r12= R_board2base[0][1]
                    r13= R_board2base[0][2]
                    r21= R_board2base[1][0]
                    r22= R_board2base[1][1]
                    r23= R_board2base[1][2]
                    r31= R_board2base[2][0]
                    r32= R_board2base[2][1]
                    r33= R_board2base[2][2]
                    T_mat.data=np.array([1,t1,t2,t3,r11,r12,r13,r21,r22,r23,r31,r32,r33])
                    # If the first number in the array is 1,hence pose is taken, elseif it is -1 
                    # pose is not taken 
                    print(T_mat.data)
                else:
                    T_mat.data= np.array([-1, 0,0,0,0,0,0,0,0,0,0,0,0])
                    print('Pose Not taken')

        k=k+1          
        pub.publish(T_mat)
        rate.sleep()
        print("Frequency:"+ str(1/(time.time()-start)) + ' Hz')
        

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass