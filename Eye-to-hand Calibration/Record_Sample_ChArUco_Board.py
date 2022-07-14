############################################################
## BRUFACE: Brussels Faculty of Engineering 
############################################################
## Master Thesis " Autonmous Grabbing with RGB-D"  
## July 2022 Manar Mahmalji      
## Version (1)     
############################################################
## This code serves for recording the pose in the from of T matrix
## of a ChArUco board for one time in an excel sheet


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
    r= M[0:3, 0:3] 
    t=  M[0:3, 3]
    return (r, t)

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
config.enable_stream(rs.stream.color, 1280 , 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)



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

# timer for oscillations
start= time.time()

reading_time= 3 # seconds
x_sum_t=0
y_sum_t=0
z_sum_t=0
x_sum_r=0
y_sum_r=0
z_sum_r=0
samplesNum=0 
value_recorded= False


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

        # Convert image to numpy array in order to use it in OpenCV
        color_image = np.asanyarray(color_frame.get_data())
        

        # Get intrinsics for RGB sensor
        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics


        # Extract camera matrix and distortion coefficients
        mat = [color_intrinsics.fx, 0,color_intrinsics.ppx,0 ,color_intrinsics.fy , color_intrinsics.ppy, 0, 0, 1]
        mat = np.array(mat)
        camera_matrix= mat.reshape((3, 3))
        dist_coeffs= np.array(color_intrinsics.coeffs)
        ##################CHANGE_IF_NEEDED##################
        # set up excel file
        filepath = 'Calibration Data/test.xlsx'
        ##################CHANGE_IF_NEEDED##################
        try:
            # append to xlsx file in new sheet
            writer = pd.ExcelWriter(filepath, mode="a", engine="openpyxl")
        except:
            # If file does not exist, create a new one the save it
            writer = pd.ExcelWriter(filepath, engine="xlsxwriter")
            writer.save()
            # append to xlsx file in new sheet
            writer = pd.ExcelWriter(filepath, mode="a", engine="openpyxl")
                
                
        ## Detection of ChArUco Board

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

         ##################CHANGE_IF_NEEDED##################
        # Board's Parameters 
        aruco_dict = aruco.Dictionary_get( aruco.DICT_4X4_1000 )
        squareLength = 47   #in mm
        markerLength = 37  #in mm
        board = aruco.CharucoBoard_create(5, 7, squareLength, markerLength, aruco_dict)
         ##################CHANGE_IF_NEEDED##################

        # Detection: we start by detecting all markers 
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image, aruco_dict, parameters=arucoParams)  
     

        if len(corners) > 0: # if there is at least one marker detected
            # Estimate board pose 
            charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, gray_image, board)
            retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, camera_matrix, dist_coeffs, rvec=np.float32([0,0,0]), tvec=np.float32([0,0,0])) 
            # tvec(in mm) is the 3D postion of the board frame wrt camera frame 
            # rvec(in rad) is the the rotaion vector of the board frame wrt camera frame, expressed in axis-angle representation
            
            if retval: # if pose estimation is successful 
                x_sum_t= x_sum_t + tvec[0]
                y_sum_t= y_sum_t + tvec[1]
                z_sum_t= z_sum_t + tvec[2] 
                x_sum_r= x_sum_r + rvec[0]
                y_sum_r= y_sum_r + rvec[1]
                z_sum_r= z_sum_r + rvec[2]
                samplesNum= samplesNum+1

            if retval and time.time()-start> reading_time: 
                # we take the average pose after reading time has elapsed 
                tvec=np.array([x_sum_t/samplesNum, y_sum_t/samplesNum,z_sum_t/samplesNum])
                rvec=np.array([x_sum_r/samplesNum, y_sum_r/samplesNum,z_sum_r/samplesNum])
                # Transformation resulting from rvec and tvec 
                T=matrix_from_rtvec(rvec,tvec) 
                # convert your matrix into a dataframe and save in excel sheet
                df = pd.DataFrame (T)
                df.to_excel(writer, index=False)
                writer.save()
                print("Recording is finished!")
                break

finally:

    # Stop streaming
    pipeline.stop()







