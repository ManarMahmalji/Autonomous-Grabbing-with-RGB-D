############################################################
## BRUFACE: Brussels Faculty of Engineering 
############################################################
## Master Thesis " Autonmous Grabbing with RGB-D"  
## July 2022 Manar Mahmalji      
## Version (1)     
############################################################
## This code serves for recording the estimated pose of an ArUco marker
## in the form of 3D postion and euler angles with respect to camera frame
##  over a fixed time interval 


import pyrealsense2 as rs
import numpy as np
import cv2.aruco as aruco
import cv2
import math
import pandas as pd 
import time 

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


        ## Detection of ArUco Marker

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
        
        # Marker's Parameters 
        ##################CHANGE_IF_NEEDED##################
        aruco_dict = aruco.Dictionary_get( aruco.DICT_5X5_1000 )
        markerSize=120 # in mm 
        markerID= 0
        ##################CHANGE_IF_NEEDED##################

        # Detection: we start by detecting all markers 
        gray_image= cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image, aruco_dict, parameters=arucoParams)  


        if len(corners) > 0: # if there is at least one marker detected
            # Draw detected markers and diplay their IDs 
            im_with_aruco_marker=aruco.drawDetectedMarkers(color_image, corners, ids)

            # look for the marker with ID= markerID
            found_ID= False
            for j in range(len(ids)): 
                if ids[j][0]==markerID:
                    found_ID= True
                    break
            

            if found_ID:
                # Estimate marker's pose 
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[j], markerSize, camera_matrix,dist_coeffs)
                # tvec(in mm) is the 3D postion of the marker frame wrt camera frame 
                # rvec(in rad) is the the rotaion vector of the marker frame wrt camera frame, expressed in axis-angle representation
                
                # Replace tvec with 3D coordinates of deprojected pixel corresponding to marker's center
            
                # Get pixel coordinates of marker's center 
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

                # Draw pose 
                im_with_aruco_marker = cv2.drawFrameAxes(im_with_aruco_marker, camera_matrix, dist_coeffs, rvec, tvec, 100)  # axis length 100 can be changed according to your requirement
                
                # Now, you have a marker frame with the following transformation wrt camera frame 
                (R_mark2cam, jac) = cv2.Rodrigues(rvec) # go to rotation matrix representation 
                t_mark2cam = tvec

                print('Position')
                print(tvec)
                print('Orientation')
                print(rvec)

    


        else:
            im_with_aruco_marker = color_image

        # Display board 
        cv2.imshow("ArUco Marker", im_with_aruco_marker)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:

    # Stop streaming
    pipeline.stop()







