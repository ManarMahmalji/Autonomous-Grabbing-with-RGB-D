############################################################
## BRUFACE: Brussels Faculty of Engineering 
############################################################
## Master Thesis " Autonmous Grabbing with RGB-D"  
## July 2022 Manar Mahmalji      
## Version (1)     
############################################################
## This code serves for displaying the estimated pose of 
## a ChArUco board with respect to camera frame. An additional 
## feature is to record a video of the dispalyed stream 


import pyrealsense2 as rs
import numpy as np
import cv2.aruco as aruco
import cv2
import pandas as pd 


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

## Uncomment to record a video 
# Recording stream
 ##################CHANGE_IF_NEEDED##################
# width= 1280
# height= 720
# writer= cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'MPEG'), 20, (width,height))
 ##################CHANGE_IF_NEEDED##################


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
                # Draw pose and display IDs of detect markers 
                im_with_charuco_board = aruco.drawDetectedCornersCharuco(color_image, charucoCorners, charucoIds, (0,255,0))
                im_with_charuco_board =cv2.drawFrameAxes(im_with_charuco_board, camera_matrix, dist_coeffs, rvec, tvec, 100)  # axis length 100 can be changed according to your requirement
                print('Position')
                print(tvec)
                print('Orientation')
                print(rvec)

        else:
            im_with_charuco_board = color_image

        # Display board 
        cv2.imshow("ChArUco Board", im_with_charuco_board)
        ## Uncomment to record a video
        # writer.write(im_with_charuco_board)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:

    # Stop streaming
    pipeline.stop()







