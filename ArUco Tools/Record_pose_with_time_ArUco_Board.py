############################################################
## BRUFACE: Brussels Faculty of Engineering 
############################################################
## Master Thesis " Autonmous Grabbing with RGB-D"  
## July 2022 Manar Mahmalji      
## Version (1)     
############################################################
## This code serves for recording the estimated pose of an ArUco board
## in the form of 3D postion and euler angles with respect to camera frame
##  over a fixed time interval 


import pyrealsense2 as rs
import numpy as np
import cv2.aruco as aruco
import cv2
import math
import pandas as pd 
import time 


# Useful Functions 
# Go from T matrix to rotation matrix r and translation vector t 
def rt_from_matrix(M):
    r= M[0:3, 0:3] 
    t=  M[0:3, 3]
    return (r, t)

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

# Information to be recorded 
##################CHANGE_IF_NEEDED##################
N= 1000
##################CHANGE_IF_NEEDED##################
pose= np.zeros((N,6))
timeMeas= np.zeros((N,1)) 
stand_dev= np.zeros((N,1)) 
k=0 # counter 

# Start timer 
start= time.time()
##################CHANGE_IF_NEEDED##################
recordingTime= 5 # in seconds 
##################CHANGE_IF_NEEDED##################
print("Recording...")


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
            # Draw detected markers and diplay their IDs 
            im_with_aruco_board=aruco.drawDetectedMarkers(color_image, corners, ids)
            # Estimate board pose 
            markNum, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs, rvec=np.float32([0,0,0]), tvec=np.float32([0,0,0]))  
            # tvec(in mm) is the 3D postion of the board frame wrt camera frame 
            # rvec(in rad) is the the rotaion vector of the board frame wrt camera frame, expressed in axis-angle representation
            
            if markNum != 0:
                # Replace tvec with 3D coordinates of  deprojected pixel corresponding to any point of the board
                # Here, we take the center of the marker of ID 0 or 3. Note that we take these
                # IDs be cause we assume that there will always be one of them detected 
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

                # Draw pose 
                im_with_aruco_board = cv2.drawFrameAxes(im_with_aruco_board, camera_matrix, dist_coeffs, rvec, tvec, 100)  # axis length 100 can be changed according to your requirement
             
               
                # Now, you have a board frame with the following transformation wrt camera frame 
                (R_board2cam, jac) = cv2.Rodrigues(rvec) # go to rotation matrix representation 
                t_board2cam = tvec

                # Get Euler Angles of baord frame wrt caemra frame
                EulerAng= rotationMatrixToEulerAngles(R_board2cam)
                
                
                # Record pose and time 
                timeMeas[k][0]= time.time()-start
                pose[k][0]= t_board2cam[0]
                pose[k][1]= t_board2cam[1]
                pose[k][2]= t_board2cam[2]
                pose[k][3]= EulerAng[0]
                pose[k][4]= EulerAng[1]
                pose[k][5]= EulerAng[2]

                # Increase counter 
                k=k+1

                # Stopping condition 
                if time.time()-start> recordingTime or k==N:
                    stand_dev[0][0]= np.std(pose[0:k,0])
                    stand_dev[1][0]= np.std(pose[0:k,1])
                    stand_dev[2][0]= np.std(pose[0:k,2])
                    stand_dev[3][0]= np.std(pose[0:k,3])
                    stand_dev[4][0]= np.std(pose[0:k,4])
                    stand_dev[5][0]= np.std(pose[0:k,5])
                    array=np.hstack((timeMeas,pose,stand_dev))
                    ## convert your array into a dataframe
                    df = pd.DataFrame (array)
                    ## save to xlsx file
                    ##################CHANGE_IF_NEEDED##################
                    filepath = 'measPose_aruco_board.xlsx'
                    ##################CHANGE_IF_NEEDED##################
                    df.to_excel(filepath,sheet_name='Sheet1',header=['Time','x','y','z','roll','pitch','yaw','std:x,y,z,roll,pitch,yaw'])
                
                    print("Recording is finished!")
                    break

        else:
            im_with_aruco_board = color_image

        # Display board 
        cv2.imshow("ArUco Board", im_with_aruco_board)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:

    # Stop streaming
    pipeline.stop()







