############################################################
## BRUFACE: Brussels Faculty of Engineering 
############################################################
## Master Thesis " Autonmous Grabbing with RGB-D"  
## July 2022 Manar Mahmalji      
## Version (1)     
############################################################
## This code serves for recording the estimated Euler angles of the 
## of an ArUco board wrt the robot base frame using the the cam2base 
## transformations obtained from the 5 calibration methods 

 
import pyrealsense2 as rs
import numpy as np
import cv2.aruco as aruco
import cv2
import math
import pandas as pd 
import time 
import os 


# Useful Functions 
# Go from T matrix to rvec and tvec 
def rt_from_matrix(M):
    r= M[0:3, 0:3] 
    t=  M[0:3, 3]
    return (r, t)

# Check if a matrix is a valid rotation matrix.
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



# Reading cam2base pose for 5 methods

R_cam2base_list=[]
t_cam2base_list=[]

# get parent directory
path = os.getcwd()
parent= os.path.abspath(os.path.join(path, os.pardir)) 
##################CHANGE_IF_NEEDED##################
cameraPose_filepath= parent+'/Calibration Data/cam2base.xlsx'
##################CHANGE_IF_NEEDED##################
for i in range(5):
    data = pd.read_excel(cameraPose_filepath, sheet_name='method'+str(i+1))
    T= data.head()
    T= T.to_numpy()
    (R_cam2base,t_cam2base)= rt_from_matrix(T)
    R_cam2base_list.append(R_cam2base)
    t_cam2base_list.append(t_cam2base)



# Pararmters used for taking the average of read data over a time interval
start =time.time() 
##################CHANGE_IF_NEEDED##################
reading_time= 5 # seconds  
##################CHANGE_IF_NEEDED##################
x_sum=[0,0,0,0,0] 
y_sum=[0,0,0,0,0]
z_sum=[0,0,0,0,0]
samplesNum=0 


print("Reading...")


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
               
                # Draw pose 
                im_with_aruco_board = cv2.drawFrameAxes(im_with_aruco_board, camera_matrix, dist_coeffs, rvec, tvec, 100)  # axis length 100 can be changed according to your requirement
                
                # Now, you have a board frame with the following transformation wrt camera frame 
                (R_board2cam, jac) = cv2.Rodrigues(rvec) # go to rotation matrix representation 
                t_board2cam = tvec


                for i in range(5):
                    # Get the cam2base transofrmation corresponding to the ith method
                    R_cam2base= R_cam2base_list[i]
                    t_cam2base= t_cam2base_list[i]

                    # Get rotation matrix of board frame wrt base 
                    R_board2base= np.dot(R_cam2base,R_board2cam)

                    # Apply relative trasnforamtion to align the board frame to gripper frame 
                    # rotation around y by 180 degrees
                    betta= np.pi
                    R_board2base=np.dot(R_board2base,Roty(betta))
                    # rotation around z by -90 degrees
                    gamma= 3*(np.pi/2)
                    R_board2base=np.dot(R_board2base,Rotz(gamma))

                    # Adding readings to get average in the end 
                    (rvec2, jac) = cv2.Rodrigues(R_board2base) 
                    x_sum[i]= x_sum[i] + rvec2[0]
                    y_sum[i]= y_sum[i] + rvec2[1]
                    z_sum[i]= z_sum[i] + rvec2[2]
                    
                samplesNum= samplesNum+1

                if (time.time()-start)> reading_time:
                    # save in an excel sheet
                    ##################CHANGE_IF_NEEDED##################
                    filepath = 'orientation_validation.xlsx'
                    ##################CHANGE_IF_NEEDED##################
                    writerEngine = pd.ExcelWriter(filepath, mode="w", engine="openpyxl")
                    #Divide sum by samples number to get average rotation matrix
                    Tsai=np.array([x_sum[0]/samplesNum, y_sum[0]/samplesNum,z_sum[0]/samplesNum])
                    (Tsai, Jac)= cv2.Rodrigues(Tsai)
                    Park=np.array([x_sum[1]/samplesNum, y_sum[1]/samplesNum,z_sum[1]/samplesNum])
                    (Park, Jac)= cv2.Rodrigues(Park)
                    Horaud=np.array([x_sum[2]/samplesNum, y_sum[2]/samplesNum,z_sum[2]/samplesNum])
                    (Horaud, Jac)= cv2.Rodrigues(Horaud)
                    Andreff=np.array([x_sum[3]/samplesNum, y_sum[3]/samplesNum,z_sum[3]/samplesNum])
                    (Andreff, Jac)= cv2.Rodrigues(Andreff)
                    Daniilidis=np.array([x_sum[4]/samplesNum, y_sum[4]/samplesNum,z_sum[4]/samplesNum])
                    (Daniilidis, Jac)= cv2.Rodrigues(Daniilidis)
                    # Save rotation matricx for each method
                    df1 = pd.DataFrame(Tsai)
                    df1.to_excel(writerEngine, sheet_name='Tsai')
                    df2 = pd.DataFrame(Park)
                    df2.to_excel(writerEngine, sheet_name='Park')
                    df3 = pd.DataFrame(Horaud)
                    df3.to_excel(writerEngine, sheet_name='Horaud')
                    df4 = pd.DataFrame(Andreff)
                    df4.to_excel(writerEngine, sheet_name='Andreff')
                    df5 = pd.DataFrame(Daniilidis)
                    df5.to_excel(writerEngine, sheet_name='Daniilidis')

                    # save Euler angles for each method
                    Tsai= rotationMatrixToEulerAngles(Tsai)
                    Park= rotationMatrixToEulerAngles(Park)
                    Horaud=rotationMatrixToEulerAngles(Horaud)
                    Andreff= rotationMatrixToEulerAngles(Andreff)
                    Daniilidis= rotationMatrixToEulerAngles(Daniilidis)
                    df6 = pd.DataFrame([Tsai, Park, Horaud, Andreff, Daniilidis],
                     index=['Tsai', 'Park','Horaud','Andreff', 'Daniilidis'],
                     columns=['roll','pitch','yaw'])
                    df6.to_excel(writerEngine, sheet_name='Euler Angles')

                    writerEngine.save()
                    print('ArUco Board orientation wrt base retrieved')
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







