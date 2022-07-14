############################################################
## BRUFACE: Brussels Faculty of Engineering 
############################################################
## Master Thesis " Autonmous Grabbing with RGB-D"  
## July 2022 Manar Mahmalji      
## Version (1)     
############################################################
## This code serves for creating an ArUco board 

import cv2.aruco as aruco
import cv2


##################CHANGE_IF_NEEDED##################
markersX = 3
markersY = 2
markerLength = 120 # in mm 
markerSeparation = 10 # in mm 
margins = 0
##################CHANGE_IF_NEEDED##################

width= markersX * (markerLength + markerSeparation) - markerSeparation + 2 * margins
height= markersY * (markerLength + markerSeparation) - markerSeparation + 2 * margins
 
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)
board = aruco.GridBoard_create(markersX, markersY, float(markerLength),
                                                    float(markerSeparation), aruco_dict)
##################CHANGE_IF_NEEDED##################
scale= 5
##################CHANGE_IF_NEEDED##################
boardImage=board.draw((width*scale,height*scale),marginSize = margins, 	borderBits = 1)
##################CHANGE_IF_NEEDED##################
cv2.imwrite('Board.png', boardImage)
##################CHANGE_IF_NEEDED##################

# Image is rescaled and printed in the image viewer of Ubuntu. Pay attention to 
# put the unit in mm then rescale it to what is expected in mm. Also, choose the
# type of paper ( A3 for example)
# If picture is too small, increase the variable scale 





