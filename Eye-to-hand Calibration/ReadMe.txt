############################################################
## BRUFACE: Brussels Faculty of Engineering 
############################################################
## Master Thesis " Autonmous Grabbing with RGB-D"  
## July 2022 Manar Mahmalji      
## Version (1)     
############################################################

It is advised to read chapter 4 of the thesis report entitled "Eye-to-Hand Calibration"
-----------------------------------------
Calibration Steps:
1- Create and print a calibration board. A common one is a ChArUco board which can be created using this online tool: https://calib.io/pages/camera-calibration-pattern-generator

2- Modify the board's parameters in the scripts: "Display_ChArUco_Board.py" and "Record_Sample_ChArUco_Board.py"

2- For the wooden plate that is screwed to the gripper, make sure it is rigid enough to the gripper. Also, if bolts reach their ends and wooden plate is not fixed, use nuts on bolts

3- After installing the wooden plate, tape the board to it 

3- Make sure there is no air pockets under the board( as best as you can) 

4- Open up the display with "Display_ChArUco_Board.py" and manipulate the robot to a desired positon. Positions should be taken as diversely as possible in all xyz directions of the base frame. Go to very close positions from the camera (the closer you are, the more accurate the results) then go furhter from the camera. When taking close positions, go the extent where you fill the frame of the camera with all the markers detected. charuco boards are accuratre if they are close to the camera ( preferrably in the range of 0.5-1 m ). 



5- Once you see on the display that a position is good( all markers are detected on the board, very small oscillations), then stop display, record the board pose with "Record_Sample_ChArUco_Board.py" and write your joint space vector in the MATLAB script "gripper2base.m". The "Record_Sample_ChArUco_Board.py" always appends the board pose to an excel file "target2cam.xlsx" in the Calibration Data directory. If it does not exists, it creates one. 

 /!\ In case you preform a new calibration, and you would like to keep the old data, rename , delete or move you files somewhere else. Otherwise, they will be overwritten. Note also that if there is already a "target2cam.xlsx" file in the Calibration Data directory, running "Record_Sample_ChArUco_Board.py" will append new samples to the current file hence the latter should either be renamed, deleted or moved elsewhere. The Calibration Data directory constains example files of what is expected to come out of the python scripts. ONLY the "target2cam.xlsx"file should be removed or renamed in case of new calibraion. For all the other files, they are overwritten by default.
 
 
6- Repeat the same process until taking at least 20 samples, then get the gripper pose wrt the robot base with the MATLAB script "gripper2base.m". Note that the MATLAB scripts overwrites the the file  "gripper2base.xlsx" in its current directory. Note also that you should move the resutling "gripper2base.xlsx" file to the Calibration Data directory.  After that, run the python script "EyeToHandCalib.py" to obtain the camera pose wrt the robot base. You need to insert the number of samples before running the script. "EyeToHandCalib.py" outputs a "cam2base.xlsx" file containing the T matrix of camera wrt base for the methods of calibration. Note that if a previous "cam2base.xlsx" exsists, the scripts overwrites it.

7- Check convergence using the python script "study_convergence.py". A more graphical visualization is done with the matlab script "study_convergence.m". 


-----------------------------------------
Validation:
1- Position: The end effector is centerd at the center of an Aruco marker. Its position wrt base is known and recorded. then we move away the end effector and we read the position of the center of the ArUco marker wrt camera frame and calculate the correspoding position wrt robot frame using the 5 differnet methods available in the eye2hand calibration function. Run the script "position_validation.py" and the data will be stored in the same directory in "positiion_validation.xlsx". If it is already there, it will be overwritten

2- Orientation
same thing as before but we use an ArUco board to read accurate orientation. we align the end effector with the board frame and record its rotation matrix then we move it away and get the rotaion matrix of board wrt base with the 5 differnt methods. Run the script "orientation_validation.py" and the data will be stored in the same directory in "orientation_validation.xlsx". If it is already there, it will be overwritten






