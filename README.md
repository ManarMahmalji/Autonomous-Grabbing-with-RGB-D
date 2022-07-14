
# BRUFACE: Brussels Faculty of Engineering 
## Master Thesis " Autonmous Grabbing with RGB-D"  
## July 2022 Manar Mahmalji 
### Software versions 
• Operating system: Ubuntu 20.04.4 LTS

• OpenCV: 4.5.5

• Python: 3.8.10

• ROS: Noetic

This repository contains all the scripts used in the master thesis entitled " Autonmous Grabbing with RGB-D", in which an Intel Realsense D455 camera is used to enable a Kuka IIWA 14 robotic arm grab and track a suspended block with the help of ArUco markers. The full report can be found [here](https://github.com/ManarMahmalji/Autonomous-Grabbing-with-RGB-D/files/9111195/2022_EM_Meca_Manar_Mahmalji-compressed.pdf).

### Contents: 
- ArUco tools: utiltiy tools for displaying and recoding the poses of ArUco markers and board wrt camera frame. 
- Eye-to-hand Calibration: this corresponds to chapter 4 of the thesis, where the transforamtion from the camera frame to the robot base is obtained. The calibration is performed with OpenCV and there are several scripts that help organize and validate the calibration data. A detailed ReadMe file is present for this process. 
<p align="center" width="100%">
    <img width="80%" src="https://user-images.githubusercontent.com/31004981/178968130-a5f84c4c-e77f-422e-b9bd-4f80a9702f36.png">
</p>

- Grabbing: this corresponds to Chapter 5 of the the thesis, where the grabbing of a suspened block is achieved, only after it becomes static. A video demonstration can be found [here](https://www.youtube.com/watch?v=c5iViaV8wGg&list=PLc2DKdGuH4n82zvEc9ee7eKdp0i5x4-w1&index=1&t=6s).
<p align="center">
  <img  width="400" src="https://user-images.githubusercontent.com/31004981/178968234-53c6100c-5db9-4adb-b476-7aa60457e6ab.png">
  <img width="400" src="https://user-images.githubusercontent.com/31004981/178968318-3511799d-7010-46c9-876d-27d311621b81.png">
</p>

- Pose Tracking: this corresponds to Chapter 6 of the the thesis, where the tracking of the grabbing pose of a block is achieved in ROS. A video demonstration can be found [here](https://www.youtube.com/watch?v=BczmKoiRnCo&list=PLc2DKdGuH4n82zvEc9ee7eKdp0i5x4-w1&index=2).
<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/31004981/178968382-949411b3-7671-42b4-b812-54ad38c3ed79.png">
  <img width="400" src="https://user-images.githubusercontent.com/31004981/178968454-d193cd94-649c-439d-baa8-ee13696fac61.png">
</p>

- MATLAB: this corresponds to the ambiguity measurements shown in section 3.4 and the static block criteria shown in section 5.2 
