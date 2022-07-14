
% ############################################################
% ## BRUFACE: Brussels Faculty of Engineering 
% ############################################################
% ## Master Thesis " Autonmous Grabbing with RGB-D"  
% ## July 2022 Manar Mahmalji      
% ## Version (1)     
% ############################################################
% ##This code is used to visualize the ambiguity problem. The Euler
% ##angles of obtained from an ArUco board are compared to those obtained from 
% ##one marker of the board

disp("hello")
close all
clear 

%%
filename1="measPose_ambiguity_parallel.xlsx"; %board parallel to camera ( little to no ambiguity)
filename2="measPose_ambiguity.xlsx"; %board and camera are not parallel

data= readmatrix(filename1);
time= data(:,2);
roll_b= data(:,3); % board
pitch_b= data(:,4);
yaw_b= data(:,5);
roll_m= data(:,6); % marker
pitch_m= data(:,7);
yaw_m= data(:,8);

std_vector= [std(roll_b), std(pitch_b),std(yaw_b), std(roll_m), std(pitch_m), std(yaw_m)]

% Remove the non-zero values
filter= roll_b~=0 & pitch_b~=0 & yaw_b~=0 & roll_m~=0 & pitch_m~=0 & yaw_m~=0 ;
time= nonzeros(filter.*time);
roll_b= nonzeros(filter.*roll_b);
pitch_b= nonzeros(filter.*pitch_b);
yaw_b= nonzeros(filter.*yaw_b);
roll_m= nonzeros(filter.*roll_m);
pitch_m= nonzeros(filter.*pitch_m);
yaw_m= nonzeros(filter.*yaw_m);

plot(time,roll_b,time,roll_m, 'LineWidth', 2)
xlabel("Time(s)")
ylabel("Roll w.r.t camera frame (deg)")
title("Roll angle reading from ArUco pose estimation")
legend("ArUco Board","ArUco Marker")

figure
plot(time,pitch_b,time,pitch_m, 'LineWidth', 2)
xlabel("Time(s)")
ylabel("Pitch w.r.t camera frame (deg)")
legend("ArUco Board","ArUco Marker")

figure
plot(time,yaw_b,time,yaw_m, 'LineWidth', 2)
xlabel("Time(s)")
ylabel("Yaw w.r.t camera frame (deg)")
title("Yaw angle reading from ArUco pose estimation")
legend("ArUco Board","ArUco Marker")