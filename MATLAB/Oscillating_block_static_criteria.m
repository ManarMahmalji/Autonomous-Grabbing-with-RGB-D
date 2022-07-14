
% ############################################################
% ## BRUFACE: Brussels Faculty of Engineering 
% ############################################################
% ## Master Thesis " Autonmous Grabbing with RGB-D"  
% ## July 2022 Manar Mahmalji      
% ## Version (1)     
% ############################################################
% ##This code is used to visualize what poses of an oscillating suspended block
% ##are considered static 

disp("hello")
close all
clear

%%
filename="measOcsills.xlsx";% osillating suspended block 
data= readmatrix(filename);
time= data(:,2);
x= data(:,3);
y= data(:,4);
z= data(:,5);
roll= data(:,6);
pitch= data(:,7);
yaw= data(:,8);
% Remove the non-zero values
filter= x~=0 & y~=0 & z~=0 & roll~=0 & pitch~=0 & yaw~=0 ;
time= nonzeros(filter.*time);
x= nonzeros(filter.*x);
y= nonzeros(filter.*y);
z= nonzeros(filter.*z);
roll= nonzeros(filter.*roll);
pitch= nonzeros(filter.*pitch);
yaw= nonzeros(filter.*yaw);

% Set rotation and and translation limits
rot_limit= 0.5;% degrees
tran_limit= 10;% mm
wind_size=20; % moving standard deviation window size 

% preallocation 
time_static= zeros(length(time),1);
x_static= zeros(length(time),1);
y_static= zeros(length(time),1);
z_static= zeros(length(time),1);
roll_static= zeros(length(time),1);
pitch_static= zeros(length(time),1);
yaw_static= zeros(length(time),1);

j=1; % counter for static points 
for i=wind_size+1:length(time)
buffer_roll= roll(i-wind_size:i);
buffer_pitch= pitch(i-wind_size:i);
buffer_yaw= pitch(i-wind_size:i);
buffer_x= x(i-wind_size:i);
buffer_y= y(i-wind_size:i);
buffer_z= z(i-wind_size:i);
    if std(buffer_roll)<rot_limit && std(buffer_pitch)<rot_limit && std(buffer_yaw)<rot_limit  && std(buffer_x)<tran_limit && std(buffer_y)<tran_limit && std(buffer_z)<tran_limit
    time_static(j)= time(i);
    x_static(j)= x(i);
    y_static(j)= y(i);
    z_static(j)= z(i);
    roll_static(j)= roll(i);
    pitch_static(j)= pitch(i);
    yaw_static(j)= yaw(i);
    j=j+1;    
    end
end

time_static= nonzeros(time_static);
x_static= nonzeros(x_static);
y_static= nonzeros(y_static);
z_static= nonzeros(z_static);
roll_static= nonzeros(roll_static);
pitch_static= nonzeros(pitch_static);
yaw_static= nonzeros(yaw_static);


plot(time,x,'-', time_static,x_static,'rx')
xlabel("Time(s)")
ylabel("x(mm)")
title("x reading from ArUco pose estimation")
figure
plot(time,y,'-',time_static,y_static,'rx')
xlabel("Time(s)")
ylabel("y (mm)")
title("y reading from ArUco pose estimation")
figure
plot(time,z,'-',time_static,z_static,'rx')
xlabel("Time(s)")
ylabel("z(mm)")
title("z reading from ArUco pose estimation")
figure
plot(time,roll,'-',time_static,roll_static,'rx')
xlabel("Time(s)")
ylabel("roll(deg)")
title("roll reading from ArUco pose estimation")
figure
plot(time,pitch,'-',time_static,pitch_static,'rx')
xlabel("Time(s)")
ylabel("pitch(deg)")
title("pitch reading from ArUco pose estimation")
figure
plot(time,yaw,'-',time_static,yaw_static,'rx')
xlabel("Time(s)")
ylabel("yaw(deg)")
title("yaw reading from ArUco pose estimation")
