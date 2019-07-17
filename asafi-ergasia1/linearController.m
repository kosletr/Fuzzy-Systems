% Fuzzy Systems 2019 - Group 1
% Letros Konstantinos 8851
% Linear PI Controller Ser06

%% Clear
clear;
close all;

%% Set value of Zero
z = -0.3 % close to pole -0.1
c = -z;

%% Insert Open Loop System

% A(s) (without gain K)
numP = [1 c];
denP = [1 10.1 1 0];
sys_open_loop = tf(numP, denP);

%% Root Locus Plot
figure;
rlocus(sys_open_loop)

%% Choose Gains
Ka = 1
Kp = 1

K=25*Ka*Kp

%% Insert Closed Loop System
sys_open_loop = K * sys_open_loop
sys_closed_loop = feedback(sys_open_loop, 1, -1)

%% Step Response Plot
figure;
step(sys_closed_loop);

%% Check System Specifications

sysinfo = stepinfo(sys_closed_loop);
overShoot = sysinfo.Overshoot;
riseTime = sysinfo.RiseTime;

if riseTime > 0.6
    fprintf('Rise Time is : %d. Try again.',sysinfo.RiseTime);
end
if overShoot > 8
    fprintf('Overshoot is : %d %. Try again.',sysinfo.Overshoot);
end

%% Calculate K_I
Ki = c*Kp