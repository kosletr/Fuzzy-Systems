% Fuzzy Systems 2019 - Group 2
% Konstantinos Letros 8851
% Car Control Ser05

%% Clear

clear;
close all;

%% Initializations

% Initialization of Car's Position and Angle
x0 = 4; % in meters
y0 = 0.4; % in meters
theta0 = [0,-45,-90]; % in degrees

% Car Velocity (constant magnitude)
u = 0.05; % in meters per second

% Set Obstacle Positions
obstacles_x = [5; 5; 6; 6; 7; 7; 10];
obstacles_y = [0; 1; 1; 2; 2; 3; 3];

% Desired Position
xd = 10;  % in meters
yd = 3.2; % in meters

%% Import Fuzzy Logic Designer files

initFzController = readfis('InitialCarController.fis'); % Initial Fuzzy Controller
imprFzController = readfis('ImprovedCarController.fis'); % Improved Fuzzy Controller

%% Plot Membership Functions and Car Trajectories

% Initial Results
MembershipFuncPlotter(initFzController);
carControl(x0,y0,theta0,u,obstacles_x,obstacles_y,xd,yd,initFzController);

% Improved Results
MembershipFuncPlotter(imprFzController);
carControl(x0,y0,theta0,u,obstacles_x,obstacles_y,xd,yd,imprFzController);

%% Functions

%% Membership Function Plotter
function MembershipFuncPlotter(fuzzyController)

% Inputs of the Fuzzy Controller

% d_v
figure;
plotmf(fuzzyController, 'input',1);
title('Membership Function of d_v');

% d_h
figure;
plotmf(fuzzyController, 'input',2);
title('Membership Function of d_h');

% theta
figure;
plotmf(fuzzyController, 'input',3);
title('Membership Function of \theta');

% Output of the Fuzzy Controller

% Delta theta
figure;
plotmf(fuzzyController, 'output',1);
title('Membership Function of \Delta\theta');

end

%% Calculate Car Trajectories depending on the given Specifications
function carControl(x0, y0, theta0, u,obstacles_x,obstacles_y, xd, yd, fuzzyController)

for i=1:length(theta0)
    
    % Initialize car's position and angle
    x = x0;
    y = y0;
    theta = theta0(i);
    
    % Boolean variable to check that the car is inside the map
    outOfBounds = false;
    
    % Initialize vectors to save the trajectories
    x_pos = x0;
    y_pos = y0;
    
    while (outOfBounds == false)
        
        % Calculate the distance from obstacles
        [dv,dh] = distance_sensors(x, y,obstacles_x,obstacles_y);
        
        % Fuzzy Controller Output
        Delta_theta = evalfis(fuzzyController,[dv dh theta]);
        
        % New angle and position of the car
        theta = theta + Delta_theta;
        x = x + u * cosd(theta);
        y = y + u * sind(theta);
        
        % Keep record of the car's trajectory
        x_pos(end+1) = x;
        y_pos(end+1) = y;
        
        % If the car is out of the map, end the loop
        if x < 0 || x >= obstacles_x(end) || y < 0 || y >= obstacles_y(end)+1
            outOfBounds = true;
        end
        
    end
    
    % Calculate the Euclidian norm Error
    eucl_dist_error = norm( [xd - x_pos(end) , yd - y_pos(end)] );
    
    % Plot the Trajectory of the car and the obstacles
    figure;
    title(['Starting angle \theta_0 = ', num2str(theta0(i)), '^{\circ}  |  Euclidian Distance Error: ' , num2str(eucl_dist_error)]);
    line(x_pos, y_pos, 'Color','blue');
    line(obstacles_x, obstacles_y, 'Color','red');
    
    % Mark the initial and desired points on the plot
    hold on;
    plot(xd, yd, '*');
    hold on;
    plot(x0, y0, '*');
end

end

%% Function to simulate car's distance sensors

function [dv,dh] = distance_sensors(x, y, obstacles_x, obstacles_y)

% Returns the vertical and horizontal distance from obstacles
% There are four different cases, depending on the x position

dv = y - obstacles_y(end);
dh = obstacles_x(end) + 1 - x;

for i = length(obstacles_x):-1:1
    if x < obstacles_x(i) && y < obstacles_y(i)
        dh = obstacles_x(i) - x;
    elseif x < obstacles_x(i) && y > obstacles_y(i)
        dv = y - obstacles_y(i);
    end
end

% Distance values must be between zero and one

dv = min(dv,1);
dv = max(dv,0);

dh = min(dh,1);
dh = max(dh,0);

end