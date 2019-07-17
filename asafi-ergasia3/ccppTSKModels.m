% Fuzzy Systems 2019 - Group 3
% Konstantinos Letros 8851
% TSK Model Ser02 - CCPP Dataset

%% Clear

clear;
close all;

%% Preparation

% Options depending on User's TSK number choise
% constant: Singleton , linear: Polynomial
mfNumber = [2 3 2 3];
outputForm = ["constant" "constant" "linear" "linear"];
mkdir Plots % Make a directory to save the plots

%% Choose TSK Model

% Get User Input
% k is the number of TSK-Model, to be chosen by the user
k = str2double(input('Choose TSK-Model (1 - 4) : ', 's'));

% Check that the variable is chosen correctly
while isnan(k) || fix(k) ~= k || k < 1 || k > 4
    k = str2double(input('Please choose an integer between 1 and 4 : ', 's'));
end

%% Initializations

% Count time until completion
tic

% Load the Dataset
load CCPP.dat

% Split the Dataset
training_set = CCPP(1 : round(0.6*size(CCPP,1)), :); % 60% will be used for training
validation_set = CCPP(round(0.6*size(CCPP,1))+1 : round(0.8 * size(CCPP,1)), :); % 20% will be used for validation
check_set = CCPP(round(0.8*size(CCPP,1))+1 : end, :); % 20% will be used for testing

%% Data Normalization

% Find min and max of the training set
training_set_min = min(training_set(:));
training_set_max = max(training_set(:));

% Normalize training set
training_set = (training_set - training_set_min) / (training_set_max - training_set_min); % Scaled to [0 , 1]
training_set = training_set * 2 - 1; % Scaled to [-1 , 1]

% Normalize validation set based on the training set data
validation_set = (validation_set - training_set_min) / (training_set_max - training_set_min); % Scaled to [0 , 1]
validation_set = validation_set * 2 - 1; % Scaled to [-1 , 1]

% Normalize check set based on the training set data
check_set = (check_set - training_set_min) / (training_set_max - training_set_min); % Scaled to [0 , 1]
check_set = check_set * 2 - 1; % Scaled to [-1 , 1]

%% FIS Generation

% Set FIS Options
opt = genfisOptions('GridPartition');
opt.NumMembershipFunctions = mfNumber(k) * ones(4,1); % k MF's for each input
opt.InputMembershipFunctionType = ["gbellmf" "gbellmf" "gbellmf" "gbellmf"]; % Bell-shaped
opt.OutputMembershipFunctionType = outputForm(k); % Constant or Linear

% Generate the FIS
InitialFIS = genfis(training_set(:, 1:end-1), training_set(:, end), opt);

% Plot input Membership Functions
InputMembershipFuncPlotter(InitialFIS);
sgtitle(['TSK Model ', num2str(k) ,' : Membership Functions before training']);
savePlot(join(['TSK_' num2str(k) '_MF_before_Training']));
pause(0.01);

%% Train TSK Model

% Set Training Options
anfis_opt = anfisOptions('InitialFIS', InitialFIS, 'EpochNumber', 400, 'DisplayANFISInformation', 0, 'DisplayErrorValues', 0, 'ValidationData', validation_set);

% Train generated FIS
[trnFIS, trnError, stepSize, chkFIS, chkError] = anfis(training_set, anfis_opt);

% Evaluate the trained FIS
y_hat = evalfis(check_set(:,1:end-1),chkFIS);
y = check_set(:,end);

%% Metrics Calculation

% Calculate MSE - RMSE
MSE = mean((y - y_hat).^2);
RMSE = sqrt(MSE);

% Calculate R^2 coefficient
SSres = sum( (y - y_hat).^2 );
SStot = sum( (y - mean(y)).^2 );
R_sqr = 1 - SSres / SStot;

% Calculate NMSE - NDEI
NMSE = var(y - y_hat) / var(y);
NDEI = sqrt(NMSE);

%% Plot Results

% Plot the Metrics
MetricsPlotter(y,y_hat,trnError,chkError,k);

% Plot trained input Membership Functions
InputMembershipFuncPlotter(chkFIS)
sgtitle(['TSK Model ', num2str(k) ,' : Membership Functions after training']);
savePlot(join(['TSK_' num2str(k) '_MF_after_Training']));

% Display Metrics
fprintf('MSE = %f RMSE = %f R^2 = %f NMSE = %f NDEI = %f\n', MSE, RMSE, R_sqr, NMSE, NDEI)

% Display Elasped Time
toc

%% Function used to Plot the Metrics
function MetricsPlotter(y,y_hat,trnError,chkError,k)

% Plot the Metrics
figure;
plot(1:length(y),y,'*r',1:length(y),y_hat, '.b');
title('Output: Net Hourly Electrical Energy Output');
legend('Reference Outputs','Model Outputs');
savePlot(join(['TSK_' num2str(k) '_Output']));

figure;
plot(y - y_hat);
title('Prediction Errors');
savePlot(join(['TSK_' num2str(k) '_Prediction_Errors']));

figure;
plot(1:length(trnError),trnError,1:length(trnError),chkError);
title('Learning Curve');
legend('Traning Set', 'Check Set');
savePlot(join(['TSK_' num2str(k) '_Learning_Curve']));

end

%% Function used to Plot input Membership Functions of the given FIS
function InputMembershipFuncPlotter(FIS)

% CCPP features
features = ["Temperature" "Ambient Pressure" "Relative Humidity" "Exhaust Vacuum"];

% Subplot with Membership Functions
figure;
for i=1:length(features)
    
    [x,mf] = plotmf(FIS,'input',i);
    subplot(2,2,i);
    plot(x,mf);
    xlabel(join(['Input' num2str(i) ' : ' features(i)]));

end

end

%% Function to automatically save plots in high resolution
function savePlot(name)

% Resize current figure to fullscreen for higher resolution image
set(gcf, 'Position', get(0, 'Screensize'));

% Save current figure with the specified name
saveas(gcf, join(['Plots/',name,'.jpg']));

% Resize current figure back to normal
set(gcf,'position',get(0,'defaultfigureposition'));

end