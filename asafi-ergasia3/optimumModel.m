% Fuzzy Systems 2019 - Group 3
% Konstantinos Letros 8851
% Best Model Ser02 - Superconductivity Dataset

%% Clear

clear;
close all;

%% Preparation

% Make a directory to save the plots
mkdir Plots

% Count time until completion
tic
            
% Load the Dataset
load superconduct.csv
load('optimum_model.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% f=15;
% r=16;
% features_indices = features_indices(1:f);
% features_number = f;
% rules_number = r;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

superconduct = superconduct(:,[features_indices , end]);

%% Shuffle the Dataset

% Initialize an Array with Shuffled Data
shuffledData = zeros(size(superconduct));

% Array of random Positions
rand_pos = randperm(length(superconduct));

% New Array with original data randomly distributed
for i = 1:length(superconduct)
    shuffledData(i, :) = superconduct(rand_pos(i), :);
end

%% Split the Dataset

training_set = shuffledData(1 : round(0.6*size(shuffledData,1)), :); % 60% will be used for training
validation_set = shuffledData(round(0.6*size(shuffledData,1))+1 : round(0.8 * size(shuffledData,1)), :); % 20% will be used for validation
check_set = shuffledData(round(0.8*size(shuffledData,1))+1 : end, :); % 20% will be used for testing

% %% Data Normalization (Normalize each feautre separately)
% 
% for i = 1 : size(training_set, 2) - 1 % for every feature
%     
%     % Find min and max of the feature
%     training_set_min = min(training_set(:, i));
%     training_set_max = max(training_set(:, i));
%     
%     % Normalize training set
%     training_set(:, i) = (training_set(:, i) - training_set_min) / (training_set_max - training_set_min); % Scaled to [0 , 1]
%     training_set(:, i) = training_set(:, i) * 2 - 1; % Scaled to [-1 , 1]
%     
%     % Normalize validation set based on the training set data
%     validation_set(:, i) = (validation_set(:, i) - training_set_min) / (training_set_max - training_set_min); % Scaled to [0 , 1]
%     validation_set(:, i) = validation_set(:, i) * 2 - 1; % Scaled to [-1 , 1]
%     
%     % Normalize check set based on the training set data
%     check_set(:, i) = (check_set(:, i) - training_set_min) / (training_set_max - training_set_min); % Scaled to [0 , 1]
%     check_set(:, i) = check_set(:, i) * 2 - 1; % Scaled to [-1 , 1]
% 
% end


%% FIS Generation

% Set Fuzzy C-Means Clustering Option
genfis_opt = genfisOptions('FCMClustering','NumClusters',rules_number,'Verbose',0);

% Generate the FIS
InitialFIS = genfis(training_set(:, 1:end-1), training_set(:, end), genfis_opt);

%% Plot some input Membership Functions

numberOfPlots = 4;

InputMembershipFuncPlotter(InitialFIS,numberOfPlots);
sgtitle('Best Model - Some Membership Functions before training');
savePlot('Best_Model_MF_before_Training');
pause(0.01);

%% Train TSK Model

% Set Training Options
anfis_opt = anfisOptions('InitialFIS', InitialFIS, 'EpochNumber', 200, 'DisplayANFISInformation', 0, 'DisplayErrorValues', 0, 'ValidationData', validation_set);

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
NMSE = (sum( (y - y_hat).^2 )/length(y)) / var(y);
NDEI = sqrt(NMSE);

%% Plot Results

% Plot the Metrics
MetricsPlotter(y,y_hat,trnError,chkError);

% Plot some trained input Membership Functions
InputMembershipFuncPlotter(chkFIS,numberOfPlots)
sgtitle('Best Model - Some Membership Functions after training');
savePlot(join(['Best_Model_MF_after_Training']));

% Display Metrics
fprintf('MSE = %f RMSE = %f R^2 = %f NMSE = %f NDEI = %f\n', MSE, RMSE, R_sqr, NMSE, NDEI)

% Display Elasped Time
toc

%% Function used to Plot the Metrics
function MetricsPlotter(y,y_hat,trnError,chkError)

% Plot the Metrics
figure;
plot(1:length(y),y,'*r',1:length(y),y_hat, '.b');
title('Output');
legend('Reference Outputs','Model Outputs');
savePlot('Best_Model_Output');

figure;
plot(y - y_hat);
title('Prediction Errors');
savePlot('Best_Model_Prediction_Errors');

figure;
plot(1:length(trnError),trnError,1:length(trnError),chkError);
title('Learning Curve');
legend('Traning Set', 'Check Set');
savePlot('Best_Model_Learning_Curve');

end

%% Function used to Plot input Membership Functions of the given FIS
function InputMembershipFuncPlotter(FIS,numberOfPlots)

% Subplot with Membership Functions
figure;
for i=1:numberOfPlots
    
    [x,mf] = plotmf(FIS,'input',i);
    subplot(2,2,i);
    plot(x,mf);
    xlabel(['Input' num2str(i)]);

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