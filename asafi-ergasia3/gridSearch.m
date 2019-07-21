% Fuzzy Systems 2019 - Group 3
% Konstantinos Letros 8851
% Grid Search Ser02 - Superconductivity Dataset

%% Clear

clear;
close all;

%% Preparation

% Make a directory to save the plots
mkdir Plots

% Count time until completion
tic

fprintf('Preparing Dataset.. \n\n');
            
% Load the Dataset
load superconduct.csv

%% Shuffle the Data of the Dataset

% Initialize an Array with Shuffled Data
shuffledData = zeros(size(superconduct));

% Array of random Positions
rand_pos = randperm(length(superconduct)); 

% New Array with original data randomly distributed
for i = 1:length(superconduct)
    shuffledData(i, :) = superconduct(rand_pos(i), :);
end

%% Initializations

% Number of Features
%NF = [5 10 15 20];
NF = [5 10];

% Number of Rules
%NR = [4 8 12 16 20];
NR = [4 8];

MeanModelError = zeros(length(NF), length(NR));
counter = 1;

%% Split the Dataset

training_set = shuffledData(1 : round(0.6*size(shuffledData,1)), :); % 60% will be used for training
validation_set = shuffledData(round(0.6*size(shuffledData,1))+1 : round(0.8 * size(shuffledData,1)), :); % 20% will be used for validation
check_set = shuffledData(round(0.8*size(shuffledData,1))+1 : end, :); % 20% will be used for testing

% %% Data Normalization
% 
% % Find min and max of the training set
% training_set_min = min(training_set(:));
% training_set_max = max(training_set(:));
% 
% % Normalize training set
% training_set = (training_set - training_set_min) / (training_set_max - training_set_min); % Scaled to [0 , 1]
% 
% % Normalize validation set based on the training set data
% validation_set = (validation_set - training_set_min) / (training_set_max - training_set_min); % Scaled to [0 , 1]
% 
% % Normalize check set based on the training set data
% check_set = (check_set - training_set_min) / (training_set_max - training_set_min); % Scaled to [0 , 1]

%% Data Normalization (Normalize each feautre separately)

for i = 1 : size(training_set, 2) % for every feature
    
    % Find min and max of the feature
    training_set_min = min(training_set(:, i));
    training_set_max = max(training_set(:, i));
    
    % Normalize training set
    training_set(:, i) = (training_set(:, i) - training_set_min) / (training_set_max - training_set_min); % Scaled to [0 , 1]
    
    % Normalize validation set based on the training set data
    validation_set(:, i) = (validation_set(:, i) - training_set_min) / (training_set_max - training_set_min); % Scaled to [0 , 1]
    
    % Normalize check set based on the training set data
    check_set(:, i) = (check_set(:, i) - training_set_min) / (training_set_max - training_set_min); % Scaled to [0 , 1]
    
end

%% ReliefF Algorithm
% Evaluate feature's importance using Relieff Algorithm

% k nearest neighbors
k = 100;

fprintf('Initiating ReleifF Algorithm.. \n\n');

% [ranks, ~] = relieff(shuffledData(:, 1:end - 1), shuffledData(:, end), k);
load('ranksMat.mat')

%% Grid Search Algorithm

fprintf('Initiating Grid Search Algorithm.. \n\n');

for f = 1 : length(NF)
    for r = 1 : length(NR)
        
        %% 5-Fold Cross Validation 
        
        % Create 5 Folds
        c = cvpartition(training_set(:, end), 'KFold', 5);
        error = zeros(c.NumTestSets, 1);
        
        % For every Fold
        for i = 1 : c.NumTestSets
            
            train_id = c.training(i);
            test_id = c.test(i);
            
            % 80% of Data for Training (default)
            training_set_x = training_set(train_id, ranks(1:NF(f)));
            training_set_y = training_set(train_id, end);
            
            % 20% of Data for Validation (default)
            validation_data_x = training_set(test_id, ranks(1:NF(f)));
            validation_data_y = training_set(test_id, end);
           
            %% FIS Generation
            
            % Set Fuzzy C-Means Clustering Options
            genfis_opt = genfisOptions('FCMClustering','NumClusters',NR(r),'Verbose',0);
            
            % Generate the FIS
            InitialFIS = genfis(training_set_x, training_set_y, genfis_opt);
            
            % Set the validation data option to avoid overfitting
            anfis_opt = anfisOptions('InitialFIS', InitialFIS, 'EpochNumber', 150, 'DisplayANFISInformation', 0, 'DisplayErrorValues', 0, 'DisplayStepSize', 0, 'DisplayFinalResults', 0, 'ValidationData', [validation_data_x validation_data_y]);
            
            % Inform the User about the current status
            disp(['Model ', num2str(counter), ' of ', num2str(length(NF)*length(NR))]);
            disp(['Number of Features : ',num2str(NF(f))]);
            disp(['Number of Rules : ',num2str(NR(r))]) ;
            disp(['Fold Number: ',num2str(i)]); 
            fprintf('Initiating FIS Training.. \n\n');
            
            % Train the FIS
            [trnFIS, trainError, ~, InitialFIS, ~] = anfis([training_set_x training_set_y], anfis_opt);
         
            % Evaluate the FIS
            y_hat = evalfis(validation_set(:, ranks(1:NF(f))), InitialFIS);
            y = validation_set(:, end);
         
            % Calculate Euclidian-Norm Error
            error(i) = norm(y - y_hat);
            
            
        end
        
        % For every model calculate Mean Error of the 5 folds 
        MeanModelError(f, r) = mean(error);
        
        % Count Total Models
        counter = counter + 1;
    end
end

%% Model Errors
fprintf('The Mean Error for every Model respectively: \n');
disp(MeanModelError)

%% Plot the Errors

% 2D Plots of All Mean Errors
figure;
sgtitle('Mean Error for different number of Features and Rules');

for i=1:length(NF)
    
    subplot(2,2,i);
    bar(MeanModelError(i,:))
    xlabel('Number of Rules');
    ylabel('Mean Square Error');
    xticklabels(string(NR));
    legend([num2str(NF(i)),' features'])
    
end

SavePlot('Subplots_Mean_Errors');


% 3D Plot of All Model Errors
figure;
bar3(MeanModelError);
ylabel('Number of Features');
yticklabels(string(NF));
xlabel('Number of Rules');
xticklabels(string(NR));
zlabel('Mean square error');
title('3D Plot of All Model Errors for different Features and Rules');

SavePlot('3Dplot_Mean_Error');

%% Best Model Decision

% The one with the minimum mean error
[~,MinIndices]=min(MeanModelError);
[~,ModelNum]=min(reshape(MeanModelError',[],1));

features_number = NF(MinIndices(1));
rules_number = NR(MinIndices(2));

% Inform the user about the best model
disp(['The Model with the minimum Error is Model ',num2str(ModelNum)]);
disp(['Number of Features : ',num2str(features_number)]);
disp(['Number of Rules : ',num2str(rules_number)]) ;

% Feature columns used (ReleifF)
features_indices = sort(ranks(1:features_number));

% Save Best Model Specs
save('best_model.mat','features_number','rules_number','features_indices')

% Display Elasped Time
toc

%% Function to automatically save plots in high resolution
function SavePlot(name)

% Resize current figure to fullscreen for higher resolution image
set(gcf, 'Position', get(0, 'Screensize'));

% Save current figure with the specified name
saveas(gcf, join(['Plots/',name,'.jpg']));

% Resize current figure back to normal
set(gcf,'position',get(0,'defaultfigureposition'));

end