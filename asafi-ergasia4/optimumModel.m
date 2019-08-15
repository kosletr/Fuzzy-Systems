% Fuzzy Systems 2019 - Group 4
% Konstantinos Letros 8851
% Grid Search Ser08 - Isolet Dataset

%% Clear

clear;
close all;

%% Preparation

% Make a directory to save the plots
mkdir Plots

% Count time until completion
tic
            
% Load the Dataset
load ../../isolet.dat %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('optimum_model.mat')
isolet = isolet(:,[features_indices ,end]);

% Sort the dataset based on the diferrent output values
sortedIsolet = sortDataset(isolet);

% Count the different output values
tbl = tabulate(sortedIsolet(:,end));

%% Split the Dataset

% Initialize arrays for the different sets
training_set = [] ; % 60% will be used for training
validation_set = [] ; % 20% will be used for validation
check_set = [] ; % 20% will be used for testing

% Prepare Indices to split the Dataset uniformly
ind = splitPrepare(tbl);

% Fill in the sets
for i = 1 : size(tbl,1)
    training_set = [ training_set ; sortedIsolet( ind(i,1) : ind(i,2) - 1 , :) ];
    validation_set = [ validation_set ; sortedIsolet( ind(i,2) : ind(i,3) - 1 , :)];
    check_set = [ check_set ; sortedIsolet( ind(i,3) : ind(i,1)+tbl(i,2) - 1 , :)];
end

% Proof that sets are correctly split
%(Almost the Same Frequency of Outputs in every set)
fprintf("\nProof that sets are correctly split\n\n");
proofFunc(tbl,training_set,validation_set,check_set);

%% Shuffle each set separately

training_set = shuffleSet(training_set);
validation_set = shuffleSet(validation_set);
check_set = shuffleSet(check_set);

% %% Data Normalization (Normalize each feautre separately)
% 
% for i = 1 : size(training_set, 2) - 1 % for every feature`
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

for i = 1 : length(InitialFIS.Output.MF)
    InitialFIS.Output.MF(i).Type = 'constant';
end

%% Plot some input Membership Functions

numberOfPlots = 4;

InputMembershipFuncPlotter(InitialFIS,numberOfPlots);
sgtitle('Membership Functions before training');
SavePlot('Best_Model_MF_before_Training');
pause(0.01);

%% Train TSK Model

% Set Training Options
anfis_opt = anfisOptions('InitialFIS', InitialFIS, 'EpochNumber', 500, 'DisplayANFISInformation', 0, 'DisplayErrorValues', 0, 'ValidationData', validation_set);

% Train generated FIS
[trnFIS, trnError, stepSize, chkFIS, chkError] = anfis(training_set, anfis_opt);

% Evaluate the trained FIS
y_hat = evalfis(check_set(:,1:end-1),chkFIS);
y = check_set(:,end);

% Output must be integer for classifying
y_hat = round(y_hat);

% Special cases if the output is out of the classification range
limitA = tbl(1,1);
limitB = tbl(end,1);
y_hat(y_hat < limitA) = limitA;
y_hat(y_hat > limitB) = limitB;

%% Metrics Calculation

% Total Number of classified values compared to truth values
N = length(check_set);

% Error Matrix
error_matrix = confusionmat(y, y_hat);

% Overall Accuracy
overall_accuracy = sum(diag(error_matrix)) / N;

% Producer's Accuracy Initialization
PA = zeros(limitB , 1);

% User's Accuracy Initialization
UA = zeros(limitB , 1);

for i = 1 : limitB
    PA(i) = error_matrix(i, i) / sum(error_matrix(:, i));
    UA(i) = error_matrix(i, i) / sum(error_matrix(i, :));
end

% Producer's Accuracy
producers_accuracy = PA;
% User's Accuracy
users_accuracy = UA;

% k_hat parameters
XirXic = zeros( limitB , 1 );
for i = 1 : limitB
    XirXic(i) = ( sum(error_matrix(i,:)) * sum(error_matrix(:,i)) ) / N^2 ;
end

% k_hat
k_hat = (overall_accuracy - sum(XirXic)) / (1 - sum(XirXic));


%% Plot Results

% Plot the Metrics
MetricsPlotter(y,y_hat,trnError,chkError);

% Plot some trained input Membership Functions
InputMembershipFuncPlotter(chkFIS,numberOfPlots)
sgtitle('Best Model - Some Membership Functions after training');
SavePlot(join(['Best_Model_MF_after_Training']));

% Display Metrics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Display Elasped Time
toc

%% Functions
%% Function used to Plot the Metrics
function MetricsPlotter(y,y_hat,trnError,chkError)

% Plot the Metrics
figure;
plot(1:length(y),y,'*r',1:length(y),y_hat, '.b');
title('Output');
legend('Reference Outputs','Model Outputs');
SavePlot('Best_Model_Output');

figure;
plot(y - y_hat);
title('Prediction Errors');
SavePlot('Best_Model_Prediction_Errors');

figure;
plot(1:length(trnError),trnError,1:length(trnError),chkError);
title('Learning Curve');
legend('Traning Set', 'Check Set');
SavePlot('Best_Model_Learning_Curve');

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

%% Function to sort the dataset based on output
function sorted = sortDataset(dataset)

[~,idx] = sort(dataset(:,end));
sorted = unique( dataset(idx,:) ,'rows','stable');

end

%% Function to get the indices needed for splitting the dataset appropriately
function ind = splitPrepare(tbl)

% Prepare indices of sets
ind = zeros(length(tbl),3);
ind(1,1) = 1;
ind(1,2) = ind(1,1) + round(0.6*tbl(1,2));
ind(1,3) = ind(1,2) + round(0.2*tbl(1,2));

for i=2:length(tbl)
    ind(i,1) = ind(i-1,1) + tbl(i-1,2);
    ind(i,2) = ind(i,1) + round(0.6*tbl(i,2));
    ind(i,3) = ind(i,2) + round(0.2*tbl(i,2));
end
end


%% Function for shuffling a Dataset
function shuffledData = shuffleSet(set)

% Initialize an Array with Shuffled Data
shuffledData = zeros(size(set));

% Array of random Positions
rand_pos = randperm(length(set));

% New Array with original data randomly distributed
for i = 1:length(set)
    shuffledData(i, :) = set(rand_pos(i), :);
end

end

%% Function that informs the user for the split dataset's output frequencies
function proofFunc(tbl,training_set,validation_set,check_set)

tbl1 = tabulate(training_set(:,end));
tbl2 = tabulate(validation_set(:,end));
tbl3 = tabulate(check_set(:,end));

% Present proof to the User
frequencyTable = table(tbl(:,1),strcat(num2str(tbl(:,3)),'%'),strcat(num2str(tbl1(:,3)),'%'),...
    strcat(num2str(tbl2(:,3)),'%'),strcat(num2str(tbl3(:,3)),'%'));
frequencyTable.Properties.VariableNames = {'Output_Values' 'Isolet_Set' 'Training_Set' 'Validation_Set' 'Check_Set'};
disp(frequencyTable)

end

%% Function to automatically save plots in high resolution
function SavePlot(name)

% Resize current figure to fullscreen for higher resolution image
set(gcf, 'Position', get(0, 'Screensize'));

% Save current figure with the specified name
saveas(gcf, join(['Plots/',name,'.jpg']));

% Resize current figure back to normal
set(gcf,'position',get(0,'defaultfigureposition'));

end