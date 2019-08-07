% Fuzzy Systems 2019 - Group 4
% Konstantinos Letros 8851
% TSK Model Ser08 - Avila Dataset

%% Clear

clear;
close all;

%% Preparation

% Make a directory to save the plots
mkdir Plots

%% Initializations

% Count time until completion
tic

% Load the Dataset
load avila.txt

% Cluster Radius parameter for Subtractive Clustering Algorithm
radius = [0.8 0.8 0.3 0.7 0.5]';

% Squash Factor - genfis option 
% (Number of Clusters-Rules respectively: 4 8 12 16 20)
sqFactor=[0.5 0.45 0.475 0.4 0.432]';

% Number of Rules produced by Subtractive Clustering Algorithm
NR = zeros(size(radius,1),1);

% Initialize (cell) arrays for evaluation metrics
error_matrix = cell(1, length(NR));
overall_accuracy = zeros(length(NR),1);
producers_accuracy = cell(1, length(NR));
users_accuracy = cell(1, length(NR));
k_hat = zeros(length(NR),1);

% Count the different output values
tbl = tabulate(avila(:,end));

% Sort the dataset based on the diferrent output values
sortedAvila = sortDataset(avila);

% Uncomment the next two code-lines to improve the training proccess due to 
% the class imbalance issue. Add duplicates of data where needed, so as to 
% have almost equal number of data for every class.
% Example: The First class has 8700 data while the Second Class has only 10 data.
% In order to solve this imbalance we make copies of the data of the Second class 870 times.

% sortedAvila = BallanceDataset(tbl,sortedAvila);
% tbl = tabulate(sortedAvila(:,end));

%% Split the Dataset

% Initialize arrays for the different sets
training_set = [] ; % 60% will be used for training
validation_set = [] ; % 20% will be used for validation
check_set = [] ; % 20% will be used for testing

% Prepare Indices to split the Dataset uniformly
ind = splitPrepare(tbl);

% Fill in the sets
for i = 1 : size(tbl,1)
    training_set = [ training_set ; sortedAvila( ind(i,1) : ind(i,2) - 1 , :) ];
    validation_set = [ validation_set ; sortedAvila( ind(i,2) : ind(i,3) - 1 , :)];
    check_set = [ check_set ; sortedAvila( ind(i,3) : ind(i,1)+tbl(i,2) - 1 , :)];
end

% Proof that sets are correctly split
%(Almost the Same Frequency of Outputs in every set)
fprintf("\nProof that sets are correctly split\n\n");
proofFunc(tbl,training_set,validation_set,check_set);

%% Shuffle each set separately

training_set = shuffleSet(training_set);
validation_set = shuffleSet(validation_set);
check_set = shuffleSet(check_set);

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

% %% Data Normalization (Normalize each feautre separately)
% 
% for i = 1 : size(training_set, 2) % for every feature
%     
%     % Find min and max of the feature
%     training_set_min = min(training_set(:, i));
%     training_set_max = max(training_set(:, i));
%     
%     % Normalize training set
%     training_set(:, i) = (training_set(:, i) - training_set_min) / (training_set_max - training_set_min); % Scaled to [0 , 1]
%     
%     % Normalize validation set based on the training set data
%     validation_set(:, i) = (validation_set(:, i) - training_set_min) / (training_set_max - training_set_min); % Scaled to [0 , 1]
%     
%     % Normalize check set based on the training set data
%     check_set(:, i) = (check_set(:, i) - training_set_min) / (training_set_max - training_set_min); % Scaled to [0 , 1]
%     
% end

%% Train 5 TSK Models

for m = 1 : length(NR)
    
    % Set Subtractive Clustering Options
    genfis_opt = genfisOptions('SubtractiveClustering','ClusterInfluenceRange',radius(m),'SquashFactor',sqFactor(m),'Verbose',0);
    
    % Generate the FIS
    InitialFIS = genfis(training_set(:,1:end-1), training_set(:,end), genfis_opt);
    
    % Number of Rules
    NR(m) = length(InitialFIS.rule);
    
    for i = 1 : length(InitialFIS.output.mf)
        InitialFIS.output.mf(i).type = 'constant';
        InitialFIS.output.mf(i).params = rand()*10;
    end
    
    % Plot Inital Membership Functions
    InputMembershipFuncPlotter(InitialFIS,2*length(NR));
    title(['TSK model ', num2str(m), ': Input MF before training']);
    
    %% Train TSK Model
    
    % Inform the User about the current status
    disp(['Model ', num2str(m), ' of ', num2str(length(NR))]);
    fprintf('Initiating FIS Training.. \n\n');
    
    % Set Training Options
    anfis_opt = anfisOptions('InitialFIS', InitialFIS, 'EpochNumber', 250, 'DisplayANFISInformation', 0, 'DisplayErrorValues', 0, 'ValidationData', validation_set);
    
    % Train generated FIS
    [trnFIS, trnError, stepSize, chkFIS, chkError] = anfis(training_set, anfis_opt);
    
    % Evaluate the trained FIS
    y_hat = evalfis(check_set(:,1:end-1),chkFIS);
    y = check_set(:,end);
    
    % Output must be integer for classifying
    y_hat = round(y_hat);
    
    % Special cases if the output is 1 or 12 and the round leads to 0 or 13
    limitA = tbl(1,1);
    limitB = tbl(end,1);
    y_hat(y_hat < limitA) = limitA;
    y_hat(y_hat > limitB) = limitB;
    
    %% Metrics Calculation
    
    % Total Number of classified values compared to truth values
    N = length(check_set);
    
    % Error Matrix
    error_matrix{m} = confusionmat(y, y_hat);
    
    % Overall Accuracy
    overall_accuracy(m) = sum(diag(error_matrix{m})) / N;
    
    % Producer's Accuracy Initialization
    PA = zeros(limitB , 1);
    
    % User's Accuracy Initialization
    UA = zeros(limitB , 1);
    
    for i = 1 : limitB
        PA(i) = error_matrix{m}(i, i) / sum(error_matrix{m}(:, i));
        UA(i) = error_matrix{m}(i, i) / sum(error_matrix{m}(i, :));
    end
    % Producer's Accuracy
    producers_accuracy{m} = PA;
    % User's Accuracy
    users_accuracy{m} = UA;
    
    % k_hat parameters
    XirXic = zeros( limitB , 1 );
    for i = 1 : limitB
        XirXic(i) = ( sum(error_matrix{m}(i,:)) * sum(error_matrix{m}(:,i)) ) / N^2 ;
    end
    
    % k_hat
    k_hat(m) = (overall_accuracy(m) - sum(XirXic)) / (1 - sum(XirXic));
    
    % Plot Final Membership Functions
    InputMembershipFuncPlotter(chkFIS,2*length(NR));
    title(['TSK model ', num2str(m), ': Input MF after training']);
    
    figure;
    plot(1:length(trnError), trnError, 1:length(trnError), chkError);
    title(['TSK model ', num2str(m), ': Learning Curve']);
    xlabel('Iterations');
    ylabel('Error');
    legend('Training Set', 'Check Set');
    SavePlot(['learning_curve_', num2str(m)]);
    
end

%% Plot Metrics

MetricsPlotter(radius,overall_accuracy,k_hat);

%% Save Metrics Information

save('metrics', 'error_matrix','overall_accuracy','k_hat','producers_accuracy','users_accuracy');

% Display Elasped Time
toc

%% Functions

%% Function to sort the dataset based on output
function sorted = sortDataset(dataset)

[~,idx] = sort(dataset(:,end));
sorted = unique( dataset(idx,:) ,'rows','stable');

end

%% Function to comfront class imbalance issue (Long Delay)
function [Arr,tbl] = BallanceDataset(tbl,sortedArr)

maxCount = max(tbl(:,2));
tempArr = cell(length(tbl),1);
count = 1;
col = sortedArr(1,end);
for i = 1:length(sortedArr)
    if(col ~= sortedArr(i,end))
       count = 1; 
    end
   col = sortedArr(i,end);
   tempArr{col}(count,:) = sortedArr(i,:);
   count = count + 1;
end

Arr = double.empty(0,size(sortedArr,2));
for i = 1:length(tbl)
    for j = 1:round(maxCount/size(tempArr{i},1))
        Arr = cat(1,Arr,tempArr{i});
    end
end

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

%% Function that informs the user for the split dataset's output frequencies
function proofFunc(tbl,training_set,validation_set,check_set)

tbl1 = tabulate(training_set(:,end));
tbl2 = tabulate(validation_set(:,end));
tbl3 = tabulate(check_set(:,end));

% Present proof to the User
frequencyTable = table(tbl(:,1),strcat(num2str(tbl(:,3)),'%'),strcat(num2str(tbl1(:,3)),'%'),...
    strcat(num2str(tbl2(:,3)),'%'),strcat(num2str(tbl3(:,3)),'%'));
frequencyTable.Properties.VariableNames = {'Output_Values' 'Avila_Set' 'Training_Set' 'Validation_Set' 'Check_Set'};
disp(frequencyTable)

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

%% Function used to Plot input Membership Functions of the given FIS
function InputMembershipFuncPlotter(FIS,MFnumber)

% Plot Membership Functions
figure;

for i = 1 : MFnumber
    
    [x,mf] = plotmf(FIS,'input',i);
    plot(x,mf);
    hold on;
    
end

xlabel('Inputs');

end

%% Function used to Plot the Metrics
function MetricsPlotter(radius,overall_accuracy,k_hat)

% Plot the Metrics

% Bar Plot - Overall Accuracy metric
figure;
bar(radius, overall_accuracy);
title('Overall accuracy with regards to number of rules');
xlabel('Radius');
SavePlot('overall_accuracy_metric');


% Bar Plot - k_hat metric
figure;
bar(radius, k_hat);
title('\textbf{ $\hat{k}$ value with regards to number of rules}','interpreter','latex');
xlabel('Radius');
SavePlot('k_hat_metric');

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