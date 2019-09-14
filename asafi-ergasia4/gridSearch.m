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

fprintf('Preparing Dataset.. \n\n');

% Load the Dataset
load ../../isolet.dat %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initializations

% Number of Features
NF = [3 9 16 21];
% NF = [3 9]; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Number of Rules
NR = [4 8 12 16 20];
% NR = [4 8]; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

MeanModelError = zeros(length(NF), length(NR));
counter = 1;

% Initialize (cell) arrays for evaluation metrics
error_matrix = cell(length(NF), length(NR));
overall_accuracy = zeros(length(NF),length(NR));
producers_accuracy = cell(length(NF), length(NR));
users_accuracy = cell(length(NF), length(NR));
k_hat = zeros(length(NF),length(NR));

% Count the different output values
tbl = tabulate(isolet(:,end));

% Sort the dataset based on the diferrent output values
sortedIsolet = sortDataset(isolet);

% Uncomment the next two code-lines to improve the training proccess due to
% the class imbalance issue. Add duplicates of data where needed, so as to
% have almost equal number of data for every class.
% Example: The First class has 8700 data while the Second Class has only 10 data.
% In order to solve this imbalance we make copies of the data of the Second class 870 times.

% sortedIsolet = BallanceDataset(tbl,sortedIsolet);
% tbl = tabulate(sortedIsolet(:,end));

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

%% ReliefF Algorithm
% Evaluate feature's importance using Relieff Algorithm

% k nearest neighbors
k = 50;

fprintf('Initiating ReleifF Algorithm.. \n\n');

% [ranks, ~] = relieff(isolet(:, 1:end - 1), isolet(:, end), k, 'method', 'classification');
load('ranksMat.mat') %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% save ranksMat.mat ranks

%% Grid Search Algorithm

fprintf('Initiating Grid Search Algorithm.. \n\n');

%% 5-Fold Cross Validation

for f = 1 : length(NF)
    for r = 1 : length(NR)
        
        
        fold_train = cell(1,5);
        fold_test = cell(1,5);
        test = cell(1,5);
        
        % Create 5 Folds - Data in every fold must be fairly distributed
        for i = 1 : size(tbl,1)
            singleClass = training_set(training_set(:,end)==tbl(i,1),:);
            c = cvpartition(singleClass(:, end), 'KFold', 5);
            for w = 1 : c.NumTestSets
                fold_train{1,w} = [fold_train{1,w};c.training(w)];
                fold_test{1,w} = [fold_test{1,w};c.test(w)];
            end
        end
        
        error = zeros(c.NumTestSets, 1);
        
        % For every Fold
        for i = 1 : c.NumTestSets
            
            train_id = logical(fold_train{1,i});
            test_id = logical(fold_test{1,i});
            
            % 80% of Data for Training (default)
            training_set_x = training_set(train_id, ranks(1:NF(f)));
            training_set_y = training_set(train_id, end);
            
            % 20% of Data for Validation (default)
            validation_data_x = training_set(test_id, ranks(1:NF(f)));
            validation_data_y = training_set(test_id, end);
            
            % Shuffle the data inside each fold before training
            training_set_x = suffleSet(training_set_x);
            training_set_y = suffleSet(training_set_y);
            validation_data_x = suffleSet(validation_data_x);
            validation_data_y = suffleSet(validation_data_y);
            
            %% FIS Generation
            
            % Set Fuzzy C-Means Clustering Options
            genfis_opt = genfisOptions('FCMClustering','NumClusters',NR(r),'Verbose',0,'FISType','sugeno');
            
            % Generate the FIS
            InitialFIS = genfis(training_set_x, training_set_y, genfis_opt);
            
            for j = 1 : length(InitialFIS.Output.MF)
                InitialFIS.Output.MF(j).Type = 'constant';
                InitialFIS.Output.MF(j).Params = (tbl(1,1)+tbl(end,1))/2;
            end
            
            % Set the validation data option to avoid overfitting
            anfis_opt = anfisOptions('InitialFIS', InitialFIS, 'EpochNumber', 150, 'DisplayANFISInformation', 0, 'DisplayErrorValues', 0, 'DisplayStepSize', 0, 'DisplayFinalResults', 0, 'ValidationData', [validation_data_x validation_data_y]);
            
            % Inform the User about the current status
            disp(['Model ', num2str(counter), ' of ', num2str(length(NF)*length(NR))]);
            disp(['Number of Features : ',num2str(NF(f))]);
            disp(['Number of Rules : ',num2str(NR(r))]) ;
            disp(['Fold Number: ',num2str(i)']);
            fprintf('Initiating FIS Training.. \n\n');
            
            % Train the FIS
            [trnFIS, trainError, ~, InitialFIS, chkError] = anfis([training_set_x training_set_y], anfis_opt);
            
            % Evaluate the FIS
            y_hat = evalfis(validation_set(:, ranks(1:NF(f))), InitialFIS);
            y = validation_set(:, end);
            
            % Output must be integer for classifying
            y_hat = round(y_hat);
            
            % Special cases if the output is out of the classification range
            limitA = tbl(1,1);
            limitB = tbl(end,1);
            y_hat(y_hat < limitA) = limitA;
            y_hat(y_hat > limitB) = limitB;
            
            % Calculate Euclidian-Norm Error
            error(i) = (norm(y-y_hat))^2/length(y);
        end
        
        %% Metrics Calculation
        
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

%% Optimum Model Decision

% The one with the minimum mean error
minMatrix = min(MeanModelError(:));
[min_row,min_col] = find(MeanModelError==minMatrix);

[~,ModelNum]=min(reshape(MeanModelError',[],1));

features_number = NF(min_row);
rules_number = NR(min_col);

% Inform the user about the Optimum model
disp(['The Model with the minimum Error is Model ',num2str(ModelNum)]);
disp(['Number of Features : ',num2str(features_number)]);
disp(['Number of Rules : ',num2str(rules_number)]) ;

% Feature columns used (ReleifF)
features_indices = sort(ranks(1:features_number));

% Save Optimum Model Specs
save('optimum_model.mat','features_number','rules_number','features_indices')

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