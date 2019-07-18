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
%NF = [5 10 15 20];
NF = [5 10]; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Number of Rules
%NR = [4 8 12 16 20];
NR = [4 8]; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

shuffledData = shuffleSet(isolet);
training_set = shuffleSet(training_set);
validation_set = shuffleSet(validation_set);
check_set = shuffleSet(check_set);

%% Data Normalization (Normalize each feautre separately)

for i = 1 : size(training_set, 2) - 1 % for every feature
    
    % Find min and max of the feature
    training_set_min = min(training_set(:, i));
    training_set_max = max(training_set(:, i));
    
    % Normalize training set
    training_set(:, i) = (training_set(:, i) - training_set_min) / (training_set_max - training_set_min); % Scaled to [0 , 1]
    training_set(:, i) = training_set(:, i) * 2 - 1; % Scaled to [-1 , 1]
    
    % Normalize validation set based on the training set data
    validation_set(:, i) = (validation_set(:, i) - training_set_min) / (training_set_max - training_set_min); % Scaled to [0 , 1]
    validation_set(:, i) = validation_set(:, i) * 2 - 1; % Scaled to [-1 , 1]
    
    % Normalize check set based on the training set data
    check_set(:, i) = (check_set(:, i) - training_set_min) / (training_set_max - training_set_min); % Scaled to [0 , 1]
    check_set(:, i) = check_set(:, i) * 2 - 1; % Scaled to [-1 , 1]

end

%% ReliefF Algorithm
% Evaluate feature's importance using Relieff Algorithm

% k nearest neighbors
k = 100;

fprintf('Initiating ReleifF Algorithm.. \n\n');

%[ranks, ~] = relieff(shuffledData(:, 1:end - 1), shuffledData(:, end), k);
load('ranksMat.mat') %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
            
            for j = 1 : length(InitialFIS.output.mf)
                InitialFIS.output.mf(j).type = 'constant';
                %InitialFIS.output.mf(i).params = rand(); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
            
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
            
            % Output must be integer for classifying
            y_hat = round(y_hat);
            
            % Special cases if the output is out of the classification range
            limitA = tbl(1,1);
            limitB = tbl(end,1);
            y_hat(y_hat < limitA) = limitA;
            y_hat(y_hat > limitB) = limitB;
            
            % Calculate Euclidian-Norm Error
            error(i) = norm(y - y_hat);
            
        end
        
        %% Metrics Calculation
                
        % For every model calculate Mean Error of the 5 folds 
        MeanModelError(f, r) = mean(error);
        
        % Total Number of classified values compared to truth values
        N = length(check_set);
        
        % Error Matrix
        error_matrix{f,r} = confusionmat(y, y_hat);
        
        % Overall Accuracy
        overall_accuracy(f,r) = sum(diag(error_matrix{f,r})) / N;
        
        % Producer's Accuracy Initialization
        PA = zeros(limitB , 1);
        
        % User's Accuracy Initialization
        UA = zeros(limitB , 1);
        
        for i = 1 : limitB
            PA(i) = error_matrix{f,r}(i, i) / sum(error_matrix{f,r}(:, i));
            UA(i) = error_matrix{f,r}(i, i) / sum(error_matrix{f,r}(i, :));
        end
        
        % Producer's Accuracy
        producers_accuracy{f,r} = PA;
        % User's Accuracy
        users_accuracy{f,r} = UA;
        
        % k_hat parameters
        XirXic = zeros( limitB , 1 );
        for i = 1 : limitB
            XirXic(i) = ( sum(error_matrix{f,r}(i,:)) * sum(error_matrix{f,r}(:,i)) ) / N^2 ;
        end
        
        % k_hat
        k_hat(f,r) = (overall_accuracy(f,r) - sum(XirXic)) / (1 - sum(XirXic));
        
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

%% Functions

%% Function to sort the dataset based on output
function sorted = sortDataset(dataset)

[~,idx] = sort(dataset(:,end));
sorted = dataset(idx,:);

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