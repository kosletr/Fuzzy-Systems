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

% Count the different output values
tbl = tabulate(avila(:,end));

% Sort the dataset based on the diferrent output values
sortedAvila = sortDataset(avila);

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

%% Functions

function sorted = sortDataset(dataset)

[~,idx] = sort(dataset(:,end));
sorted = dataset(idx,:);

end

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

function proofFunc(tbl,training_set,validation_set,check_set)

tbl1 = tabulate(training_set(:,end));
tbl2 = tabulate(validation_set(:,end));
tbl3 = tabulate(check_set(:,end));

% Present proof to the User
frequencyTable = table(tbl(:,1),strcat(num2str(tbl(:,3)),'%'),strcat(num2str(tbl1(:,3)),'%'),strcat(num2str(tbl2(:,3)),'%'),strcat(num2str(tbl3(:,3)),'%'));
frequencyTable.Properties.VariableNames = {'Output_Values' 'Avila_Set' 'Training_Set' 'Validation_Set' 'Check_Set'};
disp(frequencyTable)

end

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