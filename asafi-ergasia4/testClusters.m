% Clear
clear

% Load the Dataset
load Fuzzy-Systems\asafi-ergasia4\avila.txt

% Cluster Radius parameter for Subtractive Clustering Algorithm
radius = [0.8 0.8 0.3 0.7 0.5]';

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

%% Shuffle each set separately

training_set = shuffleSet(training_set);
validation_set = shuffleSet(validation_set);
check_set = shuffleSet(check_set);


C = cell(length(sqFactor),length(radius));
k = 1;

for i=1:length(sqFactor)
    j=i;
%    for j=1:length(radius)
        opt = [sqFactor(i),0.5,0.15,0];
        C{k}=subclust(training_set,radius(j),'Options',opt);
        disp(['radius = ',num2str(radius(j)),' sqFactor = ',num2str(sqFactor(i)),' ClustersNum = ',num2str(size(C{k},1))])
        k = k + 1;
%    end
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
