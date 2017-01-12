% * TITLE ****************************************************************
% *                                                                      *
% *               Pattern Recognition and Neural Networks                *
% *                     Programming Assignment 2015                      *
% *                                                                      *
% *   Author 1: << Chen Haoyu 2474045 >>                               *
% *                                                                      *
% *   NOTE: The file name for this file MUST BE 'classify.m'!            *
% *         Everything should be included in this single file.           *
% *                                                                      *
% ************************************************************************

function classify()
 clc;
    clear;
    %%Loading the data
    load('training_data');
    samples = trainingData;
    classes = class_trainingData;
    
    sizeOfSamples = size(samples,1);

    %%Splitting the samples to training samples and testing samples half by half
    trainingSamples = samples(1:int64(0.5 * sizeOfSamples), :);
    trainingclasses = classes(1:int64(0.5 * sizeOfSamples), :);
    
    testSamples = samples(int64(0.5 * sizeOfSamples):sizeOfSamples, :);
    testClasses = classes(int64(0.5 * sizeOfSamples):sizeOfSamples, :);
    
  
    %%Training the classifer to get parameters
    parameters = trainClassifier(trainingSamples, trainingclasses);
    
    %%Testing the classifer with given parameters
    predictedClasses = evaluateClassifier(testSamples, parameters);
    
    %%Examining the performance of the classifier
    accuracy = AccuracyCalculator(predictedClasses,testClasses);
    fprintf('%s',getNickName());
    fprintf('Accuracy: %.4f\n', accuracy);
end

function nick = getNickName()
    nick = 'AlphaOulu';   % CHANGE THIS!
end

function parameters = trainClassifier(samples, classes)
%     feature selection
    [A, D] = size(samples);
    
%     randomly divide data
    randomClass = randi([0, 1], A, 1, 'double');
    trainInd = find(randomClass == 0);
    testInd = find(randomClass == 1);
    trainsam = samples(trainInd, :);
    trainclass = classes(trainInd);
    testsam = samples(testInd, :);
    testclass = classes(testInd);
    j = 1;
    
%     feature selection
%     loop through each feature to see which features are useful
    for i = 1:D
        seltr = trainsam(:, i);
        selte = testsam(:, i);
        estiClass(:,i) = KNN(seltr, trainclass, selte, 7); % use KNN to classify
        vali = testclass - estiClass(:,i);
        benc(i) = size(find(vali == 0)) / size(testclass);

%         save wanted feature indices into sel
        if benc(i) > 0.4;
            feature(j) = i;
            j = j + 1;
            
        end
    end
    
%     pass parameters
    parameters = struct('trainsamples',samples,'trainclasses',classes,'feature',feature);
end

function results = evaluateClassifier(samples, parameters)
%     receive parameters
    trainsamples = parameters.trainsamples;
    trainclasses = parameters.trainclasses;
    feature = parameters.feature;
    
%     select features
    trainsam = trainsamples(:, feature);
    testsam = samples(:, feature);
    
%     normalize
    m = mean(trainsam);
    s = std(trainsam);
    for i = 1:size(trainsam, 2)
        trainsam(:, i) = (trainsam(:, i) - m(i))./s(i);
    end
    for j = 1:size(testsam, 2)
        testsam(:, j) = (testsam(:, j) - m(j))./s(j);
    end
    
%     use KNN to classify
    results = KNN(trainsam, trainclasses, testsam, 7);
end

function class = KNN(samples, classes, test, k)
%     find all class labels
    classLabel = unique(classes);
%     count classes
    classCount = length(classLabel);
    A = size(samples,1);
    dist = zeros(size(samples,1),1);
    
    for j = 1:size(test,1)
        vote = zeros(classCount,1);
        
%         find distances from current point to other points
        for i = 1:A
            dist(i) = norm(samples(i,:) - test(j,:));
        end
        [d,distRank] = sort(dist);
        
%        loop through the nearest k points
        for i = 1:k
             % find the class of the point
            classInd = find(classLabel == classes(distRank(i)));
            % vote that class by distance
            vote(classInd) = vote(classInd) + 1/d(i);
        end
        % find the max voted class
        [m,classInd] = max(vote);
        class(j) = classLabel(classInd);
    end
    
    class = class.';
    
end


