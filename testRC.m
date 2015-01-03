clc;clear;close all;

dirName = {'testset'};

counter = 1;

for j = 1:length(dirName)
    theDir = dir(dirName{j});
    
    for i = 1:size(theDir,1)
        if ~theDir(i).isdir
            filename = theDir(i).name;
            if (filename(length(filename)-3:length(filename)) == '.jpg')
                groundTruthFilename = filename(1:length(filename)-4);
                groundTruthFilename = strcat(groundTruthFilename,'.png');
                path_to_file = dirName{j};
                path_to_file = strcat(path_to_file,'/');
                path_to_file = strcat(path_to_file,filename);
                path_to_groundTurth = dirName{j};
                path_to_groundTurth = strcat(path_to_groundTurth,'/');
                path_to_groundTurth = strcat(path_to_groundTurth,groundTruthFilename);
                im = imread(path_to_file);
                result = detect_RC(im);
                GT_image = imread(path_to_groundTurth);
                pos_result = find(result == 1);
                pos_GT = find(GT_image == 255);
                resultSize = length(pos_result);
                GTSize = length(pos_GT);
                intersectionSize = length(intersect(pos_GT,pos_result) );
                precision = intersectionSize / resultSize;
                recall = intersectionSize / GTSize;
                precisionList(counter) = precision;
                recallList(counter) = recall;
                counter = counter + 1;
            end            
        end 
    end
end

averagePrecision = sum(precisionList) / length(precisionList);
averageRecall = sum(recallList) / length(recallList);

csvwrite('precision_RC.csv',precisionList);
csvwrite('recall_RC.csv',recallList);
csvwrite('precision_average_RC.csv',averagePrecision);
csvwrite('recall_average_RC.csv',averageRecall);