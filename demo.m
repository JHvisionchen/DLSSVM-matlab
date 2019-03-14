% ********************************************************************
% you need configure the opencv for run this program.                %
% The program is successfully run under opencv 2.4.8                 %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% path to the folder with OTB sequences
base_path = '/media/cjh/datasets/tracking/OTB100/';
% choose name of the OTB sequence
sequence_name = choose_video(base_path);
video_path=[base_path sequence_name '/'];
filename = [video_path 'groundtruth_rect.txt'];
f = fopen(filename);
ground_truth = textscan(f, '%f,%f,%f,%f', 'ReturnOnError',false);  
ground_truth = cat(2, ground_truth{:});
fclose(f);
image_path=[base_path sequence_name '/img'];
gt=ground_truth(1,:);


res = tracker(image_path,'jpg',true,gt);