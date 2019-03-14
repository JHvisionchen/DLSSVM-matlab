function result = tracker(input, ext, show_img, init_rect, start_frame, end_frame, s_frames)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function: implement the dlssvm tracker                             %
% parameters:                                                        %
%      input: path of image sequences                                %
%      ext:extension name of file, for example, '.jpg'               %
%      show_img:                                                     %
%      init_rect: initial position of the target                     %
%      start_frame:                                                  %
%      end_frame:                                                    %
%      s_frames: the number of frames                                %
%                                                                    %
% ********************************************************************
%     you need configure the opencv for run this program.            %
%     The program is successfully run under opencv 2.4.8             %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% show_img=1;
addpath(genpath('.'));
D = dir(fullfile(input,['*.', ext]));
file_list={D.name};

if nargin < 4
    init_rect = -ones(1,4);
end
if nargin < 5
    start_frame = 1;
end
if nargin < 6
    end_frame = numel(file_list);
end

global sampler
global tracker
global config
global finish

config.display = true;
sampler = createSampler();
finish = 0;
timer = 0;
result.res = nan(end_frame-start_frame+1,4);
result.len = end_frame-start_frame+1;
result.startFrame = start_frame;
result.type = 'rect';

output = zeros(1,4);

patterns = cell(1, 1);
params = makeParams();
k = 1;

for frame_id = start_frame:end_frame
    if finish == 1
        break;
    end

    if ~config.display
        clc
        display(input);
        display(['frame: ',num2str(frame_id),'/',num2str(end_frame)]);
    end
    
    if nargin == 7
        I_orig=imread(s_frames{frame_id-start_frame+1});
    else
        I_orig=imread(fullfile(input,file_list{frame_id}));
    end
    
    if frame_id==start_frame  % for the first fram
        init_rect = round(init_rect);
        % set config parameters according to the region size of the first frame
        config = makeConfig(I_orig,init_rect,true,false,true,show_img);
        tracker.output = init_rect*config.image_scale;
        tracker.output(1:2) = tracker.output(1:2) + config.padding;
        tracker.output_exp = tracker.output;        
        output = tracker.output;
    end
        
    [I_scale]= getFrame2Compute(I_orig);
    
    sampler.roi = rsz_rt(output,size(I_scale),config.search_roi,true);
    I_crop = I_scale(round(sampler.roi(2):sampler.roi(4)),round(sampler.roi(1):sampler.roi(3)),:);
    
    % we employ the the feature used by MEEM (Jianming, Zhang et al, ECCV2014)
    % to represent the object
    [BC, F] = getFeatureRep(I_crop,config.hist_nbin);
    
    tic
    
    if frame_id==start_frame
        initSampler(tracker.output,BC,F,config.use_color);
        patterns{1}.X = sampler.patterns_dt;
        % patterns{1}.X is denoted by phi_i(y)=phi(x_i,y_i)-phi(x_i,y)
        % in next line
        patterns{1}.X = repmat(patterns{1}.X(1, :), size(patterns{1}.X, 1),1) - patterns{1}.X;
        patterns{1}.Y = sampler.state_dt;   % structured output
        patterns{1}.lossY = sampler.costs;  % loss function: L(y_i,y)
        patterns{1}.supportVectorNum=[];    % save structured output index whose alpha is not zero
        patterns{1}.supportVectorAlpha=[];  % save dual variable
        patterns{1}.supportVectorWeight=[]; % save weight related to dual variable
        
        w0 = zeros(1, size(patterns{1}.X, 2));  % initilize the classifer w0
        
        % training classifier w0 by the proposed dlssvm optimization method
        [w0, patterns]=dlssvmOptimization(patterns,params, w0);
        
        if config.display
            %figure(1);
            %imshow(I_orig);
            res = tracker.output;
            res(1:2) = res(1:2) - config.padding;
            res = res/config.image_scale;
            %rectangle('position',res,'LineWidth',2,'EdgeColor','b')
            
            figure('Number','off', 'Name',['Tracker - ' 'MEEM']);
            im_handle = imshow(uint8(I_orig), 'Border','tight', 'InitialMag', 100 + 100 * (length(I_orig) < 500));
            rect_handle = rectangle('Position',res, 'EdgeColor','r');
            rect_handle2 = rectangle('Position',res, 'EdgeColor','b');
            text_handle = text(10, 10, int2str(frame_id));
            set(text_handle, 'color', [0 1 1]);
        end
    else
        if config.display
            %figure(1)
            %imshow(I_orig);
            roi_reg = sampler.roi; roi_reg(3:4) = sampler.roi(3:4)-sampler.roi(1:2)+1;
            roi_reg(1:2) = roi_reg(1:2) - config.padding;
            %rectangle('position',roi_reg/config.image_scale,'LineWidth',1,'EdgeColor','r');
            
            try  %subsequent frames, update GUI
                set(im_handle, 'CData', I_orig)
                set(rect_handle, 'Position', roi_reg/config.image_scale)
                set(text_handle, 'string', int2str(frame_id));
            catch
                return
            end
        end

        feature_map = imresize(BC,config.ratio,'nearest');  % get the feature map of candiadte region
        ratio_x = size(BC,2)/size(feature_map,2);
        ratio_y = size(BC,1)/size(feature_map,1);
        detX = im2colstep(feature_map,[sampler.template_size(1:2), size(BC,3)],[1, 1, size(BC,3)]);
        x_sz = size(feature_map,2)-sampler.template_size(2)+1;
        y_sz = size(feature_map,1)-sampler.template_size(1)+1;
        [X Y] = meshgrid(1:x_sz,1:y_sz);
        detY = repmat(tracker.output,[numel(X),1]);
        detY(:,1) = (X(:)-1)*ratio_x + sampler.roi(1);
        detY(:,2) = (Y(:)-1)*ratio_y + sampler.roi(2);
        
        % detect the object
        % detX is feature(Lab+LIF+Explicit feature map), w0 is linear classifer
        % because we use linear w0, we can evaluate the candidate region by simple dot product
        score = w0 * detX;  
        [~,maxInd]=max(score);     
        output = detY(maxInd, :);  % detect the target position by maximal response
        % end to detect the object
        
        if config.display
            figure(1) 
            res = output;
            res(1:2) = res(1:2) - config.padding;
            res = res/config.image_scale;
            % display the object
            % rectangle('position',res,'LineWidth',2,'EdgeColor','b');
            set(rect_handle2, 'Position', res)
        end
        
        step = round(sqrt((y_sz*x_sz)/120));
        mask_temp = zeros(y_sz,x_sz);
        mask_temp(1:step:end,1:step:end) = 1;
        mask_temp = mask_temp > 0;
        mask_temp(maxInd) = 0;
        k = k+1;
        
        % construct the training set from the current tracking results.
        % detX(:,maxInd) (tracking results) is true output, its loss is zero.
        patterns{k}.X = [detX(:, maxInd)'; detX(:,mask_temp(:))'];
        patterns{k}.X = repmat(patterns{k}.X(1, :), size(patterns{k}.X, 1),1) - patterns{k}.X;
        patterns{k}.Y = [detY(maxInd, :); detY(mask_temp(:),:)];
        patterns{k}.lossY = 1 - getIOU(patterns{k}.Y,output);
        patterns{k}.supportVectorNum=[];
        patterns{k}.supportVectorAlpha=[];
        patterns{k}.supportVectorWeight=[];
        [w0,patterns]=dlssvmOptimization(patterns,params, w0);
        k=size(patterns,2);
    end

    timer = timer + toc;
    res = output;
    res(1:2) = res(1:2) - config.padding;
    result.res(frame_id-start_frame+1,:) = res/config.image_scale;
end

result.fps = result.len/timer;

clearvars -global sampler tracker config finish 
