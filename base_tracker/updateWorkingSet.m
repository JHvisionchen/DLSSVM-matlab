function [patterns,H,id,flag]=updateWorkingSet(patterns,w0,patternID,mode)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function: modify one structural output of  the patterns{patterID} as  % 
%           support vector to be optimized next                         % 
% parameters:                                                           %
%   patterns:                                                           %
%   w0: classifer                                                       %
%   patternID:                                                          %
%   mode:                                                               %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

thr=0.0001;
%start=cputime;
if (sum(w0) == 0)   % for the first frame
    score=patterns{patternID}.lossY; 
else  % for the other frame
    score = patterns{patternID}.lossY' - w0 * patterns{patternID}.X';
end

% find the most violated index (label), its dual coefficient 
% will be updated
[H,id]=max(score);  

if H<=thr && mode==1
    flag=0;
    return;
end

flag=1;

if ~ismember(id,patterns{patternID}.supportVectorNum)
    alpha=0.0000001;   % new dual coefficient
    weight=0;
    patterns{patternID}.supportVectorNum=[patterns{patternID}.supportVectorNum,id];
    patterns{patternID}.supportVectorAlpha=[patterns{patternID}.supportVectorAlpha,alpha];
    patterns{patternID}.supportVectorWeight=[patterns{patternID}.supportVectorWeight,weight];
end