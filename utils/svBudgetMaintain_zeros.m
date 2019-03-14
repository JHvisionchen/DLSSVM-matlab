function [patterns,deletePat]=svBudgetMaintain_zeros(patterns,s,id)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function: delete the support vector with dual coefficient zero.
% paramters:
%    patterns: training set
%    s: pattern ID
%   id: support vector related to structural output id of pattern(ID) will
%       be deleted
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

jj=find(patterns{s}.supportVectorNum==id);
k=size(patterns{s}.supportVectorNum,2);  % the number of support vectors of pattern s

deletePat=0;

if k<=1 
    if s==1  % pattern of the first frame does not deleted
        deletePat=0;
    elseif s~=1
        deletePat=1;
        patterns(s)=[];
    end
    return;
else
    patterns{s}.supportVectorNum(jj)=[];
    patterns{s}.supportVectorAlpha(jj)=[];
    patterns{s}.supportVectorWeight(jj)=[];
    deletePat=0;
end