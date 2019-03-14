function [w0, patterns]=updateOneAlpha(patterns,w0,params,idPat)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function: update one dual coefficient of patterns(idPat) 
%           please refer to eq(5)~eq(8)
% parameters:
%   patterns: training set with support vectors
%   w0: classifer
%   params:
%   idPat: training sample to be processed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n=size(patterns{idPat}.supportVectorNum,2);
if n==0
    patterns(idPat)=[];
    return;
end

svn=patterns{idPat}.supportVectorNum;  % get all support vectors of pattern id 
if isempty(sum(w0) == 0)  % for initial state
    mH=1;
else
    mH = patterns{idPat}.lossY(svn, :)' - w0 * patterns{idPat}.X(svn, :)';
end

[D,ind]=sort(mH,'ascend');

patterns{idPat}.supportVectorNum=patterns{idPat}.supportVectorNum(ind);
patterns{idPat}.supportVectorAlpha=patterns{idPat}.supportVectorAlpha(ind);
patterns{idPat}.supportVectorWeight=patterns{idPat}.supportVectorWeight(ind);

% yi needed to upated is placed in the end of support vectors
sampleID=patterns{idPat}.supportVectorNum(n);
H = patterns{idPat}.lossY(sampleID) - w0 * patterns{idPat}.X(sampleID, :)';

alpha_old=patterns{idPat}.supportVectorAlpha(n);
s=params.lambda-(sum(patterns{idPat}.supportVectorAlpha)-alpha_old);
kerProduct = patterns{idPat}.X(sampleID,:) * patterns{idPat}.X(sampleID,:)';
d=H/kerProduct;
alpha=min(max(alpha_old+d,0),s);
d = alpha - alpha_old;
w0 = w0 + d * patterns{idPat}.X(sampleID,:);
weight=alpha*alpha*kerProduct;
if alpha==0
    [patterns,deletePat]=svBudgetMaintain_zeros(patterns,idPat,sampleID);
    if deletePat==1
        return;
    end
else
    patterns{idPat}.supportVectorAlpha(n)=alpha;
    patterns{idPat}.supportVectorWeight(n)=weight;
end