function [w0, patterns]=svBudgetMaintain(w0, patterns)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function: maintain the number of support vectors less than budgets.
% parameters:
%    w0: classifer
%    patterns:
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n=size(patterns,2);
minWeight=1000;

for i=1:n
    k=size(patterns{i}.supportVectorNum,2);
    
    if i==1 && k==1  
        continue;
    end
    for j=1:k
        w=patterns{i}.supportVectorWeight(j);
        if w<minWeight
            minWeight=w;
            idPat=i;
            idSamp=j;
        end
    end
end

k=size(patterns{idPat}.supportVectorNum,2);

if k<=1
    alpha = patterns{idPat}.supportVectorAlpha(1);
    num = patterns{idPat}.supportVectorNum(1);
    w0 = w0 - alpha * patterns{idPat}.X(num, :);
    patterns(idPat)=[];
else
    alpha = patterns{idPat}.supportVectorAlpha(idSamp);
    num = patterns{idPat}.supportVectorNum(idSamp);
    w0 = w0 - alpha * patterns{idPat}.X(num, :);
    patterns{idPat}.supportVectorNum(idSamp)=[];
    patterns{idPat}.supportVectorAlpha(idSamp)=[];
    patterns{idPat}.supportVectorWeight(idSamp)=[];
end