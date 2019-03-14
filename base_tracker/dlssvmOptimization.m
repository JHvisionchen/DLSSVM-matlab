%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	DLSSVM optimization method described in
%	"Object Tracking via Dual Linear Structured SVM and Explicit Feature Map", 
%   Jifeng Ning, Jimei Yang, Shaojie Jiang, Lei Zhang and Ming-Hsuan Yang, 
%   CVPR,Las Vegas, 4266-4274, June, 2016.
%	
%	Copyright (C) 2016 Jifeng Ning, Shaojie Jiang
%	e-mail:shaojiejiang@126.com, jf ning@sina.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [w0, patterns]=dlssvmOptimization(patterns,params, w0)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% parameters:
%   patterns: all training set
%   params:
%   w0: initial discriminative classifer
% returns:
%   w0: updated classifer
%   patterns: updated training set, whose support vectors may be updated
%             and some traing set may be removed.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P=5;
% start=cputime;
for p=1:P
    n=size(patterns,2);
    % select a sample to update its dual cofficient via DCD optimization
    if p==1
        idPat=n;
    else
        idPat = n - floor((p-1)*n/P);
    end

    patterns=updateWorkingSet(patterns,w0, idPat,1);
    [w0, patterns]=updateOneAlpha(patterns,w0,params,idPat); % DCD optimizatoion
    
    % maintian the budgets for support vectors 
    svSize=getSVSize(patterns);
    while svSize > params.nBudget
        [w0, patterns]=svBudgetMaintain(w0, patterns);
        svSize=getSVSize(patterns);
    end
    
    for q=1:10
        n=size(patterns,2);
        % select a sample to update its dual cofficient via DCD optimization
        idPat = n - floor(n*(q-1)/10);
        [w0, patterns]=updateOneAlpha(patterns,w0,params,idPat); % DCD optimization
    end
end