function [w0, patterns]=onlineStructSVMOptimization(patterns,find_MVC, params, w0)

P=5;

% start=cputime;
for p=1:P
    n=size(patterns,2);
    if p==1
        idPat=n;
    else
        %idPat=randi(n);
        idPat = n - floor((p-1)*n/P);
    end

    patterns=updateWorkingSet(patterns,w0, params,find_MVC,idPat,1);
    [w0, patterns]=updateAllAlpha(patterns,w0,params,idPat);
    svSize=getSVSize(patterns);
    while svSize > params.nBudget
        [w0, patterns]=svBudgetMaintain(w0, patterns);
        svSize=getSVSize(patterns);
    end
    
    for q=1:10
        n=size(patterns,2);
        idPat = n - floor(n*(q-1)/10);
        [w0, patterns]=updateAllAlpha(patterns,w0,params,idPat);
    end
end