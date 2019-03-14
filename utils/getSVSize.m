function nSVs=getSVSize(patterns)

n=size(patterns,2);
nSVs=0;
for i=1:n
    nSVs=nSVs+size(patterns{i}.supportVectorNum,2);
end