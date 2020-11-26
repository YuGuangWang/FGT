function treeG = addweight1(treeG)
% adds weights for each level of the chain treeG
% INPUTS:
%  treeG - a chain with clusters at each level
%  treeG - the chain at level has weight property; treeG{j}.w

J = length(treeG);
N = length(treeG{1}.clusters);
W = zeros(J,N);
W(1,:) = 1; % equal weights for the first level (i.e. the finest level)
for j = 1:J
    clusterj = treeG{j}.clusters;
    Nj = length(clusterj);
    w = zeros(1,Nj);
    for k = 1:Nj
        cj_k = length(clusterj{k});
        w(k) = 1/sqrt(cj_k);
    end
    treeG{j}.w = w;
    % compute weights W_k^{(j)}
    if j>1
        wj = w;
        wj1 = zeros(1,N);
        % expand the weights from level j to 1
        for k = 1:Nj
            % find the indices of level 1 who have the same parent is the kth
            % vertex of level j
            idx = tree_idx2(treeG,k,j,1);
            wj1(idx) = wj(k);
            W(j,:) = W(j-1,:).*wj1;
        end
    end
end

treeG{end}.W = W;