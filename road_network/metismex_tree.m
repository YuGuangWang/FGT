function treeG = metismex_tree(adjacency_matrix, K)
% adjacency_matrix: Symmetric weighted adjacency matrix (full).
% K: A list of the number of nodes of the subgraphs.
%    Do not include the number of nodes of the original graph.

levels = numel(K) + 1;
parents = cell(1, levels - 1);
adj_list = cell(1, levels);
W = adjacency_matrix;
[~, N_start] = size(W);
adj_list{1} = W;

for k = 1:(levels - 1)
    % pair the vertices and construct the root vector
    [idx_row, idx_col, val] = find(W);
    % idx_row, idx_col, val are all column vectors
    [~, perm] = sort(idx_row);
    rr = idx_row(perm);
    cc = idx_col(perm);
    vv = val(perm);
    % perm, rr, cc, vv are all column vectors
    % Using metismex at each level
    cluster_id = metismex('PartGraphRecursive', sparse(W), K(k));
    cluster_id = cluster_id + 1;
    % cluster_id is a row vector
    parents{k} = cluster_id;
    % Compute the edges weights for the new graph
    nrr = cluster_id(rr);
    ncc = cluster_id(cc);
    nvv = vv;
    Nnew = max(cluster_id);
    % saving the coarsened adj
    W = full(sparse(nrr, ncc, nvv, Nnew, Nnew));
    W = (W + W') / 2;
    adj_list{k + 1} = W;
end
% Constructing the tree
treeG = cell(1, levels);
% Obtain the clusters and IDX, saved in treeG
for i = 1:levels
    % special case for the base level, corresponding to the original graph
    if i == 1
        parents_ini = 1:N_start;
        [~, idx_sort] = sort(parents_ini);
        sorted_records_array = parents_ini(idx_sort);
        [~, idx_start, ~] = unique(sorted_records_array);
        cluster_1 = np_split(idx_sort, idx_start(2:end));
        treeG{i}.IDX = 1:N_start;
        treeG{i}.clusters = cluster_1;
        treeG{i}.A = adj_list{i};
    else
        % the second level to the top level
        [~, idx_sort] = sort(parents{i - 1});
        sorted_records_array = parents{i - 1}(idx_sort);
        [~, idx_start, ~] = unique(sorted_records_array);
        cluster_temp = np_split(idx_sort, idx_start(2:end));
        treeG{i}.IDX = parents{i - 1};
        treeG{i}.clusters = cluster_temp;
        treeG{i}.A = adj_list{i};
    end
end

end






