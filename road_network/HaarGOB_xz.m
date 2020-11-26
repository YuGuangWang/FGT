function treeG = HaarGOB(treeG)
% HaarGOB generates Haar Global Orthonormal Basis for a chain or tree treeG
%
% INPUTS:
%  treeG  - chain or tree for a graph
%
% OUTPUTS:
%  treeG1 - the updated chain treeG which is added for each level a Haar
%           orthonormal basis and for the bottom level the basis is the
%           Haar Global Orthonormal Basis

% number of level of the chain (or tree)
Ntr = numel(treeG);

%% reorder chain (optional)
% reordering each level so that in each level the nodes are in the
% descent order of degrees

% compute u_l^c for level J_0 (top level)
clusterJ0 = treeG{Ntr}.clusters;
N0        = numel(clusterJ0);
% Each clusterJ0{j} is a vertex indexed as j for G_J0. j = 1, ..., N0;

% Construct orthonormal basis U=V*diag(v0) on G_J0.
% data structure of V = {v_1,...,v_N0}; v_l: {val1, supp1, val2, supp2}, i.e.
% v_l(supp1) = val1, v_l(supp2) = val2, and supp1\cup supp2 = {1:N0}.
% Note the the SPOC of each v_l is at most 2.
V     = cell(1,N0);
v0    = 1/sqrt(N0)*ones(1,N0);     % diagonal vector
V{1}  = {1, [1, N0], 0, [1, 0]};     % initial one v_1 = 1.
for ell = 2:N0
    val1   = sqrt((N0-ell+1)/(N0-ell+2))*sqrt(N0);  % value
    sup1   = [ell-1,ell-1];                         % support index
    val2   = -val1/(N0-ell+1);
    sup2   = [ell,N0];
    V{ell} = {val1, sup1, val2, sup2};
end
treeG{Ntr}.V  = V;
treeG{Ntr}.v0 = v0;

%% compute the next level orthonormal basis ulk and stored into u
for j_tr = Ntr-1:-1:1    

    
    v0_pre   = treeG{j_tr+1}.v0;
%    V_pre    = treeG{j_tr+1}.V;      
    clusters = treeG{j_tr+1}.clusters;  
    N0       = numel(clusters);


    N1 = numel(treeG{j_tr}.clusters);
    V  = cell(1,N1-N0);
    v0 = zeros(1,N1);
    
    i        = 0;%N0;
    
    for ell = 1:N0
        
        % extension
        idx     = clusters{ell};       % clusterl = {indices of next level}
        kl      = numel(idx); 
        c0      = v0_pre(ell)/sqrt(kl);
        v0(idx) = c0;
        
%         vl      = V_pre{ell};
%         val1    = vl{1};
%         supp1   = vl{2};
%         val2    = vl{3};
%         supp2   = vl{4};        
%         V{ell}  = {val1, [clusters{supp1}],val2, [clusters{supp2}]};
        
        % orthonormalization
        if kl > 1
            for k = 2:kl
                i      = i+1;
                val1   = sqrt((kl-k+1)/(kl-k+2))/c0;
                val2   = -val1/(kl-k+1);
                supp1  = [k-1,k-1];
                supp2  = [k,kl];
                V{i}   = {val1, supp1, val2, supp2};
            end            
        end
        
    end
    treeG{j_tr}.v0 = v0;
    treeG{j_tr}.V = V;
end