function fhat = DFT(f,treeG)
% fast computes the Discrete Fourier Transform (DFT)
% of a vector f for the tree treeG of a graph.
% This function brings the signal from time domain to Fourier domain.
%
% INPUTS
%  f      -  real-valued vector with length numel(treeG{1}.clusters)
%  treeG  -  a tree with a Haar global orthonormal basis
% OUTPUT
%  fhat   -  aDFT of f under the Haar global orthonormal basis for treeG

% compute \mathcal{S}^{(j)} for j=1,...,J

J  = length(treeG);
s = f;
N  = length(f);
fhat = zeros(1,N);
for j = 1:J-1
    
    cls = treeG{j+1}.clusters;
    N0  = length(cls);
    v0  = treeG{j}.v0;
    V   = treeG{j}.V;
    s0  = zeros(1,N0);
    i   = 1;
    for ell = 1:N0
        
        cls_ell = cls{ell};
        
        kl      = numel(cls_ell);       
        
        for k = 2:kl
            
            % u_ell
            v     = V{i};            
            val1  = v{1};
            sup1  = v{2}; i1 = sup1(1); i2 = sup1(2);
            val2  = v{3};
            sup2  = v{4}; i3 = sup2(1); i4 = sup2(2);
            
            fhat(N0+i) = fhat(N0+i) + val1*sum(s(cls_ell(i1:i2)).*v0(cls_ell(i1:i2)));
            
            fhat(N0+i) = fhat(N0+i) + val2*sum(s(cls_ell(i3:i4)).*v0(cls_ell(i3:i4)));            
          
            i = i+1;
        end
       
        s0(ell)  = sum(s(cls_ell))/sqrt(numel(cls_ell));
    end
    s = s0;
end

v0  = treeG{end}.v0;
N0  = length(v0);
V   = treeG{end}.V;
for ell = 1:N0
    
    v = V{ell};
    val1 = v{1};
    sup1 = v{2}; i1 = sup1(1); i2 = sup1(2);
    val2 = v{3};
    sup2 = v{4}; i3 = sup2(1); i4 = sup2(2);
    
    fhat(ell) = fhat(ell)+ sum(s(i1:i2).*v0(i1:i2))*val1;
    fhat(ell) = fhat(ell)+ sum(s(i3:i4).*v0(i3:i4))*val2;   
    
end


