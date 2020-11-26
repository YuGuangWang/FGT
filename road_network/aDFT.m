function f = aDFT(c,treeG)
% fast computes the adjoint Discrete Fourier Transform (aDFT) of
% a sequanece c, using chain (or tree) structure based on Haar 
% Global Orthonormal Basis for the chain.
% This function brings the signal from Fourier domain back to time domain.
%
% INPUTS
%  c      -  sequence with length numel(treeG{1}.clusters)
%  treeG  -  a tree with a hierarchical orthonormal basis treeG{j}.u
% OUTPUT
%  f      -  DFT of c under the Haar orthonormal basis of treeG

% % compute s(c,v^{(j)}_k) and weights for each level of the treeG

Ntr = length(treeG);
cls = treeG{Ntr}.clusters;
N0  = length(cls);
v0  = treeG{Ntr}.v0;
V   = treeG{Ntr}.V;

s   = zeros(1,N0);

for ell = 1:N0
    
    v = V{ell};
    val1 = v{1};
    sup1 = v{2}; i1 = sup1(1); i2 = sup1(2);
    val2 = v{3};
    sup2 = v{4}; i3 = sup2(1); i4 = sup2(2);
    
    s(i1:i2) = s(i1:i2) + c(ell)*val1;
    s(i3:i4) = s(i3:i4) + c(ell)*val2;
     
end
s = s.*v0;

for j = Ntr-1:-1:1
    
    cls = treeG{j+1}.clusters;
    N0  = numel(cls);
    v0  = treeG{j}.v0;
    N1  = length(v0);
    V   = treeG{j}.V;
    
    s0  = s;
    s   = zeros(1,N1);
    i   = 1;
    
    for ell = 1:N0
        % extension part. 
        
        s(cls{ell}) = s0(ell)/sqrt(numel(cls{ell}));
        cls_ell = cls{ell};
        kl  = numel(cls_ell);
        
        % orthonormalization part.
        for k = 2:kl
            
            v     = V{i};
            val1  = v{1};
            sup1  = v{2}; i1 = sup1(1); i2 = sup1(2);
            val2  = v{3};
            sup2  = v{4}; i3 = sup2(1); i4 = sup2(2);
            
            s(cls_ell(i1:i2)) = s(cls_ell(i1:i2))+ val1*c(N0+i)*v0(cls_ell(i1:i2));
            s(cls_ell(i3:i4)) = s(cls_ell(i3:i4))+ val2*c(N0+i)*v0(cls_ell(i3:i4));
            
            i = i+1;          
            
        end
    end
end
f = s;