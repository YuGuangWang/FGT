function [ha,hb1,hb2] = filter_bank(t,Nj,Nj_1)
% computes the filter bank for control points N_j, Nj_1 given the variable t
%
% INPUTS:
% Nj, Nj_1     - control points, Nj > Nj_1
% OUTPUTS:
% ha, hb1, hb2 - low-pass filter ha and high-pass filters hb1 and hb2 at t

% a_hat
a_cR = (1+Nj_1)/2;
a_epsR = Nj_1-a_cR;
a_cL = -a_cR;
a_epsL = a_epsR;
% b_hat_1
b1_cL = a_cR;
b1_epsL = a_epsR;
b1_cR = (Nj_1+Nj)/2;
b1_epsR = Nj-b1_cR;
% b_hat_2
b2_cL = b1_cR;
b2_epsL = b1_epsR;
b2_cR = 2*Nj;
b2_epsR = 1;

ha  = hmask(t,a_cL,a_epsL,a_cR,a_epsR);  % supp(ha)  = [0,1/4]
hb1 = hmask(t,b1_cL,b1_epsL,b1_cR,b1_epsR);     % supp(hb1) = [1/8,1/2]
hb2 = hmask(t,b2_cL,b2_epsL,b2_cR,b2_epsR);         % supp(hb2) = [1/4,1/2]
end

function y = hmask(x,cL,epsL,cR,epsR)
%% Generate a filter   
%
%  y = hmask(x,cR,epsR,cL,epsL)
%
%  Generate a filter y(x) satisfies
%   1. y(x) is supported on [cL-epsL,cR+epsR] 
%   2. y(x) = 1 on [cL+epsL,cR-epsR]
%
% INPUT:
%   x           - data points on the real line R
%   (cL,cR)     - control points, essential support
%   (epsL,epsR) - shape parameters
%
% OUTPT:
%   y           - y(x) supported on [cL-epsL,cR+epsR]

%%
y = (x <= cR-epsR).*(x >= cL+epsL);

y = y + sin(pi/2*nu((x-cL+epsL)/epsL/2)).*(x < cL+epsL);

y = y + cos(pi/2*nu((x-cR+epsR)/epsR/2)).*(x > cR-epsR);
end

function y = nu(t)
%% Elementary funciton nu(t) satisfies
%
%  1. nu(t) + nu(1-t) = 1
%  2. nu(t) = 0 for t <= 0
%  3. nu(1) = 1 for t >= 1

y = (t >= 0 & t <1).*t.^4.*(35-84*t+70*t.^2-20*t.^3) + (t >=1);
end