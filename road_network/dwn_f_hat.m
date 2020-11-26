function f_hat_dwn = dwn_f_hat(f_hat,nj1)
%% Downsample f_hat vector to level j-1 with length Nj1
% 
% INPUT
%  f_hat      - a vector of size nj > nj1.
% 
% OUTPUT
%  f_hat_down - a vector which components are those of f_hat from 1 to nj1
%

f_hat_dwn = f_hat(1:nj1);
end