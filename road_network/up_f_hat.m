function f_hat_up = up_f_hat(f_hat,Nj)
%% Upsample f_hat vector to level j with length Nj
% 
% INPUT
%  f_hat      - a vector of size nj1 < nj.
% 
% OUTPUT
%  f_hat_up   - a vector which components are those of f_hat from 1 to nj,
%               from 1:nj1, f_hat_down is f_hat, from nj1+1:nj, f_hat_down is zeros
%

%%
Nj1 = numel(f_hat);
f_hat_up = zeros(1,Nj);
f_hat_up(1:Nj1) = f_hat;
end