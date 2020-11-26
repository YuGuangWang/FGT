%% Generating Orthonnormal Basis on the Coarse-Grained Chain (ONBC)

%% Generate a undirected weighted Graph
clear,clc
close all

% subdir = '\'; % Windows 
subdir = '/'; % Unix and Linux

ld = ['minnesota' '.mat'];
load(ld)

%%
N_G = size(A,1);
nodename = cell(1,N_G);
for i = 1:N_G
    nodename{i} = ['v' num2str(i)];
end
G = graph(A,nodename); %%%
% plot the graph
% plt_G = figure;
% p = plot(G);
% p.NodeColor = 'red';
% layout(p,'force2');

% axis off
% 
% sv = ['Graph_N' num2str(N_G) '.jpg'];
% print(plt_G,'-djpeg','-r300',sv);

% % number of nodes of subgraphs
K = [1000 300 100];

t0_kmean_tree = tic;

% treeG = tree_kmean(G,A,K);
treeG = metismex_tree(A, K);
t_kmean_tree = toc(t0_kmean_tree);
treeG{:}

fprintf('\n')
fprintf('Total CPU time of k-mean clustering of tree generating: %.2f s.\n',t_kmean_tree)

%% plot all figures
% for i = 1:length(treeG)
%     G_i = treeG{i}.G;
%     figure
%     plot(G_i);
% end

% display tree structure
% fprintf('\n')
% fprintf('The tree of the graph G is\n')
% treeG{:}

%% Obtain orthonormal basis of the tree by ONBC_Laplacian using Gram-Schmidt 
 % orthogonalization

% t0_ONBC = tic;
% 
% fprintf('\n')
% fprintf('Compute the orthonormal basis for each subgraph ...');

% treeG1 = ONBC_Laplacian(treeG);
% [V,treeG1] = ONBC_Laplacian_1(treeG);

% % display orthonormal basis
% for k = 1:numel(treeG1)
%     fprintf('\n')
%     fprintf('The orthonormal basis for the subgraph G_{%d} is\n',k);
%     treeG1{k}.V;
% end

% time
% t_ONBC = toc(t0_ONBC);

% fprintf('\n')
% fprintf(' - total CPU time of ONBC: %.2f s.\n',t_ONBC)

%%
% halph  = @(t) hmask(t,-5/8,5/8,3/8,3/8);  % supp(hal)  = [0,1/4]
% hg1 = @(t) hmask(t,-1/4,1/4,1/4,1/4);     % supp(hb1) = [1/8,1/2]
% hg2 = @(t) hmask(t,1/2,3/2,1/4,1/2);
% hg1    = @(t) hmask(1/2*t,-3/16,1/16,3/16,1/16);  % supp(ha)  = [0,1/4]
% hg2    = @(t) hmask(1/2*t,3/16,1/16,3/8,1/8);     % supp(hb1) = [1/8,1/2]
% hg3    = @(t) hmask(1/2*t,3/8,1/8,1,1/8);         % supp(hb2) = [1/4,1/2]

% ha  = @(t) hmask(t,-3/16,1/16,3/16,1/16);  % supp(ha)  = [0,1/4]
% hb1 = @(t) hmask(t,3/16,1/16,3/8,1/8);     % supp(hb1) = [1/8,1/2]
% hb2 = @(t) hmask(t,3/8,1/8,1,1/8);         % supp(hb2) = [1/4,1/2]

% %% for the jth layer
% j = 4;
% N(j-1) = 3;
% N(j) = 6;
% % a_hat
% a_cR = (1+N(j-1))/2;
% a_epsR = N(j-1)-a_cR;
% a_cL = -a_cR;
% a_epsL = a_epsR;
% % b_hat_1
% b1_cL = a_cR;
% b1_epsL = a_epsR;
% b1_cR = (N(j-1)+N(j))/2;
% b1_epsR = N(j)-b1_cR;
% % b_hat_2
% b2_cL = b1_cR;
% b2_epsL = b1_epsR;
% b2_cR = 2*N(j);
% b2_epsR = 1;
% 
% t = -1:0.1:N(j)+1;
% 
% ha  = hmask(t,a_cL,a_epsL,a_cR,a_epsR);  % supp(ha)  = [0,1/4]
% hb1 = hmask(t,b1_cL,b1_epsL,b1_cR,b1_epsR);     % supp(hb1) = [1/8,1/2]
% hb2 = hmask(t,b2_cL,b2_epsL,b2_cR,b2_epsR);         % supp(hb2) = [1/4,1/2]

% %%
% Nj = 9;
% Nj_1 = 3;
% t = -1:0.1:Nj+1;
% [ha,hb1,hb2] = filter_bank(t,Nj,Nj_1);
% 
% figure;
% plot(t,ha,t,hb1,t,hb2,'linewidth',2)
% grid on
% ylim([0,1.1])
% 
% % filtbk = {halph,hg1,hg2,hg3};
% filtbk = {ha,hb1,hb2};

%% test DFT and aDFT
% add weight to chain
treeG = addweight1(treeG);
% generate orthonormal basis for chain
t0 = tic;
fprintf('\n')
fprintf('Compute the orthonormal basis for each level of chain ...\n');
treeG1 = HaarGOB_xz(treeG);
t1 = toc(t0);
fprintf(' - CPU time of Haar basis generation: %.4f s.\n',t1)
fprintf('\n')
c     = ones(1,N_G);
t0 = tic;
f     = aDFT(c,treeG1);
t1 = toc(t0);
fprintf('CPU time of DFT: %.6f s.\n', t1);
% Compute fh from f
t0 = tic;
fh    = DFT(f,treeG1);
t1 = toc(t0);
fprintf('CPU time of aDFT: %.6f s.\n', t1);
fprintf('l2 error for aDFT and DFT: %.4e\n',norm(c-fh));

%% FGT Decomposition
fprintf('\n')
fprintf('FGT Decomposition for treeG ...\n');

t0 = tic;

J  = numel(treeG1);        % level of decomposition
coefs = cell(1,J);

for j = 1:J-1
    nj       = numel(treeG1{j}.clusters);   % n_j, number of vertices in level j
    nj1      = numel(treeG1{j+1}.clusters); % n_{j-1}, number of vertices in level j-1
    % generate filter bank for level j
    ell      = 1:nj;
    [ha, hb1, hb2] = filter_bank(ell,nj,nj1);
    sh = ha.^2+hb1.^2+hb2.^2;
%     fprintf('Level %d, ha.^2+hb1.^2+hb2.^2 = %.2e\n',j,sh)
%     figure;
%     plot(ell,ha,ell,hb1,ell,hb2,'linewidth',2)
%     grid on
%     ylim([0,1.1])
    % % for high-pass filter
    coefs{j} = cell(1,2);                   % high-pass filtered coefficients at level j
    % convolution with high-pass filter b
    fhb1    = bsxfun(@times,fh,hb1);    % convolute with hb1
    fhb2    = bsxfun(@times,fh,hb2);    % convolute with hb2
    % go back to time domain
    % wb1     = aDFT(fhb1,treeG1(j:end));
    % wb2     = aDFT(fhb2,treeG1(j:end));
    coefs{j}{1} = fhb1;
    coefs{j}{2} = fhb2;
    % % for low-pass filter
    fha     = bsxfun(@times,fh,ha);                        % conv with ha
    fh      = dwn_f_hat(fha,nj1);                          % downsampling  
end
% va          = aDFT(fh,treeG1(J));
coefs{J}    = fh;

% time
t1 = toc(t0);
fprintf(' - CPU time of FGT_Dec: %.6f s.\n',t1)

% test by straightforward computing DFT
fprintf('\n')
fprintf('FGT Recontruction for treeG ...\n');

%% FGT Reconstruction
t0 = tic;

J  = numel(treeG1);        % level of decomposition

% Compute fh from low-pass coeff at level J
fh1    = coefs{J};
% fh1   = DFT(va,treeG1(J));
% fprintf(' - Decomposition error: %.4e\n',norm(fh-fh1)/norm(fh))

for j = J-1:-1:1
    % eigenvalues at levels j-1 and j   
    nj1     = numel(treeG1{j+1}.clusters);   % n_{j-1}, number of vertices in level j-1
    nj      = numel(treeG1{j}.clusters);   % n_{j}, number of vertices in level j
    ell     = 1:nj;
    % generate filter bank for level j
    [ha, hb1, hb2] = filter_bank(ell,nj,nj1);
    % % for low-pass filter
    % upsamping and convolution of low-pass filter    
    fh      = up_f_hat(fh,nj);             % upsampling for low-pass
    fh      = bsxfun(@times,fh,ha);      % conv with ha    
    % for high-pass filter
    fhb1     = coefs{j}{1};                       % coeff in time domain at level j-1 of high-pass
    fhb2     = coefs{j}{2};
    % fhb1    = DFT(wb1,treeG1(j:end));    % adjoint DFT to evaluate Fourier coeff at level j-1 of high-pass
    % fhb2    = DFT(wb2,treeG1(j:end));
    % convolution with high-pass filter b1 and b2 and add the low pass
    fh      = fh + bsxfun(@times,fhb1,hb1);     
    fh      = fh + bsxfun(@times,fhb2,hb2);
end
% DFT to evaluate f
f_rec       = aDFT(fh,treeG1);

% time
t1 = toc(t0);

fprintf(' - CPU time of FGT_Rec: %.6f s.\n',t1)
fprintf(' - Reconstruction error: %.4e\n',norm(f_rec-f))

%% Plot low-pass coefficients G4
% fprintf('Plot low-pass coefficients\n');
% x=xy(:,1);
% y=xy(:,2);
% 
% % marker size
% msize = 100;
% 
% color_coef = ones(N_G, 1);
% color_coef = color_coef / 0;
% 
% for i = 1:numel(coefs{4})
%     idx = tree_idx2(treeG, i, 4, 1);
%     color_coef(idx(1)) = coefs{4}(i);
% end
% 
% bg = find(color_coef == Inf);
% color = find(color_coef ~= Inf);
% assert((length(bg) + length(color)) == N_G);
% % plot_order = [find(color_coef == 0)', find(color_coef)']';
% 
% fig = figure;
% set(gcf,'renderer','zbuffer');
% fprintf('Displaying traffic graph\n');
% set(gcf,'OuterPosition',[0,600,400,400]);
% hold on
% scatter(x(bg), y(bg), msize, [0.9 0.9 0.9], '.');
% scatter(x(color), y(color), msize, color_coef(color), '.');
% colormap('jet');
% % c = colorbar('Location', 'northoutside');
% caxis([0 1])
% % c.Ticks = [-0.6, -0.3, 0, 0.3, 0.6, 0.9];
% % c.FontSize = 16;
% set(gca,'Xtick',[]);
% set(gca,'Ytick',[]);
% axis tight
% axis off

%% Plot high-pass coefficients G3, coefs{3}{1}
% fprintf('Plot high-pass coefficients coefs{3}{1}\n');
% x=xy(:,1);
% y=xy(:,2);
% 
% % marker size
% msize = 100;
% 
% color_coef = ones(N_G, 1);
% color_coef = color_coef / 0;
% 
% for i = 1:numel(coefs{3}{1})
%     idx = tree_idx2(treeG, i, 3, 1);
%     color_coef(idx(1)) = coefs{3}{1}(i);
% end
% 
% bg = find(color_coef == Inf);
% color = find(color_coef ~= Inf);
% assert((length(bg) + length(color)) == N_G);
% % plot_order = [find(color_coef == 0)', find(color_coef)']';
% 
% fig = figure;
% set(gcf,'renderer','zbuffer');
% fprintf('Displaying traffic graph\n');
% set(gcf,'OuterPosition',[0,600,400,400]);
% hold on
% scatter(x(bg), y(bg), msize, [0.9 0.9 0.9], '.');
% scatter(x(color), y(color), msize, color_coef(color), '.');
% colormap('jet');
% % c = colorbar('Location', 'northoutside');
% caxis([0 1])
% % c.Ticks = [0, 0.2, 0.4, 0.6, 0.8, 1];
% % c.FontSize = 16;
% set(gca,'Xtick',[]);
% set(gca,'Ytick',[]);
% axis tight
% axis off

%% Plot high-pass coefficients G3, coefs{3}{2}
% fprintf('Plot high-pass coefficients coefs{3}{2}\n');
% x=xy(:,1);
% y=xy(:,2);
% 
% % marker size
% msize = 100;
% 
% color_coef = ones(N_G, 1);
% color_coef = color_coef / 0;
% 
% for i = 1:numel(coefs{3}{2})
%     idx = tree_idx2(treeG, i, 3, 1);
%     color_coef(idx(1)) = coefs{3}{2}(i);
% end
% 
% bg = find(color_coef == Inf);
% color = find(color_coef ~= Inf);
% assert((length(bg) + length(color)) == N_G);
% % plot_order = [find(color_coef == 0)', find(color_coef)']';
% 
% fig = figure;
% set(gcf,'renderer','zbuffer');
% fprintf('Displaying traffic graph\n');
% set(gcf,'OuterPosition',[0,600,400,400]);
% hold on
% scatter(x(bg), y(bg), msize, [0.9 0.9 0.9], '.');
% scatter(x(color), y(color), msize, color_coef(color), '.');
% colormap('jet');
% % c = colorbar('Location', 'northoutside');
% caxis([0 1])
% % c.Ticks = [-0.6, -0.3, 0, 0.3, 0.6, 0.9];
% % c.FontSize = 16;
% set(gca,'Xtick',[]);
% set(gca,'Ytick',[]);
% axis tight
% axis off

%% Plot high-pass coefficients G2, coefs{2}{1}
% fprintf('Plot high-pass coefficients coefs{2}{1}\n');
% x=xy(:,1);
% y=xy(:,2);
% 
% % marker size
% msize = 100;
% 
% color_coef = ones(N_G, 1);
% color_coef = color_coef / 0;
% 
% for i = 1:numel(coefs{2}{1})
%     idx = tree_idx2(treeG, i, 2, 1);
%     color_coef(idx(1)) = coefs{2}{1}(i);
% end
% 
% bg = find(color_coef == Inf);
% color = find(color_coef ~= Inf);
% assert((length(bg) + length(color)) == N_G);
% % plot_order = [find(color_coef == 0)', find(color_coef)']';
% 
% fig = figure;
% set(gcf,'renderer','zbuffer');
% fprintf('Displaying traffic graph\n');
% set(gcf,'OuterPosition',[0,600,400,400]);
% hold on
% scatter(x(bg), y(bg), msize, [0.9 0.9 0.9], '.');
% scatter(x(color), y(color), msize, color_coef(color), '.');
% colormap('jet');
% % c = colorbar('Location', 'northoutside');
% caxis([0 1])
% % c.Ticks = [-0.6, -0.3, 0, 0.3, 0.6, 0.9];
% % c.FontSize = 16;
% set(gca,'Xtick',[]);
% set(gca,'Ytick',[]);
% axis tight
% axis off

%% Plot high-pass coefficients G2, coefs{2}{2}
fprintf('Plot high-pass coefficients coefs{2}{2}\n');
x=xy(:,1);
y=xy(:,2);

% marker size
msize = 100;

color_coef = ones(N_G, 1);
color_coef = color_coef / 0;

for i = 1:numel(coefs{2}{2})
    idx = tree_idx2(treeG, i, 2, 1);
    color_coef(idx(1)) = coefs{2}{2}(i);
end

bg = find(color_coef == Inf);
color = find(color_coef ~= Inf);
assert((length(bg) + length(color)) == N_G);
% plot_order = [find(color_coef == 0)', find(color_coef)']';

fig = figure;
set(gcf,'renderer','zbuffer');
fprintf('Displaying traffic graph\n');
set(gcf,'OuterPosition',[0,600,400,400]);
hold on
scatter(x(bg), y(bg), msize, [0.9 0.9 0.9], '.');
scatter(x(color), y(color), msize, color_coef(color), '.');
colormap('jet');
% c = colorbar('Location', 'northoutside');
caxis([0 1])
% c.Ticks = [-0.6, -0.3, 0, 0.3, 0.6, 0.9];
% c.FontSize = 16;
set(gca,'Xtick',[]);
set(gca,'Ytick',[]);
axis tight
axis off

%% Plot Colorbar
% ax = axes;
% colormap('jet');
% c = colorbar(ax);
% caxis([0 1]);
% % c.Ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
% c.FontSize = 20;
% ax.Visible = 'off';







