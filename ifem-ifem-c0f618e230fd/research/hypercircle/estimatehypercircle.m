function [eta,etaTotal,sigmaD,etaResTotal] = estimatehypercircle(node,elem,uh,pde,bdFlag,option)
% [eta,etaTotal,sigmaDtotal] = estimatehypercircle(node,elem,uh,pde,bdFlag,option) 
% computes the robust equilibrated residual type a posterior error estimator 
% for the diffusion equation taking into account
% (1) Algebraic error
% (2) Dirichlet boundary fix
% using Hybridized mixed local problem in a non-vectorized fashion.
% Local variable naming convention:
% (following local indexing or not)_(Type)+(geometry)
% eta is the local error indicator = \|sigma^{D}\|_{L^2(K)}
% sigmaDtotal is the sigma^{D} in RT^0_{-1}
% References: 
% [1] Braess, Schoeberl 2007
% [2] Cai, Zhang 2012
% S. Cao 2016
% requirement: L. Chen's iFEM


%% Set up options
if ~exist('option','var')
    option.correction = 'none'; 
    option.DirFix = 'yes';
elseif ~isfield(option,'DirFix'), option.DirFix = 'yes';
end


%% retrieve matrix info
% option.solver = 'none';
% [~,~,eqn] = Poisson(node,elem,pde,bdFlag,option);
% ResAlg = eqn.b-eqn.Lap*uh; 

% algebraic residual = Violation of Galerkin orthogonality

%%
tic;

%% Data structure needed
N = size(node,1);
NT = size(elem,1);
T = auxstructure(elem);
edge = double(T.edge); 
NE = size(edge,1);
edge2elem = T.edge2elem;
elem2edge = T.elem2edge;
c = [3 1 2 3 1]; % cyclic group indexing of {1,2,3}: c(i)=i-1, c(i+2)=i+1

%%  Boundary 
idxDir = (bdFlag(:) == 1 | bdFlag(:) == 2);     % all Dirichlet & Neumann edges in bdFlag (and Neumann for simplicity)
isBdEdge = false(NE,1);
isBdEdge(elem2edge(idxDir)) = true;  % index of fixed boundary edges
idxBdEdge = find(isBdEdge);
idxIntEdge = find(~isBdEdge);
[idxBdNode,~,isBdNode] = findboundary(elem); 
idxIntNode = find(~isBdNode); % all interior nodes
isBdElem = isBdEdge(elem2edge(:,1)) | isBdEdge(elem2edge(:,2)) | isBdEdge(elem2edge(:,3));
% isBdElem only counts when an element has an edge lying on the boundary

%% quadrature set up
fquadorder = 3;
[lambda,weight] = quadpts(fquadorder);
phi = lambda;                 % linear bases
nQuad = size(lambda,1);

%% Diffusion coefficient

if ~isfield(pde,'d'), pde.d = []; end
if isnumeric(pde.d) && ~isempty(pde.d)
    K = pde.d;    % d is an array
elseif isempty(pde.d)
    pde.d = @(p) ones(size(p,1),1);
end
if ~isempty(pde.d) && ~isnumeric(pde.d)       % d is a function   
    K = zeros(NT,1);
    for p = 1:nQuad
		pxy = lambda(p,1)*node(elem(:,1),:) ...
			+ lambda(p,2)*node(elem(:,2),:) ...
			+ lambda(p,3)*node(elem(:,3),:);
        K = K + weight(p)*pde.d(pxy);      
   end
end


%% numerical flux
[Duh,area,Dlambda] = gradu(node,elem,uh);
Duh = Duh.*repmat(K,1,2); % flux
valence = accumarray(elem(:),ones(3*NT,1),[N 1]);
patchArea = accumarray(elem(:),repmat(area,3,1), [N 1]);

%% right hand side
if ~isfield(pde,'f') || (isreal(pde.f) && (any(pde.f)==0))
    pde.f = [];
end


%% compute the edge jump following the edge indexing
ne = -Dlambda.*repmat(2*area,[1 2 3]); % scaled outward normal vector

ne2e = zeros(NE,2); %scaled normal vector following edge indexing
JumpEdge = zeros(NE,1);
for kk = 1:3 % three local edges
    ne2e = ne2e + repmat(edge2elem(:,3)==kk,1,2).*ne(edge2elem(:,1),:,kk);
end

% interior edge. If this is used for bdEdge, the jump will be zero
JumpEdge(idxIntEdge,:) = dot(Duh(edge2elem(idxIntEdge,1),:)...
                       -Duh(edge2elem(idxIntEdge,2),:),ne2e(idxIntEdge,:),2);
clear ne ne2e

%% Compute the local element residual for the liner element
ResElem2v = zeros(NT,3); % (f-cu_h,\phi_k)_{K_j} for k-th vertex of j-th element

if isreal(pde.f) % f is a real number or vector and not a function
   switch length(pde.f)
       case NT  % f is piecewise constant
         ResElem2v = repmat(pde.f.*area/3,3,1);
       case N   % f is piecewise linear
         ResElem2v(:,1) = area.*(2*pde.f(elem(:,1)) + pde.f(elem(:,2)) + pde.f(elem(:,3)))/12;
         ResElem2v(:,2) = area.*(2*pde.f(elem(:,2)) + pde.f(elem(:,3)) + pde.f(elem(:,1)))/12;
         ResElem2v(:,3) = area.*(2*pde.f(elem(:,3)) + pde.f(elem(:,1)) + pde.f(elem(:,2)))/12;
       case 1   % f is a scalar e.g. f = 1
         ResElem2v = repmat(pde.f*area/3,3,1);
   end
end
if ~isempty(pde.f) && ~isreal(pde.f)  % f is a function 
    ResElem2v = zeros(NT,3);
    for p = 1:nQuad
		% quadrature points in the x-y coordinate
		pxy = lambda(p,1)*node(elem(:,1),:) ...
			+ lambda(p,2)*node(elem(:,2),:) ...
			+ lambda(p,3)*node(elem(:,3),:);
		fp = pde.f(pxy);
        for zi = 1:3
            ResElem2v(:,zi) = ResElem2v(:,zi) + weight(p)*phi(p,zi)*fp;
        end
    end
    ResElem2v = ResElem2v.*repmat(area,1,3);
end
ResElem = sum(ResElem2v,2);
clear pxy fp

%% Constructing the local problems
e2v = sparse([1:NE,1:NE],[edge(:,1); edge(:,2)],1,NE,N);
t2v = sparse([1:NT,1:NT,1:NT],elem,1,NT,N);
sigmaD = zeros(NT,3); % total correction \sigma^{\Delta}
ResCor = zeros(NT,3); % element residual projection error from iter
etaz = zeros(N,1); % local estimator for patch omega_{z}

tic;
for zi = idxIntNode' % vertex z_i
    %% local data structures
    idxElemZ = find(t2v(:,zi)); % all the element in patch \omega_{z_i}
    loc_NT = valence(zi);
    elemZ = elem(idxElemZ,:);
%     elem2edgeZ = elem2edge(idxElemZ,:);
    
    idxE2vz = find(e2v(:,zi)); % this returns all the edges connecting Z with global indices
    idxIntEdgeZ = idxE2vz(~ismember(idxE2vz,idxBdEdge)); % interior edge no Dirichlet edges
    idxAllEdgeZ = unique(elem2edge(idxElemZ,:));
    [~,loc_IdxIntEdge] = ismember(idxIntEdgeZ,idxAllEdgeZ);
    
    %% find the connections
    loc_T = auxstructure(elemZ);
    loc_Edge2elem = loc_T.edge2elem;
    
    %% Dofs setting up
    % Dofs correspond to the same local elem2edge(idxElemZ,:)
    % but the index is the local index for RT_{-1}
    loc_Dof = (1:3*loc_NT)';
    loc_elem2Dof = reshape(loc_Dof,3,loc_NT)'; 
    loc_BdFlag = setboundary(node,elemZ,'Dirichlet');
    loc_Ndof = size(loc_Dof,1);
    
    % find local interior edges (RT_{-1} numbering)
    [loc_freeDof, ~, loc_bdDof] = dofdisRT(zi,elemZ,loc_BdFlag);
    
    
    %% Map from RT dof to RT_{-1} dof
    loc_edge2disedge = RT2disRT(loc_elem2Dof,loc_Edge2elem);
    
    %% assembling the matrix (div/jump/redundant dofs)
    D = sparse(loc_Ndof,loc_Ndof);
    
    for kk = 1:3
        % the divergence block
        D = D + sparse(1:loc_NT,loc_elem2Dof(:,kk),1,loc_Ndof,loc_Ndof);
    end
    
    for kk = 1:2
        % the jump block (needs some fix when translated to global)
        D = D + ...
            sparse(loc_NT+1:2*loc_NT,loc_edge2disedge(loc_IdxIntEdge,kk),1,loc_Ndof,loc_Ndof);
    end
    
    % redundant variable boundary block
    D = D + sparse(2*loc_NT+1:3*loc_NT,loc_bdDof,1,loc_Ndof,loc_Ndof);
    %% the null space is gradient perp for div+jump (matrix D)
    % lamgp = rotation of gradient of lambda_{zi}, kernle of discrete Div
    lamgp = zeros(loc_Ndof,1);
    lamgp2Dof = lamgp(loc_elem2Dof);    
    
    % lambda_perp for vertex z_i: local index for z_i is ii at t-th element
    % at t-th element lamb_perp = 1*(local RT basis associated with edge ii-1)
    % +(-1)*(local RT basis associated with edge ii+1)
    for kk = 1:3
        lamgp2Dof(:,c(kk)) = lamgp2Dof(:,c(kk)) ...
            + (repmat(zi,loc_NT,1)==elemZ(:,kk));
        lamgp2Dof(:,c(kk+2)) = lamgp2Dof(:,c(kk+2)) ...
            -(repmat(zi,loc_NT,1)==elemZ(:,kk));
    end
    lamgp(loc_elem2Dof) = lamgp2Dof;
    
    
    %% right hand side 
    R = zeros(loc_Ndof,1);
    R(1:loc_NT) = 1/3*ResElem(idxElemZ,:); % divergence eq rhs
    R(loc_NT+1:2*loc_NT) = (1/2)*JumpEdge(idxIntEdgeZ,:); % edge jump rhs
    
    %% correction for iterative solvers
    switch option.correction
        case 'projection'
            KernelCor = -R'*lamgp/(3*loc_NT)*lamgp;
            % correction to project the R perpendicular to the kernel
            % so that the right hand side to the range of D
        case 'balance'
            % subtract excessive jumps from the divergence
            % weighted by the area and the coefficients
            % ResJumpDiff = sum_{local Edge's} (Jump,phi_z) 
            % - sum_{local T's} (Residual,phi_z) = -Algebraic Residual;
            % ResJumpDiff = -ResAlg(zi,:);
            ResJumpDiff = sum(R(loc_NT+1:2*loc_NT)) - sum(R(1:loc_NT));
            Kmax = max(K(idxElemZ));
            loc_idxKmax = find(K(idxElemZ) == Kmax);
            idxKmax = idxElemZ(loc_idxKmax);
            ACor = accumarray(loc_idxKmax,patchArea(zi)/sum(area(idxKmax)),[loc_NT 1]);
            elemResCor = repmat(ResJumpDiff,loc_NT,1).*area(idxElemZ)/patchArea(zi).*ACor;
            KernelCor = [elemResCor; zeros(2*loc_NT,1)];
        case 'none'
            KernelCor = zeros(3*loc_NT,1);
    end
    R = R + KernelCor(:);  
    
    %% SOLVE (D's rank is Ndof - 1)
    loc_sigD = zeros(3*loc_NT,1);
    loc_sigD(loc_freeDof,1) = D(1:2*loc_NT-1,loc_freeDof)\R(1:2*loc_NT-1);

    
    %% assemble the local RT0_{-1} mass matrix for edges
    loc_M = getmassdisRT(loc_elem2Dof,area(idxElemZ),Dlambda(idxElemZ,:,:),K(idxElemZ));
    
    %% perform the minimization
    projperp = (loc_sigD'*loc_M*lamgp)/(lamgp'*loc_M*lamgp);
    loc_sigD = loc_sigD - projperp*lamgp;
    
    %% compute the local indicator
    etaz(zi) = loc_sigD'*loc_M*loc_sigD/loc_NT;
    
    %% redistribute the local solution into global indices
    sigma2elem = reshape(loc_sigD,3,loc_NT)';
    Rcor2elem = reshape(KernelCor,3,loc_NT)';
    sigCorTmp = zeros(NT,3);
    % map the sigma on this patch to global indices
    RcorTmp = zeros(NT,3);
    % element residual projection error on patch w/ global indices
    sigCorTmp(idxElemZ,:) = sigma2elem(1:loc_NT,:);
    RcorTmp(idxElemZ,:) =  Rcor2elem(1:loc_NT,:);
    
    % compute the globally correction
    sigmaD = sigmaD + sigCorTmp;
    ResCor = ResCor + RcorTmp;
end


for zi = idxBdNode'
    %% Dirichlet Boundary node
    %% local data structures
    loc_NT = valence(zi);
    idxElemZ = find(t2v(:,zi));
    elemZ = elem(idxElemZ,:);
    %     elem2edgeZ = elem2edge(idxElemZ,:);
    
    idxE2vz = find(e2v(:,zi)); % this returns only interior edge (globally defined index)
    idxIntEdgeZ = idxE2vz(~ismember(idxE2vz,idxBdEdge)); % interior edge no Dirichlet edges
    %     DirEdge_ind = idxE2vz(ismember(idxE2vz,idxBdEdge));
    idxAllEdgeZ = unique(elem2edge(idxElemZ,:));
    [~,loc_IdxIntEdge] = ismember(idxIntEdgeZ,idxAllEdgeZ);
    idxBdElem = find(t2v(:,zi) & isBdElem);
    [~,loc_idxBdElem] = ismember(idxBdElem,idxElemZ);
    
    
    %% find the connections
    loc_T = auxstructure(elemZ); 
    loc_Edge2elem = loc_T.edge2elem;
    
    %% Dof corresponds to the same loc_elem2edge but for RT_{-1} for boundary node
    loc_Dof = (1:3*loc_NT)';
    loc_elem2Dof = reshape(loc_Dof,3,loc_NT)'; 
    loc_BdFlag = setboundary(node,elemZ,'Dirichlet'); % local boundary
    Dir_bdFlag = bdFlag(idxElemZ,:); % global Dirichler boundary
    loc_Ndof = size(loc_Dof,1);
    
    [loc_freeDof, loc_DirDof, loc_bdDof] = dofdisRT(zi,elemZ,loc_BdFlag,Dir_bdFlag);
    %% Map from RT dof to RT_{-1} dof
    loc_edge2disedge = RT2disRT(loc_elem2Dof,loc_Edge2elem);
    
    %% assembling the matrix (div/jump/redundant dofs)
    D = sparse(loc_Ndof,loc_Ndof);
    
    for kk = 1:3
        % the divergence block
        D = D + sparse(1:loc_NT,loc_elem2Dof(:,kk),1,loc_Ndof,loc_Ndof);
    end

    for kk = 1:2
        % the jump block for boundary patch,(#int edge)= (#element-1)
        D = D + ...
            sparse(loc_NT+1:2*loc_NT-1,loc_edge2disedge(loc_IdxIntEdge,kk),1,loc_Ndof,loc_Ndof);
    end
    % Dirichlet boundary
    D = D + sparse(2*loc_NT,loc_DirDof,1,loc_Ndof,loc_Ndof);
    
    % redundant variable boundary block
    D = D + sparse(3*loc_NT-size(loc_bdDof,1)+1:3*loc_NT,loc_bdDof,1,loc_Ndof,loc_Ndof);
    %% the null space is gradient perp of lambda_z for div+jump D matrix
    % boundary node z needs including 2 boundary edges
    lamgp = zeros(loc_Ndof,1);
    lamgp2Dof = reshape(lamgp(loc_elem2Dof),loc_NT,3);
    
    for kk = 1:3
        lamgp2Dof(:,c(kk)) = lamgp2Dof(:,c(kk)) ...
            + (repmat(zi,loc_NT,1)==elemZ(:,kk));
        lamgp2Dof(:,c(kk+2)) = lamgp2Dof(:,c(kk+2)) ...
        -(repmat(zi,loc_NT,1)==elemZ(:,kk));
    end
    lamgp(loc_elem2Dof) = lamgp2Dof;
    
    %% right hand side with correction for boundary on the rhs
    R = zeros(loc_Ndof,1);
    R(1:loc_NT) = (1/3)*ResElem(idxElemZ,:);
    R(loc_NT+1:2*loc_NT-1) = (1/2)*JumpEdge(idxIntEdgeZ,:);
    
    %% correction for boundary nodes local problem (same for soln and iters)
    if strcmp(option.DirFix,'yes')
        % compensation term
        ResJumpDiff = sum(R(loc_NT+1:2*loc_NT-1))- sum(R(1:loc_NT));
        
        % for interface
        Kmax = max(K(idxBdElem));
        loc_idxKmax = find(K(idxBdElem) == Kmax);
        idxKmax = idxBdElem(loc_idxKmax);
        ACor = accumarray(loc_idxKmax,sum(area(idxBdElem))./area(idxKmax),size(idxBdElem));       
        DivCor = repmat(ResJumpDiff,size(idxBdElem,1),1).*ACor.*area(idxBdElem)/sum(area(idxBdElem));       
        R(loc_idxBdElem,:) = R(loc_idxBdElem,:) + DivCor;
        R(2*loc_NT) = -ResJumpDiff; % rhs for the Dir Dofs
    end
    
    %% solve a non full rank problem
    loc_sigD = zeros(3*loc_NT,1);
    loc_sigD([loc_freeDof; loc_DirDof(1)],1) = ...
        D(1:2*loc_NT,[loc_freeDof; loc_DirDof(1)])\R(1:2*loc_NT);

    %% assemble the local RT0_{-1} mass matrix for edges
    loc_M = getmassdisRT(loc_elem2Dof,area(idxElemZ),Dlambda(idxElemZ,:,:),K(idxElemZ));
    
    %% perform the minimization
    projperp = (loc_sigD'*loc_M*lamgp)/(lamgp'*loc_M*lamgp);
    loc_sigD = loc_sigD - projperp*lamgp;
    
    %% compute the local indicator
    etaz(zi) = loc_sigD'*loc_M*loc_sigD/loc_NT;
    
    %% redistribute the local solution into global indices
    sigma2elem = reshape(loc_sigD,3,loc_NT)';
    sigCortmp = zeros(NT,3);
    % the sigma coefficient on this patch w/ global indices
    sigCortmp(idxElemZ,:) = sigma2elem(1:loc_NT,:);
    % compute the globally correction
    sigmaD = sigmaD + sigCortmp;
    
end

%% assemble the global RT0_{-1} mass matrix for edges
elem2Dof = reshape(1:3*NT,3,NT)'; % total 3 dof per triangle
M = getmassdisRT(elem2Dof,area,Dlambda,K);

%% total eta
sigmaDtotal = reshape(sigmaD',3*NT,1);
etaTotal = sqrt(sigmaDtotal'*M*sigmaDtotal);

%% local indicator eta_K
eta2elem = etaz(elem);
eta = sqrt(sum(eta2elem,2));

%% direction element-wise computation of element residual (do not count boundary)
etaRes = zeros(NT,1); % residual for the \div sigma + c u = f
divSigmaD = sum(sigmaD,2)./area;
% uh2elem = uh(elem);

if isreal(pde.f) % f is a real number or vector and not a function
    switch length(pde.f)
        case NT  % f is piecewise constant
            f = pde.f;
            etaRes = (divSigmaD - f).*area;
        case N   % f is piecewise linear
            f2elem = pde.f(elem);
            for p = 1:nQuad
                fp = lambda(p,1)*f2elem(:,1)+lambda(p,2)*f2elem(:,2) ...
                    + lambda(p,3)*f2elem(:,3);
                etaRes = etaRes + weight(p)*(divSigmaD - fp).^2.*area;
            end
    end
end    

if ~isempty(pde.f) && ~isreal(pde.f)  % f is a function 
    for p = 1:nQuad
        % quadrature points in the x-y coordinate
        pxy = lambda(p,1)*node(elem(:,1),:) ...
            + lambda(p,2)*node(elem(:,2),:) ...
            + lambda(p,3)*node(elem(:,3),:);
        fp = pde.f(pxy);
        etaRes = etaRes + weight(p)*(divSigmaD - fp).^2.*area;
    end
end

etaRes = sqrt(sum(etaRes,2));
etaResTotal = norm(etaRes);

%% include the correction term into the global estimator
% uh2elem_avg = sum(uh(elem),2)/3;
% RcorTotal = 2*abs(sum(ResCor,2)'*uh2elem_avg)/3;

%%
% etaTotal = sqrt(etaTotal^2 + etaResTotal^2);
% etaTotal = sqrt(etaTotal^2 + etaResTotal^2 + RcorTotal);

%%
esttime = toc;
fprintf('time to estimate the error = %1.2g s\n',esttime);

%% subfunctions section
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subfunctions 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% subfunction get mass matrix for RT_{-1}
    function M = getmassdisRT(elem2Dof,area,Dlambda,K)
        Ndof = 3*size(elem2Dof,1); % total 3 edges per triangle
        M = sparse(Ndof,Ndof);
        csub = [3 1 2 3 1]; % cyclic group indexing of {1,2,3}: c(i)=i-1, c(i+2)=i+1
        for ii = 1:3
            for jj = 1:3
                if ii==jj
                    nini = dot(Dlambda(:,:,ii),Dlambda(:,:,ii),2);
                    njnk = dot(Dlambda(:,:,csub(ii)),Dlambda(:,:,csub(ii+2)),2);
                    Mij = (nini - 3*njnk).*area./K/6;
                else
                    ninj = dot(Dlambda(:,:,ii),Dlambda(:,:,jj),2);
                    ni2 = dot(Dlambda(:,:,ii),Dlambda(:,:,ii),2);
                    nj2 = dot(Dlambda(:,:,jj),Dlambda(:,:,jj),2);
                    Mij = -(ni2 + nj2 + 3*ninj).*area./K/6;
                end
                M = M + sparse(elem2Dof(:,ii),elem2Dof(:,jj),Mij,Ndof,Ndof);
            end
        end
    end

%% subfunction get local Dof for RT_{-1}
    function [freeDof, DirDof, bdDof] = dofdisRT(nodeZ,elemZ,bdFlagZ,DirFlagZ)
        NTz = size(elemZ,1);
        Dof = (1:3*NTz)';
        Elem2Dof = reshape(Dof,3,NTz)';
        Ndof = size(Dof,1);
        if nargin<4; DirFlagZ= zeros(NTz,3);end
        
        % find all edges connecting vertex i
        isDof2V = false(Ndof,1);
        % if vertex i is the 1st first vertex in loc_elem(t,:), then the 2nd
        % and 3rd edges in local elemZ(t,:) must be connecting vertex i
        isDof2V(Elem2Dof(repmat(nodeZ,NTz,1)==elemZ(:,1),[2 3])) = true;
        isDof2V(Elem2Dof(repmat(nodeZ,NTz,1)==elemZ(:,2),[3 1])) = true;
        isDof2V(Elem2Dof(repmat(nodeZ,NTz,1)==elemZ(:,3),[1 2])) = true;
        
        % find local Dirichlet edges connecting vertex i
        isDirDof = false(Ndof,1);
        isDirDof(Elem2Dof(DirFlagZ(:,1) == 1 | DirFlagZ(:,1) == 2,1)) = true;
        isDirDof(Elem2Dof(DirFlagZ(:,2) == 1 | DirFlagZ(:,2) == 2,2)) = true;
        isDirDof(Elem2Dof(DirFlagZ(:,3) == 1 | DirFlagZ(:,3) == 2,3)) = true;
        DirDof = Dof(isDirDof & isDof2V);
        
        % find local interior free edges (RT_{-1} numbering)
        isBdDof = false(Ndof,1);
        isBdDof(Elem2Dof(bdFlagZ(:,1) == 1 | bdFlagZ(:,1) == 2,1)) = true;
        isBdDof(Elem2Dof(bdFlagZ(:,2) == 1 | bdFlagZ(:,2) == 2,2)) = true;
        isBdDof(Elem2Dof(bdFlagZ(:,3) == 1 | bdFlagZ(:,3) == 2,3)) = true;
        isBdDof = isBdDof & (~isDof2V);
        bdDof = Dof(isBdDof,:);
        freeDof = find(~isBdDof & ~isDirDof & isDof2V);
    end

%% subfunction mapping RT to RT_{-1}
    function idxMapEdge = RT2disRT(elem2edge,edge2elem)
        % idxMapEdge(e,1) is the 1st edge in RT_{-1} of the e-th edge in the
        % 1st element of edge2elem
        % idxMapEdge(e,2) is the 2nd edge in RT_{-1} of the e-th edge in the
        % 2nd element of edge2elem
        NRT = size(edge2elem,1);
        idxMapEdge = zeros(NRT,2);
        NdRT = size(elem2edge,1)*3;
        
        idx_edge1 = false(NdRT,3); % logical value of the LOCAL index for each edge in each elem
        idx_edge1(edge2elem(:,3)==1,1)=true;
        idx_edge1(edge2elem(:,3)==2,2)=true;
        idx_edge1(edge2elem(:,3)==3,3)=true;
        
        Dof2elem1 = elem2edge(edge2elem(:,1),:); % the 1st element all DoF edges neighboring
        idxMapEdge(idx_edge1(:,1),1) = Dof2elem1(idx_edge1(:,1),1);
        idxMapEdge(idx_edge1(:,2),1) = Dof2elem1(idx_edge1(:,2),2);
        idxMapEdge(idx_edge1(:,3),1) = Dof2elem1(idx_edge1(:,3),3);
        
        idx_edge2 = false(NdRT,3); % logical value of the LOCAL index for each edge in each elem
        idx_edge2(edge2elem(:,4)==1,1)=true;
        idx_edge2(edge2elem(:,4)==2,2)=true;
        idx_edge2(edge2elem(:,4)==3,3)=true;
        
        Dof2elem2 = elem2edge(edge2elem(:,2),:); % the 2nd element all DoF edges neighboring
        idxMapEdge(idx_edge2(:,1),2) = Dof2elem2(idx_edge2(:,1),1);
        idxMapEdge(idx_edge2(:,2),2) = Dof2elem2(idx_edge2(:,2),2);
        idxMapEdge(idx_edge2(:,3),2) = Dof2elem2(idx_edge2(:,3),3);
    end

end
