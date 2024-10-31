function Kst = exact(hyp, X)
    % EXACT Computes the exact covariance matrix operations for the GP model
    %
    % Syntax:
    %   Kst = exact(hyp, X)
    %
    % Inputs:
    %   hyp - Hyperparameters vector. The last two elements are sv and nv.
    %   X   - Training data matrix (n x D).
    %
    % Outputs:
    %   Kst - Function handle that computes log determinant and other metrics based on W
    %
    % Example:
    %   Kst = exact(hyp, X);
    %
    % Dependencies:
    %   - covScale.m
    %   - solve_chol.m

    %% Input Validation
    validateattributes(hyp, {'numeric'}, {'vector'}, mfilename, 'hyp', 1);
    validateattributes(X, {'numeric'}, {'2d'}, mfilename, 'X', 2);

    %% Compute Scaled Covariance Matrix
    [K, dK] = covScale(hyp, X);
    K = K + 1e-4 * eye(size(X, 1));  % Adding jitter for numerical stability

    %% Define the Exact Computation Function Handle
    Kst = @(W) ldB2_exact(W, K, dK);
end

%% Subfunction: ldB2_exact
function [ldB2, solveKiW, dW, dldB2, L, triB] = ldB2_exact(W, K, dK)
    % LDB2_EXACT Computes log determinant and related metrics for exact covariance
    %
    % Syntax:
    %   [ldB2, solveKiW, dW, dldB2, L, triB] = ldB2_exact(W, K, dK)
    %
    % Inputs:
    %   W  - Weight vector (n x 1).
    %   K  - Covariance matrix (n x n).
    %   dK - Function handle for derivative of covariance matrix.
    %
    % Outputs:
    %   ldB2    - Log determinant of B divided by 2.
    %   solveKiW - Function handle to solve K^{-1}W * r.
    %   dW      - Derivative of log determinant with respect to W.
    %   dldB2   - Function handle for derivative of ldB2.
    %   L       - Cholesky factor of B.
    %   triB    - Trace of B^{-1}.

    n = numel(W);                        
    sW = sqrt(W); 

        % Compute Cholesky decomposition of B = I + S W S K, where S = diag(sW)
    S = diag(sW);
    B = eye(n) + (S * K * S);
    try
        L = chol(B);
    catch ME
        error('exact:ldB2_exact:CholeskyFailed', 'Cholesky decomposition failed: %s', ME.message);
    end
    
    ldB2 = sum(log(diag(L)));  % log(det(B))/2

%     % Compute Cholesky decomposition of B = I + S W S K, where S = diag(sW)
%     B = eye(n) + (sW * sW'.* K);
%     try
%         L = chol(B);
%     catch ME
%         error('exact:ldB2_exact:CholeskyFailed', 'Cholesky decomposition failed: %s', ME.message);
%     end
%     
%     ldB2 = sum(log(diag(L)));  % log(det(B))/2
    
    % Define function handle to solve B \ (S * r)
    solveKiW = @(r) bsxfun(@times,solve_chol(L,bsxfun(@times,r,sW)),sW);

    Q = bsxfun(@times,1./sW,solve_chol(L,diag(sW))); % Q is the inverse of the B which is (I+1/sigma^2K)
    
    dW = sum(Q .* K, 2) / 2;  % d log(det(B))/2 / d W = diag(inv(B) * K) / 2
    triB = trace(Q);          % trace(inv(B))
    
    % Define function handle for derivative of ldB2
    dldB2 = @(alpha) ldB2_deriv_exact(W, dK, Q, alpha);
end

%% Subfunction: ldB2_deriv_exact
function dhyp = ldB2_deriv_exact(W, dK, Q, alpha)
    % LDB2_DERIV_EXACT Computes the derivative of ldB2 with respect to hyperparameters
    %
    % Syntax:
    %   dhyp = ldB2_deriv_exact(W, dK, Q, alpha)
    %
    % Inputs:
    %   W     - Weight vector (n x 1).
    %   dK    - Function handle for derivative of covariance matrix.
    %   Q     - Inverse of B matrix times S.
    %   alpha - Auxiliary vector (n x 1).
    %
    % Outputs:
    %   dhyp - Gradient vector with respect to hyperparameters.
    
    R = alpha * alpha'; 
    % Compute the gradient
    term = bsxfun(@times, Q,W) - R;  % Element-wise multiplication
    dhyp = dK(term) / 2;
end
