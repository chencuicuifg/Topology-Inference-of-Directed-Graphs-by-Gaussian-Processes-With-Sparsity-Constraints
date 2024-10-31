function [lZ, dlZ] = MPinf(hyp, X, y, psvpdf, dpsvpdf, pnvpdf, dpnvpdf, c)
    % MPINF Computes the log marginal likelihood and its gradient
    %
    % Syntax:
    %   [lZ, dlZ] = MPinf(hyp, X, y, psvpdf, dpsvpdf, pnvpdf, dpnvpdf, c)
    %
    % Inputs:
    %   hyp      - Hyperparameters vector. The last two elements are sv and nv.
    %   X        - Training data matrix (n x D).
    %   y        - Target vector (n x 1).
    %   psvpdf   - Handle to the prior PDF function for scalar variance.
    %   dpsvpdf  - Handle to the derivative of the prior PDF for scalar variance.
    %   pnvpdf   - Handle to the prior PDF function for noise variance.
    %   dpnvpdf  - Handle to the derivative of the prior PDF for noise variance.
    %   c        - Sparsity level parameter (scalar).
    %
    % Outputs:
    %   lZ  -  Log marginal likelihood (scalar).
    %   dlZ - Gradient of the Log marginal likelihood (vector).
    
    %% Input Validation
    validateattributes(hyp, {'numeric'}, {'vector'}, mfilename, 'hyp', 1);
    validateattributes(X, {'numeric'}, {'2d'}, mfilename, 'X', 2);
    validateattributes(y, {'numeric'}, {'vector'}, mfilename, 'y', 3);
    validateattributes(psvpdf, {'function_handle'}, {}, mfilename, 'psvpdf', 4);
    validateattributes(dpsvpdf, {'function_handle'}, {}, mfilename, 'dpsvpdf', 5);
    validateattributes(pnvpdf, {'function_handle'}, {}, mfilename, 'pnvpdf', 6);
    validateattributes(dpnvpdf, {'function_handle'}, {}, mfilename, 'dpnvpdf', 7);
    validateattributes(c, {'numeric'}, {'scalar'}, mfilename, 'c', 8);
    
    %% Extract Hyperparameters
    D = size(X, 2);  % Number of features
    n = size(X, 1);  % Number of data points
    
    % Ensure hyp has at least D + 2 elements
    if length(hyp) < D + 2
        error('The hyperparameter vector "hyp" must have at least D + 2 elements.');
    end
    
    % Extract scalar variance (sv) and noise variance (nv)
    sv = hyp(end-1);
    nv = hyp(end);
    
    %% Define Parameters
    a = c * ones(D, 1);  % Sparsity parameters
    sn2 = exp(2 * nv);    % Noise variance
    W = ones(n, 1) / sn2; % Noise precision (inverse variance)
    
    %% Covariance Approximation
    Kst = exact(hyp(1:end-1), X);  % Covariance approximation function handle
    
    % Compute necessary components from covariance approximation
    [ldB2, solveKiW, dW, dhyp_cov] = Kst(W);
    
    %% Compute Alpha
    alpha = solveKiW(y);
    
    %% Prior Probabilities
    ppdf = -sum(a .* exp(hyp(1:D))) + psvpdf(sv) + pnvpdf(nv);
    dppdf = [-a .* exp(hyp(1:D)); dpsvpdf(sv); dpnvpdf(nv)];
    
    %% Log Marginal Likelihood
    lZ = -((y' * alpha) / 2 + ldB2 + n * log(2 * pi * sn2) / 2) + ppdf;
    
    %% Gradient of Log Marginal Likelihood
    dlZ_cov = dhyp_cov(alpha); 
    dlZ_n = -sn2 * (alpha' * alpha) - 2 * sum(dW) / sn2 + n;
    dlZ = -[dlZ_cov; dlZ_n] + dppdf;
end
