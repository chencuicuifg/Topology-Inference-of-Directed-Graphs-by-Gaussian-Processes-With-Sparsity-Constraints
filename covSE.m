function [K, dK, D2] = covSE(hyper, X)
    % COVSE Squared Exponential Covariance Function with Unit Amplitude
    %
    % Syntax:
    %   [K, dK, D2] = covSE(hyper, X)
    %
    % Inputs:
    %   hyper - Hyperparameters vector. Each element corresponds to a length-scale parameter for each dimension.
    %           For a D-dimensional input space, hyper should be a D x 1 vector.
    %   X     - Training data matrix (n x D), where n is the number of data points and D is the dimensionality.
    %
    % Outputs:
    %   K  - Squared Exponential covariance matrix (n x n).
    %   dK - Function handle for computing the derivative of the covariance matrix with respect to hyperparameters.
    %   D2 - Matrix of squared Euclidean distances between data points (n x n).
    %
    % Example:
    %   hyper = log([1; 1]); % Example hyperparameters for 2-dimensional data
    %   X = randn(5, 2);      % Example data
    %   [K, dK, D2] = covSE(hyper, X);
    %
    % Dependencies:
    %   - covMaha.m : Mahalanobis distance covariance function
    %
    % See also:
    %   covMaha

    %% Extract Hyperparameters
    D = length(hyper);  % Number of dimensions
    if size(X, 2) ~= D
        error('covSE:DimensionMismatch', 'Number of hyperparameters must match the number of columns in X.');
    end

    %% Define Squared Exponential Covariance and Its Derivative
    k = @(d2) exp(-d2 / 2);                % Squared Exponential covariance function
    dk = @(d2, K) (-0.5) * K;              % Derivative of covariance with respect to d2

    %% Compute Covariance Matrix and Derivative
    [K, dK, D2] = covMaha(k, dk, hyper, X);
end
