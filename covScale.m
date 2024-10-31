function [K, dK] = covScale(hyp, X)
    % COVSCALE Computes the scaled covariance matrix and its derivative
    %
    % Syntax:
    %   [K, dK] = covScale(hyp, X)
    %
    % Inputs:
    %   hyp - Hyperparameters vector. The last element is the log scale factor (lsf).
    %         The first (n-1) elements are hyperparameters for the base covariance function.
    %   X   - Training data matrix (n x D), where n is the number of data points and D is the dimensionality.
    %
    % Outputs:
    %   K  - Scaled covariance matrix (n x n).
    %   dK - Function handle for computing the derivative of the covariance matrix with respect to hyperparameters.
    %
    % Example:
    %   hyp = [log(1.5); log(0.5); log(2)]; % Example hyperparameters
    %   X = randn(10, 3); % Example data
    %   [K, dK] = covScale(hyp, X);
    %
    % Dependencies:
    %   - covSE.m : Squared Exponential covariance function
    %
    % See also:
    %   covSE

    %% Input Validation
    validateattributes(hyp, {'numeric'}, {'vector', 'nonempty'}, mfilename, 'hyp', 1);
    validateattributes(X, {'numeric'}, {'2d', 'nonempty'}, mfilename, 'X', 2);

    %% Extract Hyperparameters
    nHyp = length(hyp);
    if nHyp < 2
        error('covScale:InsufficientHyperparameters', 'The hyperparameter vector "hyp" must contain at least two elements.');
    end

    lsf = hyp(end);          % Log scale factor
    hyp_cov = hyp(1:end-1);  % Hyperparameters for the base covariance function
    hyp_cov = hyp_cov(:);    % Ensure column vector

    %% Compute Base Covariance and Its Derivative
    [K0, dK0] = covSE(hyp_cov, X);

    %% Compute Scale Factor and Scaled Covariance
    sfx = exp(lsf);        % Scale factor
    S = sfx^2;             % Scaling (since S = sfx * sfx)

    K = S .* K0;           % Scaled covariance matrix

    %% Define Derivative Function Handle
    dK = @(Q) dirder(Q, S, K0, dK0, sfx);

    %% Local Function: dirder
    function dhyp = dirder(Q, S, K0, dK0, sfx)
        % DIRDER Computes the derivative of the scaled covariance matrix
        %
        % Syntax:
        %   dhyp = dirder(Q, S, K0, dK0, sfx)
        %
        % Inputs:
        %   Q    - Matrix involved in the derivative computation (n x n).
        %   S    - Scaling factor (scalar).
        %   K0   - Base covariance matrix (n x n).
        %   dK0  - Function handle for the derivative of the base covariance matrix.
        %   sfx  - Scale factor (scalar).
        %
        % Outputs:
        %   dhyp - Gradient vector with respect to hyperparameters (nHyp x 1).

        % Compute derivative with respect to base hyperparameters
        dhyp0 = dK0(Q .* S);

        % Compute derivative with respect to the scale factor
        Q_scaled = Q .* K0;
        qx = sum(Q_scaled * sfx, 2);
        dhyp_scale = 2 * sfx * sum(qx); % 2 * dsfx(qx), where dsfx(qx) = sfx * sum(qx)

        % Combine derivatives
        dhyp = [dhyp0; dhyp_scale];
    end
end
