function [Samples,LMP] = FBGPs(X, y, varargin)
    % FBGPs_2 Samples from HMC for FBGP Model
    %
    % Syntax:
    %   Samples = FBGPs(X, y)
    %   Samples = FBGPs(X, y, 'Name', Value, ...)
    %
    % Inputs:
    %   X - Training data, T x N matrix
    %   y - Target data, T x 1 vector
    %
    % Name-Value Pair Arguments:
    %   'numSamples' - Number of HMC samples (default: 2000)
    %   'psv'        - Prior of scalar variance (options: 'none', 'constant', 'halfnormal', 'inversegamma'; default: 'halfnormal')
    %   'pnv'        - Prior of noise variance (options: 'none', 'noninfor', 'halfnormal', 'inversegamma'; default: 'halfnormal')
    %   'sparsity' - Sparsity level parameter (default: 1)
    %   'BI'          - Burn-in iterations (default: 3000)
    %   'ini'          - Initial parameters, N x 1 vector (default: log([normrnd(0,1,N,1); std(y); std(y)]))
    %
    % Outputs:
    %   Samples - HMC samples

    %% Input Parsing
    p = inputParser;
    
    % Required inputs
    addRequired(p, 'X', @(x) isnumeric(x) && ismatrix(x));
    addRequired(p, 'y', @(y) isnumeric(y) && isvector(y));
    
    % Optional name-value pairs
    addParameter(p, 'numSamples', 2000, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'psv', 'halfnormal', @(x) ischar(x) && ismember(x, {'none', 'constant', 'halfnormal', 'inversegamma'}));
    addParameter(p, 'pnv', 'halfnormal', @(x) ischar(x) && ismember(x, {'none', 'noninfor', 'halfnormal', 'inversegamma'}));
    addParameter(p, 'sparsity', 1, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'BI', 3000, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'ini', [], @(x) isempty(x) || (isnumeric(x) && isvector(x)));
    
    parse(p, X, y, varargin{:});
    
    numSamples = p.Results.numSamples;
    psv = p.Results.psv;
    pnv = p.Results.pnv;
    sparsity = p.Results.sparsity;
    BI = p.Results.BI;
    ini = p.Results.ini;
    
    %% Set Default Initial Parameters if Not Provided
    if isempty(ini)
        ini = log([abs(normrnd(0,1,size(X,2),1)); std(y); std(y)]);
    end
    
    %% Define Priors
    [psvpdf, dpsvpdf] = definePrior(psv, 'sv');
    [pnvpdf, dpnvpdf] = definePrior(pnv, 'nv');
    
    %% Define the Model Function
    INMP = @(hyp) MPinf(hyp, X, y, psvpdf, dpsvpdf, pnvpdf, dpnvpdf, sparsity);
    
    %% Initialize and Run HMC Sampler
    smpMP = hmcSampler(INMP, ini, 'CheckGradient', true, 'StepSize', 0.06);
    
    %% Draw Samples
    Samples = drawSamples(smpMP, 'NumSamples', numSamples, 'burnin', BI);

    %% Measure Log-Marginal Posterior
    estimate = mean(Samples);
    LMP = INMP(estimate(:));
end

function [pdf, dpdf] = definePrior(priorType, varType)
    % Define prior PDFs and their derivatives based on type and variable
    switch priorType
        case 'none'
            pdf = @(x) 0;
            dpdf = @(x) 0;
        case 'constant'
            pdf = @(x) 0;
            dpdf = @(x) 0;
        case 'halfnormal'
            pdf = @(x) log(2) - 0.5 * log(2 * pi) - 0.5 * (exp(4 * x));
            dpdf = @(x) -2 * exp(4 * x);
        case 'inversegamma'
            if strcmp(varType, 'sv')
                a = 1; b = 0.5;
            elseif strcmp(varType, 'nv')
                a = 3; b = 0.5;
            else
                error('Invalid variable type for inversegamma prior.');
            end
            pdf = @(x) a * log(b) - gammaln(a) - (a + 1) * log(exp(2 * x)) - b / exp(2 * x);
            dpdf = @(x) -2 * (a + 1) + 2 * b / exp(2 * x);
        case 'noninfor'
            if strcmp(varType, 'nv')
                pdf = @(nv) -2 * nv;
                dpdf = @(nv) -2;
            else
                error('Noninformative prior is only defined for noise variance (nv).');
            end
        otherwise
            error('Unknown prior type: %s', priorType);
    end
end
