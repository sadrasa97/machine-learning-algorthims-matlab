function [z, mu, sigma] = zscore(x)

    if ~isnumeric(x)
        error('Input must be numeric.');
    end

    mu = mean(x, 1);

    sigma = std(x, 0, 1); % "0" uses the normalization by (N-1).

    sigma(sigma == 0) = eps;

    z = (x - mu) ./ sigma;
end

