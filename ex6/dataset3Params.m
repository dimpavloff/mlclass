function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vec = sigma_vec = [0.1 0.3 1 3 ];
error_val = zeros(length(C_vec),length(sigma_vec));

for i = 1:length(C_vec)
    for j = 1:length(sigma_vec)
        C = C_vec(i);
        sigma = sigma_vec(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        pred = svmPredict(model, Xval);
        error_val(i,j) = mean(double(pred ~= yval));
    end
end
[val min_row] = min(error_val);
[val min_ind] = min(val);
C = C_vec(min_row(min_ind));
sigma = sigma_vec(min_ind);

fprintf('optimal values C = %f, sigma = %f',C,sigma);

% =========================================================================

end
