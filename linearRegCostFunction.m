function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



%Calculate h(x) for a linear function
h_x = X*theta;

%Calculate the regularized cost
J= sum((h_x-y).^2)*(1/(2*m)) + sum(theta.^2)*(lambda/(2*m));

%Calculate gradients
%Multiplying with X is elementwise and not matirx multiplication
grad(1) = sum((h_x-y) .*X(:,1))*(1/m);
grad(2) = sum((h_x-y) .*X(:,2))*(1/m) + theta(2) * lambda/m;








% =========================================================================

grad = grad(:);

end
