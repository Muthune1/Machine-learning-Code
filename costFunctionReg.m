function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
z=zeros(m);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i = 1:m
  for j= 1: size(theta)
   z(i)=z(i)+ theta(j)*X(i,j);       %Calculate z value
  end 
  J= J+(-y(i) * log(sigmoid(z(i))))/m -(1-y(i))* log(1-sigmoid(z(i)))/m; %Calculate first part of J
end
for i = 1: size(theta)
    J= J + (theta(i)*theta(i))*lambda/(2*m); %Calculate and add second part of J
end

%Calculate first gradient
  
    for i=1:m
    grad(1) = grad(1) + (sigmoid(z(i)) - y(i))*X(i,1)/m;
  end

  %Calculate other gradients
  for j=2:size(theta)
    for i=1:m
    grad(j) = grad(j) + (sigmoid(z(i)) - y(i))*X(i,j)/m;
    end
  grad(j) = grad(j)+lambda*(theta(j))/m;           %Add regularization factor
  
 



% =============================================================

end
