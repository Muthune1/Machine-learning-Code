function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

X = [ones(m, 1) X];

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
%Calculate 1st layer output

a1 = X*Theta1';
z2 = sigmoid(a1); % Compute first layer
n= size(z2,1);    %Get row size
z2 = [ones(n, 1) z2]; %Pad ones for a0
a2= Theta2*z2';
z3=sigmoid(a2);
[q,p] = max(z3',[],2); 
%Calcualte h(x) , p gives position corresponding  to max value which denotes the predicted label
%q gives the maximum value








% =========================================================================


end
