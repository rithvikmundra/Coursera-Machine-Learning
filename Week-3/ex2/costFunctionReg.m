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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%theta = theta (2,:);

%theta (1) = 0;
h = 1 ./ (1 + exp(-X * theta));

a = log(h) .* y;
b = log(1 - h) .* (1-y);
numer = sum(-a - b);
j = numer / m ;

reg1 = sum(theta(2:length(theta)) .* theta(2:length(theta))) * (lambda/(2*m));

%reg1 = sum(theta(2:) .* theta(2:)) * (lambda/(2*m));

J= j +reg1 ;


%Gradient
theta (1) = 0;
error = h - y;
c = X' * error;
reg2 = theta * (lambda/m);
grad  = (1/m) .* c + reg2 ;
% =============================================================

end
