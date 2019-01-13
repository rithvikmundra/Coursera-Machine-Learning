function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    h = X * theta;

    error = h - y;

    %error_1 = sum(error)

    %theta[1] = theta[1] - ((alpha/m) * error_1 ))

    %error_2 = X[:,2]' * error
    %error_2 = sum (error_2)

    %theta[2] = theta[2] - ((alpha/m) * error_2 ))

    error_3 = X' * error;
    theta_diff = (alpha/m) * error_3;
    theta = theta - theta_diff;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
