function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% first computing h(x) using NN algorithm
X = [ones(m,1) X];   %5000*401 dimension
Z1 = X * Theta1'; %5000*25 dimension
A2 = sigmoid(Z1); %5000*25 dimension
A2 = [ones(m,1) A2]; %5000*26 dimension

Z2 = A2 * Theta2';  %5000 * 10 dimension
H = sigmoid(Z2);    %5000 * 10 dimension


y_new=zeros(m,num_labels);
for i = 1:m,
    %b=zeros(1,10);
    %b(1,y(i))=1;
    y_new(i,y(i))= 1; 
end


%Mine which works in Local matlab perfectly

%y_new=zeros(m,num_labels);
%for i = 1:m,
    %b=zeros(10,1);
    %b(y(i))=1;
%    y_new(i,:))= b; 
%end


J = (1/m)*sum(sum(-y_new .* log(H) - ((1-y_new) .* log(1 - H)))) ;

% Cost Regularization 

Theta1_new = Theta1(:,2:end);
Theta2_new = Theta2(:,2:end);

J = J + (lambda/(2*m)) * (sum(sum(Theta1_new .* Theta1_new)) + sum(sum(Theta2_new .* Theta2_new)))


%part 2 - backpropagation


for t = 1: m
    a1 = X(t,:); %X has the bias line from above (1 * 401)
    z2 = a1 * Theta1';  % (1* 401) * (401*25)
    a2 = sigmoid(z2); % (1 * 25)
    a2 = [ 1 a2];  %(1 * 26)
    z3 = a2 * Theta2'; %(1 * 26) * (26*10) = (1 *10)
    a3 = sigmoid(z3);
    
    yk = ([1:num_labels]==y(t));

    d3 = a3 - yk; %calculating Delta3 (1 *10)

    d2 = ((Theta2)' * d3') .* sigmoidGradient([1 ;z2']) ;  % (26 *10) * (10*1)

    d2 = d2(2:end); %removing 1st column of D2

    Theta2_grad = Theta2_grad + ( d3' * a2) ; %(26*1) * (1 * 10)= (26 *10)

    Theta1_grad = Theta1_grad + (d2 * a1);  %(25*1) (1 * 401) = (25 * 401)
end
    
Theta2_grad = Theta2_grad/m;

Theta1_grad = Theta1_grad/m;


%part3 Regularized cost

%for J>=1

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda/m)* Theta1(:,2:end));  
 

Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda/m)* Theta2(:,2:end));




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
