function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

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


X = [ones(m, 1) X];

%a2 = sigmoid(Theta1 * X');

%ones_row =ones(1,length(a2));
%a2 = [ones_row; a2];
%h = sigmoid(Theta2 * a2);

% both are fine for gettuing marks... but the above solution I implemented
% is not letting the final prediction to pop up images top check my
% asnwers

a2 = sigmoid(X * Theta1');

a2 = [ones(m,1) a2];

h = sigmoid( a2 * Theta2');


[j,p] = max(h,[],2);


% =========================================================================


end
