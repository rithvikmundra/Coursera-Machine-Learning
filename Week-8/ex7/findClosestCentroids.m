function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%


%tried other way but was getting 
% Submission failed: Index in position 1 exceeds array bounds (must not exceed 5).

%{
for i = 1:length(X)
    dist_min = Inf;
    for j = 1:length(centroids)
        %dist= norm(X(i,:) - centroids(j,:));
        dist = sum((X(i,:) - centroids(j,:)).^2);
        if dist < dist_min
           dist_min = dist
           idx(i,1) = j
        end
    end
end

%}

for i = 1:length(X)
    dist = zeros(1,length(K))
    for j = 1:K
        %dist(1,j)= norm(X(i,:) - centroids(j,:));
        dist(1,j) = sqrt(sum(power((X(i,:) - centroids(j,:)),2)))
    end
    [d, index] = min(dist)
    idx(i,1) = index
end



% =============================================================

end

