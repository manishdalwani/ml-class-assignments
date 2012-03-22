function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

result = sigmoid(X*theta);
J = (-1 * (sum((y .* log(result)) + ((1-y) .* log(1-result)))/m)) + sum(theta(2:size(theta),1) .^ 2)*(lambda/(2*m));

grad = ((X' * (result - y))+(lambda * theta))/m;
grad(1) = grad(1) - ((lambda * theta(1))/m); 

end
