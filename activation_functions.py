import numpy as np

def step_function(inputs, weights, bias):
	"""
	Returns 1 if the linear combination of inputs, weights, and bias is positive, and 0 otherwise.

	Parameters
	----------
	inputs : array_like
		The inputs to the function.
	weights : array_like
		The weights of the function.
	bias : float
		The bias of the function.

	Returns
	-------
	int
		1 if the linear combination of inputs, weights, and bias is positive, and 0 otherwise.
	"""
	input_value = linear_combination(inputs, weights, bias)

	if input_value > 0:
		return 1
	else:
		return 0

def step_derivative(x):
	"""
	Returns 0 if x is 0, and raises a ValueError otherwise.

	Parameters
	----------
	x : float
		The input to the function.

	Returns
	-------
	float
		0 if x is 0, and raises a ValueError otherwise.
	"""
	if x==0:
		raise ValueError("Step function non-differentiable at 0")
	else:
		return 0

def linear_function(inputs, weights, bias, m=1):
	"""
	Returns the linear combination of inputs, weights, and bias multiplied by m.

	Parameters
	----------
	inputs : array_like
		The inputs to the function.
	weights : array_like
		The weights of the function.
	bias : float
		The bias of the function.
	m : float
		The multiplier of the function.

	Returns
	-------
	float
		The linear combination of inputs, weights, and bias multiplied by m.
	"""
	input_value = linear_combination(inputs, weights, bias)

	return m*input_value

def linear_derivative(x, m=1):
	"""
	Returns m if x is not 0, and raises a ValueError otherwise.

	Parameters
	----------
	x : float
		The input to the function.
	m : float
		The multiplier of the function.

	Returns
	-------
	float
		m if x is not 0, and raises a ValueError otherwise.
	"""
	if x==0:
		raise ValueError("Linear function non-differentiable at 0")
	else:
		return m

def sigmoid_function(inputs, weights, bias):
	"""
	Computes the sigmoid function for a given set of inputs, weights, and bias.

	Parameters
	----------
	inputs : array_like
	    The inputs to the sigmoid function.
	weights : array_like
	    The weights of the sigmoid function.
	bias : float
	    The bias of the sigmoid function.

	Returns
	-------
	output : float
	    The output of the sigmoid function.

	"""

	input_value = linear_combination(inputs, weights, bias)

	output = 1/(1 + np.e**(-input_value))

	return output

def sigmoid_derivative(x):
	"""
	Computes the derivative of the sigmoid function for a given input.

	Parameters
	----------
	x : float
	    The input to the sigmoid function.

	Returns
	-------
	derivative_output : float
	    The derivative of the sigmoid function.

	"""

	derivative_output = np.e**(-x)/(1 + np.e**(-x))**2

	return derivative_output

def softmax_function(inputs, weights):
	"""
	Computes the softmax function for a given set of inputs and weights.

	Parameters
	----------
	inputs : array_like
	    The inputs to the softmax function.
	weights : array_like
	    The weights of the softmax function.

	Returns
	-------
	output : array_like
	    The output of the softmax function.
	"""

	weighted_values = np.multiply(inputs, weights)

	output = (np.e*weighted_values)/sum(weighted_values)

	return output # should output a vector

def tanh_function(inputs, weights, bias):
	"""
	Computes the tanh function for a given set of inputs, weights, and bias.

	Parameters
	----------
	inputs : array_like
	    The inputs to the tanh function.
	weights : array_like
	    The weights of the tanh function.
	bias : float
	    The bias of the tanh function.

	Returns
	-------
	output : float
	    The output of the tanh function.
    """

	input_value = linear_combination(inputs, weights, bias)

	output = (np.e**input_value - np.e**(-input_value))/\
	(np.e**input_value + np.e**(-input_value))

	return output

def tanh_derivative(x):
	"""
	Computes the derivative of the tanh function for a given input.

	Parameters
	----------
	x : float
	    The input to the tanh function.

	Returns
	-------
	derivative_output : float
	    The derivative of the tanh function.
    """

	tanh_func = (np.e**x - np.e**(-x))/\
	(np.e**x + np.e**(-x))

	derivative_output = 1 - tanh_func**2

	return derivative_output

def ReLU_function(inputs, weights, bias, leak_constant=0):
	"""
	Computes the rectified linear unit function for given inputs, weights, and bias.

	Parameters
	----------
	inputs : array_like
		The inputs to the ReLU function.
	weights : array_like
		The weights of the ReLU function.
	bias : array_like
		The bias of the ReLU function.
	leak_constant : float, optional
		The leak constant of the ReLU function. The default is 0.

	Returns
	-------
	numpy.ndarray
		The output of the ReLU function.
	"""

	input_value = linear_combination(inputs, weights, bias)

	if input_value < 0:
		return leak_constant*input_value
	else:
		return input_value

def ReLU_derivative(x, leak_constant=0):
	"""
	Computes the derivative of the rectified linear unit function for given input.

	Parameters
	----------
	x : float
		The input to the derivative of the ReLU function.
	leak_constant : float, optional
		The leak constant of the ReLU function. The default is 0.

	Returns
	-------
	float
		The output of the derivative of the ReLU function.
	"""

	if x < 0:
		return leak_constant
	else:
		return 1

def linear_combination(inputs, weights, bias):
	"""
	Computes the linear combination of inputs, weights, and bias.

	Parameters
	----------
	inputs : array_like
		The inputs to the linear combination.
	weights : array_like
		The weights of the linear combination.
	bias : array_like
		The bias of the linear combination.

	Returns
	-------
	numpy.ndarray
		The output of the linear combination.
	"""

	return np.dot(inputs, weights) + bias