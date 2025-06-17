import numpy as np


def spline_cubic(x, xdata, ydata):
	"""
	Computes cubic spline interpolation at given x values,
	using function values and finite-difference derivatives at xdata.
	"""
	m = compute_derivatives(xdata, ydata)
	y = np.array([], dtype=float)

	for i in range(1, xdata.size):
		x1 = xdata[i-1]
		x2 = xdata[i]
		y1 = ydata[i-1]
		y2 = ydata[i]
		m1 = m[i-1]
		m2 = m[i]

		# get interpolation values in this range:
		if i < x.size - 1:
			mask = (x1 <= x) & (x < x2)
		else:
			mask = (x1 <= x) & (x <= x2)
		x_in_range = x[mask]

		# interpolate this range:
		y_in_range = p(x_in_range, x1, x2, 
						y1, y2, m1, m2)
		y = np.append(y, y_in_range)

	return y


def p(x, x1, x2, y1, y2, m1, m2):
	"""
	The interpolation function over [x1, x2]

	Parameters:
		x [array[float]]: x-coord of interpolated values
		x1 [float]: start of interval
		x2 [float]: end of interval
		y1 [float]: f(x1)
		m1 [float]: f'(x1)
		m2 [float]: f'(x2)
	"""
	coeffs = solve_coeffs(x1, x2, y1, y2, m1, m2)
	power = np.arange(coeffs.size)
	reshaped_x_diff = (x-x1)[:, np.newaxis]  # reshape for broadcasting
	y_interp = np.sum(coeffs * (reshaped_x_diff)**power, axis=1)
	return y_interp


def solve_coeffs(x1, x2, y1, y2, m1, m2):
	"""
	Solves for coefficients of cubic equation over [x1, x2]
	based on the constraints (where p is the polynomial 
	for interpolation):
		1. p(x1) = f(x1)
		2. p(x2) = f(x2)
		3. p'(x1) = f'(x1)
		4. p'(x2) = f'(x2)

	Parameters:
		x1 [float]: start of interval
		x2 [float]: end of interval
		y1 [float]: f(x1)
		y2 [float]: f(x2)
		m1 [float]: f'(x1)
		m2 [float]: f'(x2)

	Returns:
		coeffs [array[float]]: [a, b, c, d]
		where p(x) = a + bx + cx^2 + dx^3
	"""
	# if Ax = b and x is our solution:
	dx = x2 - x1
	A = np.array([[1, 0, 0, 0],
				  [1, dx, dx**2, dx**3],
				  [0, 1, 0, 0],
				  [0, 1, 2*dx, 3*dx**2]])
	b = np.array([y1, y2, m1, m2])

	# solve:
	coeffs = np.linalg.solve(A, b)
	return coeffs


def compute_derivatives(x, y):
	"""
	Computes the numerical derivatives at each point using finite differences
	Uses central differences except first and last index.

	Parameters:
		x [array[float]]: size n
		y [array[float]]: size n
	
	Returns:
		m [array[float]]: size n - array of numerical derivatives
	"""
	m = np.empty_like(y)

	for i in range(m.size):
		if i == 0:
			m[i] = (y[i+1]-y[i])/(x[i+1]-x[i])  # forward difference
		elif i == n - 1:
			m[i] = (y[i]-y[i-1])/(x[i]-x[i-1])  # backward difference
		else:
			m[i] = (y[i+1]-y[i-1])/(x[i+1]-x[i-1])  # central difference

	return m