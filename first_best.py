import numpy as np
import matplotlib.pyplot as plt



def first_best(gamma, kappa,k,T,M):
    """
    =======================================================================================
    Computes the first-best solution of the control problem using an explicit formula

    Parameters: gamma, kappa, k, T, M : float 
    
    Returns:
        t_grid, y_grid : 2D arrays (meshgrid of t and y)
        u_values : 2D array (solution values on the grid)
        y_alpha_equal, t_alpha_equal : list of floats (coordinates where alpha* = alpha^M)
    =======================================================================================
    """

    def U_A(x):
        return (1 - np.exp(-gamma * x)) / gamma


    def alpha_star(t, y):
        numerator = -(k + kappa * gamma * (T - t)) + np.sqrt((k + kappa * gamma * (T - t))**2 - 2 * k * gamma * (T - t) * (gamma * y - 1 + kappa))
        denominator = gamma * (T - t) * k
        return numerator / denominator

    def alpha_M(t, y):
        numerator = -kappa * (T - t) + np.sqrt(kappa**2 * (T - t)**2 - 2 * k * (T - t) * (y - U_A(M)))
        denominator = k * (T - t)
        return numerator / denominator


    def F(t, y, alpha):
        if t<T:
            term1 = alpha * (T - t)
            term2 = (1 / gamma) * np.log(1 - gamma * y - gamma * (T - t) * (k / 2 * alpha**2 + kappa * alpha))
            return term1 + term2
        else:
            return 1/gamma*np.log(1-gamma*y)

    def u(t, y):
        if y <= U_A(M) and y <= (1 - kappa) / gamma:
            return F(t, y, min(alpha_star(t, y), alpha_M(t, y)))
        elif y <= U_A(M) and y > (1 - kappa) / gamma:
            return F(t, y, 0)
        else:
            return -np.inf


    t_vals = np.linspace(0, T-10**(-5), int(T * 500))
    y_vals = np.linspace(0, U_A(M), max(50,int(U_A(M) * 1000)))
    t_grid, y_grid = np.meshgrid(t_vals, y_vals)
    u_values = np.vectorize(u)(t_grid, y_grid)

    # Find alpha_star = alpha_M 
    y_alpha_equal = []
    t_alpha_equal = []

    for t in t_vals:

        for y in y_vals:
            if abs(alpha_star(t, y) - alpha_M(t, y)) < 1e-3:
                y_alpha_equal.append(y)
                t_alpha_equal.append(t)
                break
                
    t_alpha_equal = np.array(t_alpha_equal)
    y_alpha_equal = np.array(y_alpha_equal)


    return t_grid, y_grid, u_values, y_alpha_equal, t_alpha_equal
