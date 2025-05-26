import numpy as np


def second_best(gamma, kappa, k, T, M, sigma):
    """
    ==========================================================================================
    Solves the second-best control problem using a finite difference method (explicit scheme)

    Parameters: gamma, kappa, k, T, M, sigma : float 

    Returns:
        t_grid, y_grid : 2D arrays (meshgrid for t and y)
        u_values : 2D array (value function)
        max_term_values : 2D array (Hamiltonian)
        dy_u_values : 2D array (dirst derivative du/dy)
        dyy_u_values : 2D array (second derivative d^2u/dy^2)
        z_values : 2D array (optimal control z*)
    ==========================================================================================
    """
    Y_max = (1 - np.exp(-gamma * M)) / gamma 
    N_t = max(50, int(T*3000)) 
    N_y = max(10, int(Y_max * 180))  
    dt = T / N_t
    dy = Y_max / N_y
    z_max = kappa+ 5

    # grid
    u_values = np.zeros((N_t+1, N_y+1),dtype=np.float64)
    max_term_values = np.zeros((N_t+1, N_y+1),dtype=np.float64)  
    dy_u_values = np.zeros((N_t+1, N_y+1),dtype=np.float64)  
    dyy_u_values = np.zeros((N_t+1, N_y+1),dtype=np.float64) 
    z_values = np.zeros((N_t+1, N_y+1),dtype=np.float64) 
    y_values = np.linspace(0, Y_max, N_y+1)
    t_values = np.linspace(0, T, N_t+1)
    t_grid, y_grid = np.meshgrid(t_values, y_values, indexing='ij')

    # boundary conditions
    u_values[-1, :] = (1/gamma) * np.log(1 - gamma * y_values)
    u_values[:, 0] = 0   
    u_values[:, -1] = -M 

    # discretization
    for n in range(N_t-1, -1, -1):
        for j in range(1, N_y):
            y = y_values[j]
            
            # derivatives
            dyy_u = (u_values[n+1, j+1] - 2*u_values[n+1, j] + u_values[n+1, j-1]) / dy**2
            dy_u = (u_values[n+1, j+1] - u_values[n+1, j-1]) / (2 * dy)
            if dyy_u>=0 : 
                dyy_u=0   #For numerical stability
                dy_u_values[n, j] = dy_u
                dyy_u_values[n, j] = dyy_u
                u_values[n, j] = u_values[n+1, j]
            else :
                dy_u_values[n, j] = dy_u
                dyy_u_values[n, j] = dyy_u
                
                # Hamiltonian
                z_val = np.linspace(kappa, z_max, 1000)
                A = ((sigma**2 / 2) * dyy_u + (1 / (2 * k)) * dy_u)
                B = (kappa**2 / (2 * k)) * dy_u + kappa / k
                max_values = z_val**2 * A + z_val / k - B

                max_index = np.argmax(max_values)
                if max_values[max_index]>0: 
                    z_values[n, j] = z_val[max_index]
                    max_term_values[n, j] = max_values[max_index]

                u_values[n, j] = u_values[n+1, j] + dt * max(0, max_values[max_index])

    return t_grid, y_grid, u_values, max_term_values, dy_u_values, dyy_u_values, z_values


