import first_best as fb
import second_best as sb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
==============================================================================================
Main script to approximate and visualize the first-best and second-best
==============================================================================================
"""

show_diagnostics=False

def plot_contour(t, y, data, title, colorbar_label, line_y=None, line_label=None):
    plt.figure(figsize=(8, 6))
    plt.contourf(t, y, data, levels=50, cmap='viridis')
    if line_y is not None:
        plt.axhline(y=line_y, color='red', linestyle='--', linewidth=1, label=line_label)
        plt.legend()
    plt.colorbar(label=colorbar_label)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title(title)
    plt.show()

# Parameters
parameters = [
[1,0.3,1,1,1,0.3],          #0...retirement only in second best
[1,0.1,1,1,1,0.3],          #1...no retirement
[0.3,0.2,0.8,1,1,0.2],      #2...no retirement
[0.3,0.3,0.8,1,10,0.2],     #3...retirement in both
[0.3,0.3,0.8,1,5,0.3],      #4...retirement in both
[0.5,0.7,0.8,1,5,0.5]       #5...Second best is only retirement
]

"""
==============================================================================================
TODO: there's one more scenario to consider: always retire in second-best
==============================================================================================
"""


gamma, kappa , k, T, M, sigma =parameters[5]   #choose scenario
print(f"Parameters: gamma={gamma}, kappa={kappa}, k={k}, T={T}, M={M}, sigma={sigma}")

retire = (1 - kappa) / gamma
does_retire = True if retire < (1 - np.exp(-gamma * M)) / gamma else False


# First Best 
# ====================================================================================================================

t_grid_fb, y_grid_fb, u_values_fb, y_alpha_equal, t_alpha_equal = fb.first_best(gamma, kappa, k, T,M)

# 2D plot
if does_retire:
    plot_contour(
        t_grid_fb, y_grid_fb, u_values_fb,
        title='First best', colorbar_label='u(t, y)',
        line_y=retire, line_label=r'$y = (1 - \kappa)/\gamma$'
    )
else:
    plt.figure(figsize=(8, 6))
    plt.contourf(t_grid_fb, y_grid_fb, u_values_fb, levels=50, cmap='viridis')
    plt.plot(t_alpha_equal, y_alpha_equal, color='red', linewidth=0.5, label=r'$\alpha^* = \alpha^M$')
    plt.colorbar(label='u(t, y)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('First best')
    plt.legend()
    plt.show()



# 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(t_grid_fb, y_grid_fb, u_values_fb, cmap='viridis', alpha=0.8)
fig.colorbar(surf, ax=ax, label='u(t, y)')


ax.set_xlabel('t')
ax.set_ylabel('y')
ax.set_zlabel('u(t, y)')
ax.set_title('3D First best')
plt.show()




    # Second Best 
# ====================================================================================================================
t_grid_sb, y_grid_sb, u_values_sb, max_term_values, dy_u_values, dyy_u_values, z_values = sb.second_best(gamma, kappa, k, T, M, sigma)


#2D plot
plot_contour(
    t_grid_sb, y_grid_sb, u_values_sb,
    title='Second best', colorbar_label='u(t, y)'
)

# 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(t_grid_sb, y_grid_sb, u_values_sb, cmap='viridis')
ax.set_xlabel('t')
ax.set_ylabel('y')
ax.set_zlabel('u(t, y)')
ax.set_title('3D Second best')
plt.show()

if show_diagnostics:
    # Hamiltonian
    plot_contour(
        t_grid_sb, y_grid_sb, max_term_values,
        title='Heatmap of the Hamiltonian', colorbar_label='max(0, max_term)'
    )

    # dy_u
    plot_contour(
        t_grid_sb, y_grid_sb, dy_u_values,
        title='Heatmap of dy_u', colorbar_label=r'$\partial u / \partial y$'
    )

    # dyy_u
    plot_contour(
        t_grid_sb, y_grid_sb, dyy_u_values,
        title='Heatmap of dyy_u', colorbar_label=r'$\partial^2 u / \partial y^2$'
    )

    # Plot z^*
    plot_contour(
        t_grid_sb, y_grid_sb, z_values,
        title='Heatmap of z*', colorbar_label='z*',
        line_y=retire if does_retire else None,
        line_label=r'$y = (1 - \kappa)/\gamma$' if does_retire else None
    )
