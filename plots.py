import numpy as np
import matplotlib.pyplot as plt

def plot_comparison(S, t, V_pinn, V_theoretical, save_dir):
    """Plot the comparison between PINN and theoretical solutions."""
    S = S.reshape(-1, int(np.sqrt(len(S))))
    t = t.reshape(-1, int(np.sqrt(len(t))))
    V_pinn = V_pinn.reshape(S.shape)
    V_theoretical = V_theoretical.reshape(S.shape)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": "3d"})
    axs[0].plot_surface(S, t, V_pinn, cmap='plasma')
    axs[0].set_title("PINN Solution")
    axs[0].set_xlabel("Stock Price (S)")
    axs[0].set_ylabel("Time (t)")
    axs[0].set_zlabel("Option Price (V)")

    axs[1].plot_surface(S, t, V_theoretical, cmap='plasma')
    axs[1].set_title("Theoretical Solution")
    axs[1].set_xlabel("Stock Price (S)")
    axs[1].set_ylabel("Time (t)")
    axs[1].set_zlabel("Option Price (V)")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparison_3d.png")
    plt.close()


    # Create 2D comparison plots at different time slices
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('2D Comparisons at Different Time Slices')

    time_indices = [0, -1]  # Beginning and end times
    titles = ['t=0', f't={t[0,-1]:.2f}']

    for i, (t_idx, title) in enumerate(zip(time_indices, titles)):
        # Plot PINN vs theoretical for this time slice
        axs[0,i].plot(S[:,0], V_pinn[:,t_idx], 'b-', label='PINN')
        axs[0,i].plot(S[:,0], V_theoretical[:,t_idx], 'r--', label='Theoretical')
        axs[0,i].set_title(title)
        axs[0,i].set_xlabel('Stock Price (S)')
        axs[0,i].set_ylabel('Option Price (V)')
        axs[0,i].legend()
        axs[0,i].grid(True)

        # Plot absolute error for this time slice
        error = np.abs(V_pinn[:,t_idx] - V_theoretical[:,t_idx])
        axs[1,i].plot(S[:,0], error, 'g-')
        axs[1,i].set_title(f'Absolute Error at {title}')
        axs[1,i].set_xlabel('Stock Price (S)')
        axs[1,i].set_ylabel('Absolute Error')
        axs[1,i].grid(True)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparison_2d.png")
    plt.close()
