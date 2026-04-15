# %%
from solidspy import solids_GUI
# solids_GUI()


disp, strain, stress = solids_GUI(plot_contours=False, compute_strains=True, folder="./") # modify the folder


# %%
import numpy as np

# Extract Results
U_x, U_y = disp[:, 0], disp[:, 1]
Sig_xx, Sig_yy, Tau_xy = stress[:, 0], stress[:, 1], stress[:, 2]
von_mises = np.sqrt(Sig_xx**2 - Sig_xx*Sig_yy + Sig_yy**2 + 3*Tau_xy**2)

#%% 

# ==============================================================================
# 4. POST-PROCESSING (Stress Projection)
# ==============================================================================
# Calculate Tangential (Hoop) Stress based on local geometry
theta_all = np.arctan2(Y, X)
theta_in_raw = np.arctan2(Y[:n_theta], X[:n_theta])
sort_idx = np.argsort(theta_in_raw)

x_in, y_in = X[:n_theta][sort_idx], Y[:n_theta][sort_idx]
theta_in_sorted = theta_in_raw[sort_idx]

# Tangent/Normal calculation
dx_in, dy_in = np.gradient(x_in), np.gradient(y_in)
alpha_inner = np.arctan2(dy_in, dx_in) - (np.pi / 2.0)

# Map normal angle to all nodes
closest_idx = np.argmin(np.abs(theta_all[:, np.newaxis] - theta_in_sorted), axis=1)
alpha_all = alpha_inner[closest_idx]
c, s = np.cos(alpha_all), np.sin(alpha_all)

# Project Stress
Sig_tt = Sig_xx * s**2 + Sig_yy * c**2 - 2 * Tau_xy * s * c


# ==============================================================================
# 5. VISUALIZATION (Consolidated Subplots)
# ==============================================================================
fig = plt.figure(figsize=(16, 10))

# --- Subplot 1: Geometry & Deformation ---
ax1 = plt.subplot(1, 3, 1)
scale = 1  # Deformation scale factor
ax1.triplot(X, Y, tri_elements, color='gray', linestyle='--', alpha=0.3)
ax1.triplot(X + U_x*scale, Y + U_y*scale, tri_elements, color='black', lw=0.5)
ax1.set_title("Deformed Mesh", fontweight='bold')
ax1.axis('equal'); ax1.axis('off')

# --- Subplot 2: Von Mises Stress ---
ax2 = plt.subplot(1, 3, 2)
levels_vm = np.linspace(0, np.max(von_mises), 50)
cont2 = ax2.tricontourf(X, Y, tri_elements, von_mises, levels=levels_vm, cmap='inferno')
ax2.set_title("Von Mises Stress", fontweight='bold')
ax2.axis('equal'); ax2.axis('off')
fig.colorbar(cont2, ax=ax2, orientation='horizontal', pad=0.05)

# --- Subplot 3: Tangential (Hoop) Stress ---
ax3 = plt.subplot(1, 3, 3)
max_tt = np.max(np.abs(Sig_tt))
levels_tt = np.linspace(-max_tt, max_tt, 50)
cont3 = ax3.tricontourf(X, Y, tri_elements, Sig_tt, levels=levels_tt, cmap='coolwarm')
ax3.set_title("Tangential (Hoop) Stress", fontweight='bold')
ax3.axis('equal'); ax3.axis('off')
fig.colorbar(cont3, ax=ax3, orientation='horizontal', pad=0.05)

plt.suptitle(f"Thick Cylinder Analysis (Pressure: {internal_pressure})", fontsize=16, fontweight='bold')
plt.tight_layout(pad=2.0)
plt.show()