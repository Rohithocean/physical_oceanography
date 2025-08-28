import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Streamlit UI
st.title("🌊 Mixed Layer Heat Budget Simulator)")

st.markdown("""
This app simulates the temperature evolution in the ocean's surface mixed layer using the heat budget equation..
""")

# Time settings
time_steps = st.slider("Number of time steps", min_value=10, max_value=200, value=100)
dt = st.number_input("Time step duration (hours)", min_value=0.1, value=1.0)

# Constants
rho0 = 1025  # kg/m³
cp = 3940    # J/kg/°C

# Initial temperature
T_init = st.number_input("Initial mixed layer temperature (°C)", value=28.0)

# deep temperature
T_sub = st.number_input("Subsurface temperature T(-h) (°C)", value=20.0)

# User inputs for time-varying parameters
st.subheader("📈 Varying Inputs")

q0     = st.slider("Surface heat flux q₀ (W/m²)", min_value=-300, max_value=300, value=0)
q_pen  = st.slider("Penetrative radiation q_pen (%)", min_value=0, max_value=100, value=0)

u_a    = st.slider("Surface current velocity u (m/s)", min_value=-1.0, max_value=1.0, value=0.00)
u_sub  = st.slider("Subsurface zonal velocity u(-h) (m/s)", min_value=-1.0, max_value=1.0, value=0.0)
w_sub  = st.slider("Vertical velocity w(-h) (m/s)", min_value=-0.01, max_value=0.01, value=0.0)

h      = st.slider("Mixed layer depth h (m)", min_value=5, max_value=100, value=50)

dTdx   = st.slider("∂T/∂x (°C/°)", min_value=-0.25, max_value=0.25, value=0.00)
dTdx_m = dTdx / (110*1000)  # °C/m

# MLD slopes (dh/dx, dh/dy)
dh_dx = st.slider("∂h/∂x (°m/°)", min_value=-0.25, max_value=0.25, value=0.00)
dh_dxm = dh_dx / (110*1000)  # °C/m


d2Tdx2 = st.number_input("Second-order Temperature difference (°C/deg²)", value=1e-4)
d2Tdx2_m = d2Tdx2/(110*1000) #°C/m

dTdz = st.number_input("∂T/∂z (°C/m)", value=1e-4)

# Eddy diffusivities
kappa_H = st.number_input("Horizontal eddy diffusivity κ_H (m²/s)", value=10.0)
kappa_Z = st.number_input("Vertical eddy diffusivity κ_Z (m²/s)", value=1e-5)

# --- New Subsurface Inputs ---

# dh/dt (if MLD is fixed, =0; else allow user to give a value)
dh_dt = st.number_input("MLD slope dh/dt (m/hour)", value=0.0)

# Initialize temperature array
T = np.zeros(time_steps)
T[0] = T_init

# Compute temperature tendency over time
for t in range(1, time_steps):
    advective_term = -(u_a * dTdx_m)
    horizontal_mixing = kappa_H * (d2Tdx2_m + d2Tdx2_m)
    vertical_mixing = (-1 / h) * kappa_Z * dTdz
    net_flux = (q0 - (q0*q_pen/100)) / (rho0 * cp * h)
    # dh/dt (constant h for now → 0)
    dh_dt = 0.0  
    # Entrainment term
    entrainment = -1 * ((T_init - T_sub) / h) * (dh_dt/3600 + w_sub/3600 + u_sub*dh_dxm + u_sub*dh_dxm)
    dTdt = net_flux + advective_term + horizontal_mixing + vertical_mixing + entrainment
    T[t] = T[t-1] + dTdt * dt * 3600  # convert dt to seconds

# ---------------- Visualization ----------------

# Normalize values for color mapping
def normalize(val, vmin, vmax):
    return (val - vmin) / (vmax - vmin)

# Colors
sky_color = plt.cm.hot(normalize(q0, -300, 300))
subsurface_color = plt.cm.Blues_r(normalize(q_pen, 0, 100))
subsurface_alpha = 0.5

# Gradient orientation
if dTdx >= 0:
    left_color = 'blue'
    right_color = 'red'
else:
    left_color = 'red'
    right_color = 'blue'

# Create figure
fig, ax = plt.subplots(figsize=(7,3))

# Sky layer
sky = patches.Rectangle((0, 0.8), 1, 0.2, color=sky_color, alpha=0.5)
ax.add_patch(sky)
ax.text(0.98, 0.95, "Atmosphere", ha="right", va="top", fontsize=10, weight="bold")

# Wavy top of ocean
wave_x = np.linspace(0, 1, 500)
wave_y = 0.8 + 0.01 * np.sin(20 * np.pi * wave_x)
ax.fill_between(wave_x, wave_y, 1.0, color=sky_color, alpha=0.5)
ax.plot(wave_x, wave_y, color='white', linewidth=5)

# Mixed layer (sloping bottom)

ml_top = 0.8
ml_depth_norm = 0.5 * (h / 100)  # scale depth to plot height
ml_bottom_left = ml_top - ml_depth_norm
ml_bottom_right = ml_bottom_left - dh_dx  # slope effect

from matplotlib.path import Path
from matplotlib.patches import PathPatch

# Mixed layer gradient: red to blue or blue to red
if dTdx >= 0:
    left_color = 'blue'
    right_color = 'red'
else:
    left_color = 'red'
    right_color = 'blue'

# Transparency scaling with gradient magnitude
alpha = min(1.0, max(0.1, abs(dTdx) / 1.0))

# Create gradient image (horizontal)
width, height = 400, 100
gradient      = np.linspace(0, 1, width)
gradient_rgb  = np.zeros((height, width, 4))

for i in range(width):
    r = gradient[i] if right_color == 'red' else 1 - gradient[i]
    b = gradient[i] if right_color == 'blue' else 1 - gradient[i]
    gradient_rgb[:, i, 0] = r  # Red channel
    gradient_rgb[:, i, 2] = b  # Blue channel
    gradient_rgb[:, i, 3] = alpha

# Show gradient (cover whole ML bounding box first)
ml_bottom_left = ml_top - ml_depth_norm
ml_bottom_right = ml_bottom_left - dh_dx
min_bottom = min(ml_bottom_left, ml_bottom_right)
max_bottom = ml_top

im = ax.imshow(gradient_rgb, extent=[0, 1, min_bottom, max_bottom], aspect='auto', zorder=1)

# Define polygon path for mixed layer (sloping bottom)
ml_polygon_x = [0, 1, 1, 0]
ml_polygon_y = [ml_top, ml_top, ml_bottom_right, ml_bottom_left]
poly_coords = list(zip(ml_polygon_x, ml_polygon_y))

path = Path(poly_coords)
patch = PathPatch(path, transform=ax.transData)

# Clip gradient to polygon shape
im.set_clip_path(patch)

#ml_polygon_x = [0, 1, 1, 0]
#ml_polygon_y = [ml_top, ml_top, ml_bottom_right, ml_bottom_left]
#ax.fill(ml_polygon_x, ml_polygon_y, color="lightblue", alpha=0.6)
ax.text(0.98, (ml_top + (ml_bottom_left+ml_bottom_right)/2)/2, "Mixed Layer",
        ha="right", va="top", fontsize=10, weight="bold", color="black")

# Arrow across bottom boundary (entrainment/detrainment)
mid_x = 0.5
mid_y = (ml_bottom_left + ml_bottom_right) / 2  # midpoint of sloping bottom

# Arrow length proportional to w_sub (scaled for plotting)
arrow_scale = 5   # adjust scale factor for visibility
ax.arrow(mid_x, mid_y, 0, w_sub*arrow_scale,
         head_width=0.05, head_length=0.02,
         fc='green', ec='green')

# Label
ax.text(mid_x+0.05, mid_y - 0.04 + w_sub*arrow_scale,
        "Vertical velocity", fontsize=8, color='green', va="center")

# Subsurface layer (below mixed layer)
subsurface_y = min(ml_bottom_left, ml_bottom_right)

deep_polygon_x = [0, 1, 1, 0]
deep_polygon_y = [ml_bottom_left, ml_bottom_right, 0, 0]
ax.fill(deep_polygon_x, deep_polygon_y, color=subsurface_color, alpha=subsurface_alpha)


#subsurface = patches.Rectangle((0, 0), 1, subsurface_y, color=subsurface_color, alpha=subsurface_alpha)
#ax.add_patch(subsurface)
ax.text(0.98, subsurface_y/2, "Deep Ocean", ha="right", va="top",
        fontsize=10, weight="bold", color="navy")

# Surface current arrow
v_a = 0
speed = np.sqrt(u_a**2 + v_a**2)
arrow_length = 0.4 * speed
arrow_dx = arrow_length * u_a / (speed + 1e-6)
ax.arrow(0.5, (ml_top+ml_bottom_left)/2, arrow_dx, 0, head_width=0.05,
         head_length=0.02, fc='black', ec='black')

# Subsurface to mixed layer velocity vector
arrow_scale = 0.4
ax.arrow(0.3, subsurface_y/2, arrow_scale*u_sub, arrow_scale*w_sub,
         head_width=0.05, head_length=0.02, fc='darkred', ec='darkred')
ax.text(0.35, subsurface_y/2+0.05, "Flow below ML", fontsize=8, color='darkred')

# Formatting
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
st.pyplot(fig)

# ---------------- Time Evolution Plot ----------------
st.subheader("📊 Mixed Layer Temperature Over Time")
fig1, ax1 = plt.subplots()
ax1.plot(np.arange(time_steps) * dt, T, label="Mixed Layer Temperature")
ax1.set_xlabel("Time (hours)")
ax1.set_ylabel("Temperature (°C)")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

