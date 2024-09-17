import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import numpy as np
from scipy.integrate import solve_ivp
from funs_ephemerides import plot_ephemeris, create_figure

G = 6.6743e-11 # N m2/kg2 # universal gravitation constant

# Define the system of equations
def system(t, y, m1, m2, G):
    dim = 3
    r1 = y[:dim]  # Position vectors
    r2 = y[dim:dim * 2]  # Position vectors
    r = r2 - r1  # relative position vector
    r_norm = np.linalg.norm(r)  # norm of r

    v1 = y[dim * 2:dim * 3]  # Velocity vectors
    v2 = y[dim * 3:]  # Velocity vectors
    dr1dt = v1
    dr2dt = v2
    # Avoid division by zero
    if r_norm == 0:
        dv1dt = np.zeros(dim)
        dv2dt = np.zeros(dim)
    else:
        dv1dt = G * m2 * r / r_norm ** 3
        dv2dt = -G * m1 * r / r_norm ** 3
    return np.concatenate((dr1dt, dr2dt, dv1dt, dv2dt))


# Set up the sidebar
st.sidebar.title("Input Values")
m1 = st.sidebar.number_input("Mass 1, kg", value=5.97219*10**24, format="%.2e",  step=1e22)
m2 = st.sidebar.number_input("Mass 2, kg", value=7.34767309*10**22, format="%.2e",  step=1e20)

days = st.sidebar.number_input("Days to simulate", value=30,  step=1)

# d = 5e4 # m

# df = pd.DataFrame({
#     'r01, m': [-d / 2, 0, 0],
#     'r02, m': [d / 2, 0, 0],
#     'v01, m/s': [0, 1.2, 0.5],
#     'v02, m/s': [0.5, 0, 0],
# },index = ['x', 'y', 'z']
# )

df = pd.DataFrame({
    'r01, m': [1.227818898341106E+10, -1.515739918198268E+11, 4.933524339802563E+06],
    'r02, m': [1.190652740725140E+10, -1.515084515028273E+11,-2.831453670664132E+07],
    'v01, m/s': [2.920857511801704E+04, 2.304921291502835E+03,  -4.653442366951976E-01],
    'v02, m/s': [2.907405219199784E+04, 1.279666570141635E+03, 2.881803559327328E01],
},index = ['x', 'y', 'z']
)


input_df = st.sidebar.data_editor(df,
                                  column_config={
                                      "r01, m": st.column_config.NumberColumn(
                                          format="%.2e",
                                      ),
                                      "r02, m": st.column_config.NumberColumn(
                                          format="%.2e",
                                      ),
                                      "v01, m/s": st.column_config.NumberColumn(
                                          format="%.2e",
                                      ),
                                      "v02, m/s": st.column_config.NumberColumn(
                                          format="%.2e",
                                      ),
                                  },
                                  )

recompute = st.sidebar.checkbox("Recompute Graphs?")

# Initial conditions: r0 (initial position), v0 (initial velocity)

r01 = input_df['r01, m']
r02 = input_df['r02, m']

v01 = input_df['v01, m/s']
v02 = input_df['v02, m/s']


y0 = np.concatenate((r01, r02, v01, v02))

# Time range
t_span = (0, days*24*60*60) # s
t_eval = np.linspace(t_span[0], t_span[1], 3000)

if recompute:
    # Solve the system
    sol = solve_ivp(system, t_span, y0, t_eval=t_eval, args=(m1, m2, G,), method='Radau')

    # Extract the solution for r
    r1_sol = sol.y[:3].T
    r2_sol = sol.y[3:6].T

    r1_sol_df = pd.DataFrame({
        'X (km)': r1_sol[:, 0],
        'Y (km)': r1_sol[:, 1],
        'Z (km)': r1_sol[:, 2],
    })
    r2_sol_df = pd.DataFrame({
        'X (km)': r2_sol[:, 0],
        'Y (km)': r2_sol[:, 1],
        'Z (km)': r2_sol[:, 2],
    })


    r = r2_sol_df - r1_sol_df  # relative position vector
    rG = (m1 * r1_sol_df + m2 * r2_sol_df) / (m1 + m2)  # Center of gravity
    r_norm = np.linalg.norm(r, axis=1)  # norm of r

    plot_m1 = plot_ephemeris(r1_sol_df, label='m1')
    plot_m2 = plot_ephemeris(r2_sol_df, label='m2')
    plot_g = plot_ephemeris(rG, label='CoM')
    # Generate data for the plot
    fig = create_figure([plot_m1, plot_m2, plot_g], axis_range=None)
    # Display the plot in the main area
    st.plotly_chart(fig, use_container_width=True)


    plot_m2_m1 = plot_ephemeris(r, label='r_m2_m1')
    plot_g_m1 = plot_ephemeris(rG-r1_sol_df, label='r_g_m1')
    plot_m1_m1 = plot_ephemeris(r1_sol_df-r1_sol_df, label='m1')
    # Generate data for the plot
    fig2 = create_figure([plot_m2_m1, plot_g_m1, plot_m1_m1], axis_range=None)
    # Display the plot in the main area
    st.plotly_chart(fig2, use_container_width=True)


    ###
    plot_m1_g = plot_ephemeris(r1_sol_df-rG, label='r_m1_g')
    plot_m2_g = plot_ephemeris(r2_sol_df-rG, label='r_m2_g')
    plot_g_g = plot_ephemeris(rG-rG, label='CoM')
    # Generate data for the plot
    fig3 = create_figure([plot_m1_g, plot_m2_g, plot_g_g], axis_range=None)
    # Display the plot in the main area
    st.plotly_chart(fig3, use_container_width=True)
