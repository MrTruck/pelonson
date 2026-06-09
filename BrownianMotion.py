import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(
    page_title="Brownian Motion Simulator",
    layout="wide"
)

st.title("Brownian Motion Simulator")

# ==========================
# SIDEBAR
# ==========================

st.sidebar.header("Simulation Parameters")

num_particles = st.sidebar.slider(
    "Number of Particles",
    1,
    100,
    30
)

num_steps = st.sidebar.slider(
    "Number of Steps",
    50,
    1000,
    400
)

step_magnitude = st.sidebar.slider(
    "Step Magnitude",
    0.1,
    2.0,
    0.5
)

distribution = st.sidebar.selectbox(
    "Distribution",
    ["Gaussian", "Uniform"]
)

min_lifetime = st.sidebar.slider(
    "Minimum Lifetime",
    10,
    num_steps,
    100
)

max_lifetime = st.sidebar.slider(
    "Maximum Lifetime",
    min_lifetime,
    num_steps,
    num_steps
)

# ==========================
# SIMULATION
# ==========================

@st.cache_data
def run_simulation(
    num_particles,
    num_steps,
    step_magnitude,
    distribution,
    min_lifetime,
    max_lifetime
):

    lifetimes = np.random.randint(
        min_lifetime,
        max_lifetime + 1,
        num_particles
    )

    x_positions = np.zeros(
        (num_steps, num_particles)
    )

    y_positions = np.zeros(
        (num_steps, num_particles)
    )

    for p in range(num_particles):

        for i in range(1, num_steps):

            if i < lifetimes[p]:

                if distribution == "Gaussian":

                    dx = np.random.normal(
                        0,
                        step_magnitude
                    )

                    dy = np.random.normal(
                        0,
                        step_magnitude
                    )

                else:

                    dx = np.random.uniform(
                        -step_magnitude,
                        step_magnitude
                    )

                    dy = np.random.uniform(
                        -step_magnitude,
                        step_magnitude
                    )

                x_positions[i, p] = (
                    x_positions[i - 1, p] + dx
                )

                y_positions[i, p] = (
                    y_positions[i - 1, p] + dy
                )

            else:

                x_positions[i, p] = (
                    x_positions[i - 1, p]
                )

                y_positions[i, p] = (
                    y_positions[i - 1, p]
                )

    return (
        x_positions,
        y_positions,
        lifetimes
    )

x_positions, y_positions, lifetimes = run_simulation(
    num_particles,
    num_steps,
    step_magnitude,
    distribution,
    min_lifetime,
    max_lifetime
)

# ==========================
# TABS
# ==========================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Trajectory",
        "Animation",
        "MSD",
        "Statistics"
    ]
)

# ==========================
# TRAJECTORY
# ==========================

with tab1:

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.plot(
        x_positions,
        y_positions,
        linewidth=1
    )

    ax.set_title(
        "Particle Trajectories"
    )

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    ax.grid(True)
    ax.set_aspect("equal")

    st.pyplot(fig)

# ==========================
# ANIMATION
# ==========================

# ==========================
# ANIMATION (PLOTLY)
# ==========================

with tab2:

    st.subheader("Interactive Brownian Motion Animation")

    frame_skip = max(1, num_steps // 100)

    frames = []

    x_min = np.min(x_positions)
    x_max = np.max(x_positions)

    y_min = np.min(y_positions)
    y_max = np.max(y_positions)

    colors = [
        f"hsl({i * 360 / num_particles},70%,50%)"
        for i in range(num_particles)
    ]

    # Initial frame
    initial_data = []

    for p in range(num_particles):

        initial_data.append(
            go.Scatter(
                x=[x_positions[0, p]],
                y=[y_positions[0, p]],
                mode="markers",
                marker=dict(
                    size=8,
                    color=colors[p]
                ),
                showlegend=False
            )
        )

    # Animation frames
    for frame in range(
        0,
        num_steps,
        frame_skip
    ):

        frame_data = []

        for p in range(num_particles):

            end_frame = min(
                frame,
                lifetimes[p]
            )

            frame_data.append(
                go.Scatter(
                    x=x_positions[
                        :end_frame + 1,
                        p
                    ],
                    y=y_positions[
                        :end_frame + 1,
                        p
                    ],
                    mode="lines",
                    line=dict(
                        width=2,
                        color=colors[p]
                    ),
                    showlegend=False
                )
            )

            if frame < lifetimes[p]:

                frame_data.append(
                    go.Scatter(
                        x=[
                            x_positions[
                                frame,
                                p
                            ]
                        ],
                        y=[
                            y_positions[
                                frame,
                                p
                            ]
                        ],
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=colors[p]
                        ),
                        showlegend=False
                    )
                )

            else:

                frame_data.append(
                    go.Scatter(
                        x=[],
                        y=[],
                        mode="markers",
                        showlegend=False
                    )
                )

        frames.append(
            go.Frame(
                data=frame_data,
                name=str(frame)
            )
        )

    fig = go.Figure(
        data=frames[0].data,
        frames=frames
    )

    fig.update_layout(

        title="Brownian Motion Animation",

        xaxis=dict(
            title="X Position",
            range=[
                x_min - 5,
                x_max + 5
            ]
        ),

        yaxis=dict(
            title="Y Position",
            range=[
                y_min - 5,
                y_max + 5
            ],
            scaleanchor="x",
            scaleratio=1
        ),

        height=700,

        updatemenus=[
            {
                "type": "buttons",

                "buttons": [

                    {
                        "label": "▶ Play",

                        "method": "animate",

                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": 50,
                                    "redraw": True
                                },

                                "fromcurrent": True,

                                "transition": {
                                    "duration": 0
                                }
                            }
                        ]
                    },

                    {
                        "label": "⏸ Pause",

                        "method": "animate",

                        "args": [
                            [None],
                            {
                                "frame": {
                                    "duration": 0
                                },

                                "mode":
                                "immediate"
                            }
                        ]
                    }

                ]
            }
        ],

        sliders=[
            {
                "steps": [

                    {
                        "method":
                        "animate",

                        "label":
                        str(frame),

                        "args": [
                            [str(frame)],
                            {
                                "mode":
                                "immediate",

                                "frame": {
                                    "duration": 0,
                                    "redraw": True
                                }
                            }
                        ]
                    }

                    for frame in range(
                        0,
                        num_steps,
                        frame_skip
                    )

                ]
            }
        ]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
# ==========================
# MSD
# ==========================

with tab3:

    msd = np.mean(
        x_positions**2 +
        y_positions**2,
        axis=1
    )

    fig, ax = plt.subplots()

    ax.plot(msd)

    ax.set_title(
        "Mean Squared Displacement"
    )

    ax.set_xlabel(
        "Time Step"
    )

    ax.set_ylabel(
        "MSD"
    )

    ax.grid(True)

    st.pyplot(fig)

# ==========================
# STATISTICS
# ==========================

with tab4:

    final_distances = np.sqrt(
        x_positions[-1]**2 +
        y_positions[-1]**2
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Distance",
        f"{np.mean(final_distances):.2f}"
    )

    col2.metric(
        "Maximum Distance",
        f"{np.max(final_distances):.2f}"
    )

    col3.metric(
        "Particles",
        num_particles
    )

    fig, ax = plt.subplots()

    ax.hist(
        final_distances,
        bins=15
    )

    ax.set_title(
        "Final Distance Distribution"
    )

    st.pyplot(fig)

    st.subheader(
        "Lifetime Distribution"
    )

    fig2, ax2 = plt.subplots()

    ax2.hist(
        lifetimes,
        bins=15
    )

    ax2.set_title(
        "Particle Lifetimes"
    )

    st.pyplot(fig2)