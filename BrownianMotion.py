import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(
    page_title="Brownian Motion Simulator",
    layout="wide"
)

st.title("Brownian Motion Simulator")
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = None

# ==========================
# SIDEBAR
# ==========================

with st.sidebar.form("simulation_form"):

    # Total number of independent particles to simulate
    num_particles = st.slider(
        "Number of Particles",
        1,
        100,
        30
    )

    # Total number of discrete time steps in each particle's trajectory
    num_steps = st.slider(
        "Number of Steps",
        50,
        1000,
        400
    )

    # Absolute temperature in Kelvin — higher T increases thermal energy,
    # which increases D and therefore larger random steps
    temperature = st.number_input(
        "Temperature (K)",
        value=300.0
    )

    # Dynamic viscosity of the surrounding fluid in Pa·s
    # Higher viscosity resists particle motion, decreasing D and step size
    # Water at 25°C ≈ 0.001 Pa·s
    viscosity = st.number_input(
        "Viscosity (Pa·s)",
        value=0.01
    )

    # Physical radius of each particle in metres
    # Larger particles experience more drag, reducing D and step size
    particle_radius = st.number_input(
        "Particle Radius (m)",
        value=1e-6,
        format="%e"
    )

    # Duration of each simulation time step in seconds
    # Larger dt means each step covers more physical time,
    # increasing sigma (step size) proportionally to sqrt(dt)
    dt = st.number_input(
        "Time Step (s)",
        value=1000
    )
    # Minimum number of steps a particle is active before it stops moving.
    # Once a particle's lifetime is reached, it freezes at its last position.
    min_lifetime = st.slider(
        "Minimum Lifetime",
        10,
        num_steps,
        100
    )

    # Maximum number of steps a particle can remain active.
    # Lifetimes are drawn uniformly from [min_lifetime, max_lifetime].
    max_lifetime = st.slider(
        "Maximum Lifetime",
        min_lifetime,
        num_steps,
        num_steps
    )

    submitted = st.form_submit_button(
        "Run Simulation"
    )

# ==========================
# DIFFUSION COEFFICIENT
# ==========================

# Boltzmann constant (J/K) — relates thermal energy to temperature
kB = 1.380649e-23

# Stokes-Einstein equation: D = k_B * T / (6 * pi * eta * r)
# D has units of m²/s and quantifies how quickly a particle spreads
# through the fluid due to thermal collisions with solvent molecules
D = (kB * temperature) / (6 * np.pi * viscosity * particle_radius) 

# Standard deviation of each Gaussian displacement step
# Derived from the mean squared displacement relation: <x²> = 2*D*dt per axis
# So one step drawn from N(0, sigma) gives the correct physical diffusion rate
sigma = np.sqrt(2*D*dt)


# ==========================
# SIMULATION
# ==========================

# @st.cache_data memoises the function — if all arguments are identical
# to a previous call, Streamlit returns the cached result immediately
# instead of re-running the simulation. This prevents redundant computation
# on every UI interaction that doesn't change simulation parameters.
@st.cache_data
def run_simulation(
    num_particles,
    num_steps,
    sigma,
    min_lifetime,
    max_lifetime
):
    # Draw each particle's lifetime as a random integer in [min_lifetime, max_lifetime].
    # Shape: (num_particles,)
    lifetimes = np.random.randint( 
        min_lifetime,
        max_lifetime + 1,
        num_particles
    ) #generates a "num_particle" long array of random numbers from min to max lifetime 

    # Pre-allocate position arrays filled with zeros.
    # Row i = time step i, column p = particle p.
    # Starting positions are implicitly (0, 0) for all particles.
    # Shape: (num_steps, num_particles)
    x_positions = np.zeros(
        (num_steps, num_particles)
    )

    y_positions = np.zeros(
        (num_steps, num_particles)
    )

    for p in range(num_particles):

        for i in range(1, num_steps):  # start at 1 because step 0 is the origin

            if i < lifetimes[p]:
                # Particle is still alive: draw a random displacement
                # from a zero-mean Gaussian with std = sigma.
                # Each axis is independent, producing isotropic 2D diffusion.
                dx = np.random.normal(
                    0, sigma 
                )
                
                dy = np.random.normal(0, sigma)

                # Integrate position: new position = previous + random step
                x_positions[i, p] = (
                    x_positions[i - 1, p] + dx
                )

                y_positions[i, p] = (
                    y_positions[i - 1, p] + dy
                )

            else:
                # Particle has exceeded its lifetime: freeze it in place
                # by copying the previous position (zero displacement)
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

if submitted:

    st.session_state.simulation_results = run_simulation(
        num_particles,
        num_steps,
        sigma,
        min_lifetime,
        max_lifetime
    )
if st.session_state.simulation_results is not None:

    x_positions, y_positions, lifetimes = (
        st.session_state.simulation_results
    )

else:
    st.info(
        "Adjust parameters and click 'Run Simulation'"
    )
    st.stop()

# Overall bounding box of all trajectories — used to set consistent axis limits
x_min = np.min(x_positions)
x_max = np.max(x_positions)

y_min = np.min(y_positions)
y_max = np.max(y_positions) 

max_displacement = max(
    abs(x_min),
    abs(x_max),
    abs(y_min),
    abs(y_max)
)

plot_limit = max_displacement * 1.1

# ==========================
# TABS
# ==========================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Trajectory",
        "Animation",
        "MSD",
        "Gaussian Distribution"
    ]
)

# ==========================
# TRAJECTORY
# ==========================

with tab1:

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    # Passing the full 2D arrays to ax.plot() draws one line per column (particle).
    # x_positions shape: (num_steps, num_particles) — matplotlib treats each column
    # as a separate series when x and y have matching shapes.
    ax.plot(
        x_positions,
        y_positions,
        linewidth=1
    )

    # Fixed axis limits keep the view consistent across reruns
    # and prevent extreme outlier trajectories from collapsing the scale
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)
    
    ax.set_title(
        "Particle Trajectories"
    )

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    ax.grid(True)
    ax.set_aspect("equal")  # equal scaling so circles look like circles

    st.pyplot(fig)

# ==========================
# ANIMATION (PLOTLY)
# ==========================

with tab2:

    st.subheader("Interactive Brownian Motion Animation")

    # Subsample frames so we render at most ~50 animation frames regardless
    # of num_steps. Fewer frames = faster build time and smoother browser playback.
    frame_skip = max(1, num_steps // 50)

    frames = []  # will hold one go.Frame object per sampled time step

    # Recompute limits inside the tab (overrides the earlier globals)
    x_min = np.min(x_positions)
    x_max = np.max(x_positions)

    y_min = np.min(y_positions)
    y_max = np.max(y_positions)

    # Assign each particle a distinct hue evenly spaced around the HSL colour wheel
    colors = [
        f"hsl({i * 360 / num_particles},70%,50%)"
        for i in range(num_particles)
    ]

    # Build the static initial marker traces shown before Play is pressed.
    # One Scattergl trace per particle, positioned at the origin (step 0).
    initial_data = []

    for p in range(num_particles):

        initial_data.append(
            go.Scattergl(           # Scattergl uses WebGL for faster rendering
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

    # Build one go.Frame for each sampled time step.
    # Each frame contains 2 traces per particle: a line (trail) + a dot (current position).
    for frame in range(
        0,
        num_steps,
        frame_skip
    ): #(min, max, step)

        frame_data = []

        for p in range(num_particles): 

            # Don't draw trail past the particle's death step
            end_frame = min(
                frame,
                lifetimes[p]
            )

            # Trace 1: trajectory trail from step 0 up to end_frame
            frame_data.append(
                go.Scattergl(
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
                # Trace 2a: particle is alive — draw dot at its current position
                frame_data.append(
                    go.Scattergl(
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
                # Trace 2b: particle is dead — add an empty trace to keep
                # the trace count per frame constant (Plotly requires this
                # for frame transitions to map traces correctly)
                frame_data.append(
                    go.Scattergl(
                        x=[],
                        y=[],
                        mode="markers",
                        showlegend=False
                    )
                )

        # Each go.Frame bundles all traces for one time step.
        # 'name' must match the slider step labels for seek-to-frame to work.
        frames.append(
            go.Frame( #looks like this: Frame0 = { data: frame data, name :"0"} and frames is a list of these objects 

                data=frame_data,
                name=str(frame)
            )
        )

    # Initialise the figure with the first frame's data so something is visible
    # before the user presses Play
    fig = go.Figure(
        data=frames[0].data, #initial display. When "play" is clicked, it goes to the next frame (with frame skip)
        frames=frames  # available animation frames 
    )

    # Dynamic scaling based on simulation results
    plot_limit = max(
    abs(x_min),
    abs(x_max),
    abs(y_min),
    abs(y_max)
    ) * 1.1

    fig.update_layout(

        title="Brownian Motion Animation",

        xaxis=dict(
            title="X Position",
            range=[
                -plot_limit,
                plot_limit
            ]
        ),

        yaxis=dict(
            title="Y Position",
            range=[
                -plot_limit,
                plot_limit
            ],
            scaleanchor="x",    # lock y scale to x so aspect ratio = 1
            scaleratio=1
        ),

        height=700,

        # Play/Pause buttons — 'animate' method drives the frame loop
        updatemenus=[
            {
                "type": "buttons",

                "buttons": [

                    {
                        "label": "▶ Play",

                        "method": "animate",

                        "args": [
                            None,           # None = play all frames in sequence
                            {
                                "frame": {
                                    "duration": 50,     # ms per frame
                                    "redraw": True      # redraw WebGL on each frame
                                },

                                "fromcurrent": True,    # resume from current slider position

                                "transition": {
                                    "duration": 0       # no interpolation between frames
                                }
                            }
                        ]
                    },

                    {
                        "label": "⏸ Pause",

                        "method": "animate",

                        "args": [
                            [None],         # list containing None = stop immediately
                            {
                                "frame": {
                                    "duration": 0
                                },

                                "mode": "immediate"     # don't wait for current frame to finish
                            }
                        ]
                    }

                ]
            }
        ],

        # Scrub slider — one step per sampled frame, labelled with the step index
        sliders=[
            {
                "steps": [

                    {
                        "method": "animate",

                        "label": str(frame),

                        "args": [
                            [str(frame)],   # seek to the frame whose name matches
                            {
                                "mode": "immediate",

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

    # Mean Squared Displacement: average of (x² + y²) across all particles at each time step.
    # For free 2D Brownian motion, MSD should grow linearly: MSD(t) = 4 * D * t.
    # Deviations from linearity (e.g. plateau) indicate confinement or particle death effects.
    msd = np.mean(
        x_positions**2 +
        y_positions**2,
        axis=1  # average across columns (particles), keeping the time axis intact
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
# GAUSSIAN DIST GRAPH
# ==========================

with tab4:

    st.subheader("Particle X-Position Distribution")

    # Extract the final x position of every particle (last row of the position array)
    final_x = x_positions[-1]

    fig, ax = plt.subplots()

    # density=True normalises the histogram to a probability density so the area = 1.
    # By the Central Limit Theorem, the sum of many independent Gaussian steps
    # is itself Gaussian — this plot lets you verify that visually.
    ax.hist(
        final_x,
        bins=20,
        density=True
    )

    ax.set_title(
        "Gaussian Distribution of X Positions"
    )

    ax.set_xlabel("X Position")
    ax.set_ylabel("Probability Density")

    ax.grid(True)

    st.pyplot(fig)