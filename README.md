# Differentiable Projective Dynamics

This codebase contains our research code for a few publications relevant to differentiable projective dynamics:
- [DiffPD: Differentiable Projective Dynamics](https://people.csail.mit.edu/taodu/diffpd/index.html) (ACM Transactions on Graphics/SIGGRAPH 2022)
- [DiffAqua: A Differentiable Computational Design Pipeline for Soft Underwater Swimmers with Shape Interpolation](http://diffaqua.csail.mit.edu/) (ACM SIGGRAPH 2021)
- [Underwater Soft Robot Modeling and Control with Differentiable Simulation](https://people.csail.mit.edu/taodu/starfish/index.html) (IEEE RA-L/RoboSoft 2021)

## Recommended systems
- Ubuntu 18.04
- (Mini)conda 4.7.12 or higher
- GCC 7.5 (Other versions might work but we tested the codebase with 7.5 only)

## Installation
```
git clone --recursive https://github.com/mit-gfx/diff_pd_public.git
cd diff_pd_public
conda env create -f environment.yml
conda activate diff_pd
./install.sh
```

## Examples
Navigate to the `python/example` path and run `python [example_name].py` where the `example_name` could be the following names. By default, we use 8 threads in OpenMP to run PD simulation. This number can be modified in most of the scripts below by changing the `thread_ct` variable. It is recommended to set `thread_ct` to be **strictly smaller** than the number of cores available.

For an extremely quick start, run the following script:
```
python routing_tendon_3d.py
python print_routing_tendon_3d.py
```

### Utilities
- `generate_texture` generates a square image with bounds. This is used for rendering only.
- `generate_torus` generates a torus model used in the examples.
- `pbrt_renderer_demo` shows how to interface pbrt using the python wrapper.
- `render_hex_mesh` explains how to use the external renderer (pbrt) to render a 3D hex mesh.
- `render_quad_mesh` explains how to use matplotlib to render a 2D quad mesh.
- `tet_demo` shows how to tetrahedralize a mesh.
- `voxelization_demo` shows how to voxelize a mesh.

### Numerical check
- `actuation_2d` and `actuation_3d` test the implementation of the muscle model.
- `collision_2d` compares the forward and backward implementation of collision models in Newton's methods and PD.
- `deformable_backward_2d` and `deformable_backward_3d` uses central differencing to numerically check the gradients of forward simulation in Newton-PCG, Newton-Cholesky, and PD methods.
- `deformable_quasi_static_3d` solves the quasi-static state of a 3D hex mesh. The hex mesh's bottom and top faces are fixed but the top face is twisted.
- `pd_energy_2d` and `pd_energy_3d` test the implementation of vertex-based and element-based projective dynamics energies.
- `pd_forward` verifies the forward simulation of projective dynamics by comparing it to the solutions from Newton's method.
- `state_force_2d` and `state_force_3d` test the implementation of state-based forces (e.g., friction, hydrodynamic force, penalty force for collisions) and their gradients w.r.t. position and velocity states.
- `run_all_tests` runs all numerical checks above.

### Evaluation
#### Sec. 6.1
- `landscape_3d.py` and `print_landscape_3d_table.py`: generate Fig. 2 of the paper.

#### Sec. 6.2
- `cantilever_3d.py` and `print_cantilever_3d_table.py`: generate Fig. 3 of the paper.
- `rolling_sphere_3d.py` and `print_rolling_sphere_3d_table.py`: generate Fig. 4 of the paper.
- `render_cantilever_3d.py`: generate mesh data for the `Cantilever` video.
- `render_rolling_sphere_3d.py`: generated mesh data for the `Rolling sphere` video.

#### Sec. 6.3
- `slope_3d.py` and `render_slope_3d.py`: generate Fig. 5 of the paper.
- `duck_3d.py` and `render_duck_3d.py`: generate Fig. 6 of the paper.
- `napkin_3d.py` and `render_napkin_3d.py`: generate Fig. 7 of the paper.

### Applications
#### Sec. 7.1
**Plant**
- `plant_3d.py`: run the `Plant` example on GCP (Google Cloud Platform. See the paper for its detail specification).
- `print_plant_3d.py`: generate data for Table 3 and Fig. 1 in supplemental material.
- `render_plant_3d.py`: generate mesh data for the `Plant` video.

**Bouncing ball**
- `bouncing_ball_3d.py`: run the `Bouncing ball` example on GCP.
- `print_bouncing_ball_3d.py`: generate data for Table 3 and Fig. 2 in supplemental material.
- `render_bouncing_ball_3d.py`: generate mesh data for the `Bouncing ball` video.

#### Sec. 7.2
**Bunny**
- `bunny_3d.py`: run the `Bunny` example on GCP.
- `print_bunny_3d.py`: generate data for Table 3 and Fig. 3 in supplemental material.
- `render_bunny_3d.py`: generate mesh data for the `Bunny` video.

**Routing tendon**
- `routing_tendon_3d.py`: run the `Routing tendon` example on GCP.
- `print_routing_tendon_3d.py`: generate data for Table 3 and Fig. 4 in supplemental materal.
- `render_routing_tendon_3d.py`: generate mesh data for the `Routing tendon` video.

#### Sec. 7.3
**Torus**
- `torus_3d.py`: run the `Torus` example on GCP.
- `print_torus_3d.py`: generate data for Table 3 and Fig. 5 in supplemental material.
- `render_torus_3d.py`: generate mesh data for the `Torus` video.

**Quadruped**
- `quadruped_3d.py`: run the `Quadruped` example on GCP.
- `print_quadruped_3d.py`: generated data for Table 3 and Fig. 6 in supplemental material.
- `render_quadruped_3d.py`: generate mesh data for the `Quadruped` video.

**Cow**
- `cow_3d.py`: run the `Cow` example on GCP.
- `print_cow_3d.py`: generate date for Table 3 and Fig. 7 in supplemental material.
- `render_cow_3d.py`: generate mesh data for the `Cow` video.

#### Sec. 7.4
Examples in this section require non-trivial setup of deep reinforcement learning pipelines. Please check out the [code](https://github.com/mit-gfx/DiffAqua) from our related paper [DiffAqua](http://diffaqua.csail.mit.edu/) for running fish examples.

#### Sec. 7.5
This section requires taking videos manually. Contact `taodu@csail.mit.edu` for more details.

#### Sec. 8
- `armadillo_3d.py`: generate the Armadillo experiment with Neohookean materials. This may take 5 minutes before rendering the results.

### Starfish
- `soft_starfish_3d.py`, `display_soft_starfish_3d.py`, and `render_soft_starfish_3d.py` are the scripts we used to generate results in the IEEE RA-L paper [Underwater Soft Robot Modeling and Control with Differentiable Simulation](https://people.csail.mit.edu/taodu/starfish/index.html) (IEEE RA-L/RoboSoft 2021). It involves non-trivial setup of hardware. Contact `taodu@csail.mit.edu` for more details.

## Contact
If you have trouble running any scripts above, please feel free to open an issue or email `taodu@csail.mit.edu`.
