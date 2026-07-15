import numpy as np
import potpourri3d as pp3d
import polyscope as ps

# 1. Initialize Polyscope (creates the native desktop window)
ps.init()

# 2. Load your 3D mesh
# Ensure "cube_100k.obj" is located in your "D:\python\Project\" folder
v, f = pp3d.read_mesh("cube_100k.obj")

# 3. Compute the actual geodesic distances (using vertex 0 as the source)
solver = pp3d.MeshHeatMethodDistanceSolver(v, f)
kernel_geodesic = solver.compute_distance(0)

# 4. Register the mesh with Polyscope
ps_mesh = ps.register_surface_mesh("cube_mesh", v, f)

# 5. Add the geodesic data as a color map (Viridis style)
ps_mesh.add_scalar_quantity("Geodesic Distance", kernel_geodesic, enabled=True, cmap='viridis')

# 6. Mark the first 5 vertices with black point markers
top_5_vertices = v[:5]
ps.register_points("source_points", top_5_vertices, radius=0.01, color=(0, 0, 0))

# 7. Open the raw desktop UI window
ps.show()
