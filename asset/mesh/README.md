Meshlab tutorial:

To improve the quality of the triangle meshes, try the following options:
- Do ``Cleaning and Repairing`` -> ``Remove Duplicated Faces`` and ``Remove Duplicated Vertex`` constantly;
- If there are many sharp triangles, try ``Cleaning and Repairing`` -> ``Merge Close Vertices`` and tune the distance. Optionally, run ``Remeshing, Simplification and Reconstruction`` -> ``Subdivision Surfaces: Loop`` to subdivide triangles before you merge them;
- Use ``Remeshing, Simplification and Reconstruction`` -> ``Uniform Mesh Sampling``, ``Iso Parametrization``, or ``Quadratic Edge Collapse Decimation`` to adjust the triangles so that they are more regular.