## Spatial Grid Configuration

ISON supports different lattice geometries for spatial transcriptomics data.  
You can select the grid layout to define neighbor relationships for spatial autocorrelation and adjacency matrices.

| Option       | Description                                                                 | Default |
|--------------|-----------------------------------------------------------------------------|---------|
| `square`     | Square grid layout                                                           | Yes     |
| `hexagonal`  | Hexagonal grid layout                                                        |         |
| `triangle`   | Triangular grid layout (same nearest-neighbor structure as hexagonal)       |         |

### Usage

Specify the lattice type using the `--lattice` argument when running `run-ison.py`:

```bash
python run-ison.py --lattice square
python run-ison.py --lattice hexagonal
python run-ison.py --lattice triangle
