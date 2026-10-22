# Multimedia Retrieval: Content-Based 3D Shape Retrieval

Given a 3D shape, the system finds and shows the most similar shapes in a database of 3D shapes.

## Setup

1. Create a virtual environment with Python 3.10 and install the dependencies:
   ```
   python3.10 -m venv .venv310
   .venv310/bin/pip install -r requirements.txt
   ```
2. Put the shape database in `data/ShapeDatabase/` (one folder per class, OBJ files inside):
   ```
   data/ShapeDatabase/Bicycle/D00016.obj
   data/ShapeDatabase/Bird/...
   ```

## Running

Run all scripts from the project root, in this order:

```
.venv310/bin/python steps/step1.py      # view a shape
.venv310/bin/python steps/step2.py      # statistics, remeshing, normalization
.venv310/bin/python steps/step3.py      # PCA alignment + flipping
.venv310/bin/python steps/step3_2.py    # 3D shape descriptors
```

Each step reads the output of the previous one (see the pipeline below). Settings such as the
target vertex count or the number of histogram bins are in `cbsr/config.py`.

## Project structure

```
cbsr/            the retrieval system (library code)
  config.py        all paths and parameters
  viewer.py        loading and viewing meshes (Step 1)
  shape_stats.py   per-shape statistics (Step 2.1)
  remeshing.py     cleaning, refinement, simplification (Step 2.3)
  normalization.py translation, scaling, PCA alignment, flipping (Steps 2.5, 3.1)
  features.py      3D shape descriptors (Step 3.2)
  pipeline.py      the whole processing for one shape, e.g. a query shape
  mesh_utils.py    shared geometry helpers
analysis/        checks and figures for the report
steps/           one run script per assignment step
data/            shape databases (not in Git): ShapeDatabase/, remeshed/, normalized/, aligned/
results/         CSV tables and figures per step (results/step2/, results/step3/)
docs/            assignment text and technical tips
```

## Pipeline

```
data/ShapeDatabase → data/remeshed → data/normalized → data/aligned → results/step3/features.csv
                     (Step 2.3)      (Step 2.5)        (Step 3.1)      (Step 3.2)
```
