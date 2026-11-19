# Multimedia Retrieval: Content-Based 3D Shape Retrieval

Given a 3D shape, the system finds and shows the most similar shapes in a database of 2483 shapes
in 69 classes. The report describes every step; this file explains how to install and run the code.

## Setup

1. Create a virtual environment with **Python 3.10** and install the dependencies:

   ```
   python3.10 -m venv .venv310
   .venv310/bin/pip install -r requirements.txt          # macOS / Linux
   .venv310\Scripts\pip install -r requirements.txt      # Windows
   ```

   In the commands below, use `.venv310/bin/python` on macOS / Linux and `.venv310\Scripts\python`
   on Windows.

2. Download the shape databases (they are too large for GitHub) and extract them into `data/`:

   | Folder | Content | Download |
   |---|---|---|
   | `data/ShapeDatabase/` | original database (input of Step 2) | [Google Drive](https://drive.google.com/file/d/1157pMhQoN9G6W7nREW9TCbbP8z7VBU0s/view?usp=sharing) |
   | `data/remeshed/` | after remeshing (Step 2.3) | [Google Drive](https://drive.google.com/file/d/1k6LDjYVz1sB5Xf8iwJj19XQs6KE1emtA/view?usp=sharing) |
   | `data/normalized/` | after translation and scaling (Step 2.5) | [Google Drive](https://drive.google.com/file/d/1foEqUaPPOXkNcVpnDjCtUxvY54yp-i4w/view?usp=sharing) |
   | `data/aligned/` | fully normalized shapes (Step 3.1), used by Steps 3.2-6 and the app | [Google Drive](https://drive.google.com/file/d/1BP7-OBeisDXsAjSDC80Rck5mA41lpzuH/view?usp=sharing) |

   After extracting, every folder must contain one subfolder per class, for example:

   ```
   data/ShapeDatabase/Bicycle/D00016.obj
   data/aligned/Bicycle/D00016.obj
   ```

   Google Drive may warn that the files are too large to be scanned for viruses; choose
   "Download anyway". To only run the app and the Steps 4-6, `data/aligned/` is enough.

The feature database (`results/step3/features.csv`) and all results in `results/` are in the
repository, so after downloading `data/aligned/` the app can be started directly (see below)
without running the steps again.

## Running the steps

Run every script from the project root, in this order. Each step reads the output of the
previous one, so after changing a step all the following steps have to be run again.

```
python steps/step1.py       # Step 1: view a shape
python steps/step2.py       # Step 2: statistics, remeshing, normalization
python steps/step3_1.py     # Step 3.1: PCA alignment, flipping, final scaling
python steps/step3_2.py     # Step 3.2: shape descriptors -> results/step3/features.csv (about an hour)
python steps/step4.py       # Step 4: feature normalization, distance weighting, comparison of the settings
python steps/step5.py       # Step 5: kd-tree, comparison with Step 4, t-SNE maps
python steps/step6.py       # Step 6: evaluation (precision@k, ROC, improved setting)
python steps/step6_figs.py  # Step 6: figures of the report
```

Options are set at the top of each script:

- `USE_FILTER` (steps 2 and 3.1): `True` runs on a small balanced subset for testing, `False` on the
  whole database (needed for the final results).
- `COMPUTE_FEATURES` (step 3.2): `False` reuses `results/step3/features.csv` if it exists.
- `CHECK_BINS` (step 3.2): `True` runs the test of the number of bins and of the D1 samples
  (`results/step3/bin_check.csv`, `d1_points_check.csv`). The first run takes about an hour; the
  recomputed histograms are then cached in `data/bin_check_histograms.npz`.
- `MAKE_QUERY_TABLE`, `COMPARE_SEARCHES`, `MAKE_TSNE`, `MAKE_CLASS_PRECISION` (step 5): which parts
  to run. For meaningful query times, run `COMPARE_SEARCHES` alone on an idle computer.

All parameters (target vertex count, number of bins, samples, seed, histogram ranges, default
distance setting, kd-tree and t-SNE settings) and all paths are in `cbsr/config.py`.

## Querying tool

```
python -m streamlit run app.py
```

The app opens in the browser. In the sidebar, choose the query (a database shape or a new OBJ file,
which goes through the whole pipeline of Steps 2 and 3), the search (Step 4 distance, kd-tree with k
nearest, or kd-tree within a distance R) and the distance setting. The tabs show the results, the
normalized feature database, the t-SNE map and the evaluation. The app reads the parameters stored
by `steps/step4.py` (`results/step4/prepared_params.json`); if they are missing or do not match
`features.csv`, it computes them at start-up and shows a warning.

## Project structure

```
app.py           querying tool (Steps 4 and 5)
cbsr/            the retrieval system (library code)
  config.py        all paths and parameters
  viewer.py        loading and viewing meshes (Step 1)
  shape_stats.py   per-shape statistics (Step 2.1)
  remeshing.py     cleaning, refinement, simplification (Step 2.3)
  normalization.py translation, scaling, PCA alignment, flipping (Steps 2.5, 3.1)
  features.py      3D shape descriptors (Step 3.2)
  retrieval.py     feature normalization, block distances, distance weighting, queries (Step 4)
  scalability.py   kd-tree index and t-SNE map (Step 5)
  pipeline.py      the whole processing of one shape, e.g. a new query shape
  mesh_utils.py    shared geometry helpers
analysis/        checks and figures for the report (not used by the app)
steps/           one run script per assignment step
data/            shape databases: ShapeDatabase/, remeshed/, normalized/, aligned/
results/         CSV tables and figures per step (results/step2/ ... results/step6/)
docs/            assignment text and technical tips of the course
```

## Pipeline

Each step reads the output of the previous one:

| Step | Reads | Writes |
|---|---|---|
| 2.3 Remeshing | `data/ShapeDatabase/` | `data/remeshed/` |
| 2.5 Translation and scaling | `data/remeshed/` | `data/normalized/` |
| 3.1 Alignment, flipping, scaling | `data/normalized/` | `data/aligned/` |
| 3.2 Descriptors | `data/aligned/` | `results/step3/features.csv` |
| 4 Normalization and distance weighting | `results/step3/features.csv` | `results/step4/prepared_params.json` |
| 5 kd-tree and t-SNE | `features.csv`, `prepared_params.json` | `results/step5/` |
| 6 Evaluation | `features.csv`, `prepared_params.json` | `results/step6/` |

The app (`app.py`) reads `data/aligned/`, `features.csv` and `prepared_params.json`.
