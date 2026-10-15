@docs/assignment_text_requirement.md

# Multimedia Retrieval: content-based 3D shape retrieval system

University group project (3 people). Given a 3D shape, find and show the most similar shapes in a database of
2,483 OBJ shapes in 69 classes (one folder per class). The work is graded on a written report (at least 25 pages),
the source code, and a live demo. The full assignment is imported above.

## Technical tips from the course (read the relevant file before answering questions about that step)

- `docs/tech_tips_step1.md`: report structure, level of detail (formulas + visual and numerical evidence), outliers
- `docs/tech_tips_step2.md`: remeshing misconceptions, order of normalization steps, face orientation
- `docs/tech_tips_step3.md`: translation / alignment / flipping / size formulas, final step order
- `docs/tech_tips_step3b.md`: mesh volume (needs consistent orientation, closed meshes), checking features
- `docs/tech_tips_step3c.md`: normalizing histograms, sampling A3/D1–D4, sample counts

## Running code

- Python environment: `.venv310/bin/python` (packages pinned in `requirements.txt`, e.g. pandas 2.3.3)
- Each step is a script in `steps/` (`step2.py`, `step3.py`, `step3_2.py`); run from the project root
- `USE_FILTER = True` in a step script = balanced subset for testing; `False` = full database (needed for final results)

## Pipeline

```
ShapeDatabase/ → ShapeDatabase_remeshed/ → ShapeDatabase_normalized/ → ShapeDatabase_aligned/ → stats/features.csv
   (input)        Step 2.3, remeshing.py     Step 2.5, normalizer.py      Step 3.1, alignment.py    Step 3.2, features.py
```

- The database folders are generated and git-ignored. Rerunning a step means rerunning every step after it.
- Order of operations follows tech tips step 3: remeshing → translation → alignment → flipping → size.
- Stats and logs are written to `stats/` (e.g. `original_db_shape_stats.csv`, `remeshed_db_shape_stats.csv`,
  `remeshing_log.csv`, `remeshing_verification.csv`, `normalization_check.csv`, `alignment_check.csv`).

## Remeshing decisions (Step 2.3, `remeshing.py`)

- Target V_target = 6000 vertices, tolerance ±10% → band [5400, 6600]. Shapes inside the band are copied
  unchanged. V_target is chosen from the assignment's 5K–10K guideline, NOT the database average.
- `clean_mesh`: remove unreferenced and duplicate vertices and duplicate faces, then repeat the non-manifold
  edge/vertex repair until 2-manifold, then clear the selection (the repair filters leave faces selected, which
  makes decimation silently do nothing).
- `refine`: area-adaptive midpoint subdivision. Each pass splits only the (V_target − V) longest edges
  (each split adds exactly one vertex). Uniform subdivision was rejected after professor feedback.
- `decimate_to`: quadric edge collapse only takes a FACE target, so we bisect on the face target until the
  VERTEX count is in the band. A fixed F = 2·V_target fails for many-part shapes (it put only 58% in the band).
- Result: about 99% of remeshed shapes end in the band. Rare exceptions (e.g. `Train/D01027`) keep the closest attempt.

## Professor feedback so far (apply in the report)

- Report exactly which functions were used (MeshLab filter names + parameters) and which logic is our own.
- Visual examples: a refined shape, a decimated shape, and a non-manifold shape before/after cleaning.
  Call them Figures, not Tables.
- Automated check over ALL shapes: a histogram of final vertex counts vs the target.
- Explain uniform vs area-adaptive subdivision.

## Report conventions

- Every operation gets a formula, with notation introduced early and used consistently.
- Every step gets visual evidence (before/after renders, wireframe on shaded) and numerical evidence
  (an aggregate or histogram of the property the step should change).
- Number and caption all figures, tables and equations. Cite tools with version + URL (pymeshlab, Open3D)
  and algorithms (e.g. Garland & Heckbert 1997 for quadric decimation).

## Known pitfalls

- Paths in CSVs must use forward slashes. Files generated on Windows contain `ShapeDatabase\Class\file.obj`,
  which breaks loading on macOS. Build paths with `os.path.join` and regenerate such CSVs locally.
- Keep a pymeshlab `MeshSet` in a variable before reading counts: `refine(f).current_mesh().vertex_number()`
  can return 0 because the temporary MeshSet is freed.
- `meshing_surface_subdivision_midpoint` defaults to `iterations=3`; always pass `iterations=1`.
- Face orientation is not yet made consistent (`meshing_re_orient_faces_coherently` is not used). This affects
  volume-based features (volume, compactness, rectangularity, convexity).



## Communication

- Don't use LaTeX math in chat replies; it doesn't render in the VS Code chat. Write formulas as plain text or code.
