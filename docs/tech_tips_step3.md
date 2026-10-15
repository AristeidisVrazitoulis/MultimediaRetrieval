# Technical tips: Step 3

*Source: 03-TechTips.pdf*

## 1. Normalization Tips

Recall, normalization includes making the models similar in terms of various properties: position, size, alignment, orientation. Some useful tips about all these steps below:

### (1) Position

This should really be very simple: If you want to make all shapes centered in the origin (that is, their barycenter **c** should be located at the origin of the coordinate frame), then the *update formula* for the coordinates **p_i** is

```
p_i_updated = p_i - c
```

Proof: Simple. Put **c** instead of **p_i** in the above formula, and you'll see that p_i_updated = 0.

Important note: All above quantities are *vectors*. So, the update is to be done coordinate-wise (for x, y, z separately).

### (2) Alignment

As discussed in the practical session, a very simple way to align a shape's eigenvectors with the xyz coordinate frame is to use *projections*. Let again **p_i** be the unaligned shape coordinates. Let **e_j** be the eigenvectors computed from the shape (here, j ∈ {1,2,3}). Let **e1**, **e2** be the major, respectively medium, eigenvectors. Assume next these are normalized (of length 1). Then the *update formula* for computing the aligned shape is

```
x_i_updated = (p_i - c) · e1
y_i_updated = (p_i - c) · e2
z_i_updated = (p_i - c) · (e1 × e2)
```

Here, · indicates dot product and × indicates cross product. So, why don't we use **e3** in the update formula for **z** above? Because, recall, PCA gives us only *directions*, not *orientations*. So, we cannot be sure that PCA outputs three vectors e1, e2, e3 that form a right-handed coordinate system like the xyz one is. So, we 'force' this by using e1 × e2 instead of e3.

**Subtle point:** So, since PCA doesn't give us orientations, but just directions, what about e1 and e2? That is, we 'fixed' e3 by replacing it with e1 × e2. But what about e1 and e2? These can still be 'flipped', right? Yes, this is the case. But the flipping test (orientation normalization) discussed below will fix this! So, at this point, we don't care about the orientation of e1 and e2.

### (3) Size

This is really very simple (see course slides). Compute the sizes D_x, D_y, D_z of the *oriented bounding box* that encloses the shape. Or, easier, compute the *axis-aligned bounding box* sizes D_x, D_y, D_z after alignment. Obviously, the above two operations yield the same result. Then, compute the largest dimension of the bounding box:

```
D_max = max(D_x, D_y, D_z)
```

Then, compute the scaling factor that would *shrink* this size to one:

```
σ = 1 / D_max
```

Then, the *update formula* is very simple:

```
p_i_updated = σ p_i
```

### (4) Orientation ('flipping test')

The idea is very simple here too: For each axis (xyz) of the already axis-aligned shape, we compute in which part of the axis (positive half, negative half) most of the 'mass' (area, volume) of the shape resides. Then, we choose a conventional orientation – we want to make the most mass be in the positive half-axis. So, if this is already the case (indicated by the flipping test), then we don't do anything. If the mass is mostly in the negative half-axis, we mirror the shape along that axis. Note that mirroring is identical to scaling (see above) with a scaling factor equal to -1.

In detail: The flipping test computes a value f_i along each axis (that is, i=0 means x, i=1 means y, i=2 means z) as:

```
f_i = Σ_t  sign(C_t,i) · (C_t,i)²
```

where the summation goes over all triangles t in the mesh, and C_t,i is the i-th coordinate of the center of triangle t.

Then, the *update formula* is very simple: mirror the mesh using the following scaling factors:

```
x_i_updated = x_i · sign(f_0)
y_i_updated = y_i · sign(f_1)
z_i_updated = z_i · sign(f_2)
```

### (5) Order of normalization steps

Refining the ideas from the last technical tips document: What is a *good* order of all operations?

1. Remeshing
2. Translation
3. Pose (alignment)
4. Flipping (mirroring)
5. Size

Why this? Look at the actual formulas and think what would happen if you did these steps in other orders.

### (6) Other normalization tips

**Should you use an oriented bounding box (OBB) or axis aligned bounding box (AABB) to do the size normalization?**
If you do this normalization *after* the pose alignment (as suggested above), the answer is that it does not matter, since after alignment, the OBB and AABB of a shape are identical.

**What is a good example of using histograms to check normalization?**
See the images below (taken from an actual student report). The first image shows the histogram of face areas *before resolution normalization* (i.e., before remeshing). We see that most faces are small, but there exist also larger faces. The 'bandwidth' of the distribution, i.e. difference between the minimal and maximal face areas, is quite large: from 1 to 50, roughly.

*[Figure in the PDF: "Face area distribution" bar chart before remeshing. X axis: face area (×1e-05) in 10 bins from 0–5.5 up to 49–55; Y axis: % of faces. The bars decrease steadily from about 0.43 in the first bin to near 0 in the last bins.]*

The next image shows the histogram of face areas *after resolution normalization* (i.e., after remeshing). We see two desirable things:

- The bandwidth of the distribution has considerably shrunk – from 1..50 to roughly 0..1. Hence, face sizes are far more uniform after this normalization
- The distribution is centered around an average face-area value of 0.5. *If this is our desired target face-area value, then this is a good distribution.*

*[Figure in the PDF: "Face area distribution" bar chart after remeshing. X axis: face area (×1e-05) in 10 bins from 0–0.1 up to 0.94–1; Y axis: % of faces. A narrow peak around 0.42–0.63 (bars of about 0.32 and 0.34), with small bars elsewhere.]*
