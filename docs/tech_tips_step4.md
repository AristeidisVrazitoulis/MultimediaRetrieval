# Technical tips: Step 4

## 1. Computing distances between two feature vectors

Let **A** = (a₁, …, aₙ) be a feature vector. Typically **A** contains elements that describe different features, each having its own nature. An example is below:

| a₁ | a₂ | a₃ | a₄ | a₅ | a₆ … a₁₅ | a₁₆ … a₂₅ | a₂₆ … a₃₅ | a₃₆ … a₄₅ |
|---|---|---|---|---|---|---|---|---|
| surface area | compactness | AABB volume | diameter | eccentricity | A3 histogram | D1 histogram | D2 histogram | D3 histogram |

Here, we have a feature vector of 45 elements. The first five elements (a₁ … a₅) indicate elementary features, e.g., surface area, compactness, AABB volume, diameter, and elongation. Next, we have 10-bin histograms for the descriptors A3 (a₆ … a₁₅), D1 (a₁₆ … a₂₅), D2 (a₂₆ … a₃₅), and D3 (a₃₆ … a₄₅).

Now consider two such vectors **A** and **B**. How to compute the distance between them? There are several problems of, and solutions for, that.

### (1) Different feature ranges

The first problem relates to the fact that the involved features may have different ranges. As such, simply using the values a₁ … a₄₅ as above could make features which have high ranges count more than features which have low ranges.

A solution for this is to normalize the features so they have the same range. This should be done differently for single-value features and for multiple-value (e.g. histogram) features:

- **Single-value features:** Normalize by standardization, which is less sensitive to outliers than min-max normalization.
- **Histogram features:** Normalize by dividing by the area (element count).

The above two normalizations bring, statistically speaking, both single-value and histogram features in the (absolute-value) range [0, 1]. By *statistically speaking*, we mean that single-value features could still be outside of the [0, 1] absolute-value range for a few samples which have really outlier values. We arguably like to keep these as such, since they describe truly different data values.

### (2) Different distance ranges

You may have noticed that there is an asymmetry in how we normalize single-value and multiple-value features: single-value features are normalized considering the entire set of feature values in the process (when computing the standard deviation). Histogram features are normalized independently per histogram: we don't compute anything like the "standard deviation histogram".

As such, there is a problem: after the normalization (1), the range of the distances between feature values for the different features a₁, a₂, … can be very different. For example:

- The range of eccentricity (elongation) values can span the whole spectrum [0, 1] after normalization. Indeed, our shape database can have all shapes ranging from close to a ball to close to a pencil. Consequently, the range of distances between elongations can also span the whole spectrum [0, 1].
- The D4 histogram values for a set of shapes can be very similar. Hence, the range of distances between D4 histograms (computed e.g. with the EMD metric or any similar histogram distance) can be much smaller than 1. For area-normalized histograms, note that a distance of 1 between two histograms is huge and would likely never be attained in practice.

If we do not address this problem, then features having a small range of distances will count little in the overall distance function.

There are multiple ways to address this problem:

#### Feature weighting

For every set of feature values which, together, form a feature, add a weight wᵢ (so that the sum of all wᵢ equals one). The table below shows this on the example feature vector from the beginning:

| a₁ | a₂ | a₃ | a₄ | a₅ | a₆ … a₁₅ | a₁₆ … a₂₅ | a₂₆ … a₃₅ | a₃₆ … a₄₅ |
|---|---|---|---|---|---|---|---|---|
| w₁ | w₂ | w₃ | w₄ | w₅ | w₆ … w₆ | w₇ … w₇ | w₈ … w₈ | w₉ … w₉ |

Then, adjust the weights wᵢ, relative to each other, so as to boost the variations of the features having small ranges. In practice, this would typically mean leaving w₁ … w₅ equal to each other and making them smaller than the weights w₇ … w₉ used for the narrow-range histograms.

- **Advantage:** very simple to implement.
- **Drawback:** playing around with weights can cost quite some time until the desired results are achieved.

#### Distance weighting

Rather than weighting the feature values, we can go to the "root" of the problem and weigh the distance values themselves. This is exactly what the standardization does for the single-value features: it actually considers the spread of values (by measuring their standard deviation, which is a distance) and normalizes them by this spread.

We can generalize this also for multiple-value features such as histograms. Consider e.g. the feature-vector elements a₆ … a₁₅ which, together, create the D1 descriptor. We can then compute all distances d(a₆ … a₁₅, b₆ … b₁₅) between elements 6 … 15 of two feature vectors **A** and **B** over an entire shape database. These give all possible values for the distances between D1 descriptors over our database, computed by any desired distance function (Euclidean, EMD, cross-bin matching, etc.). Then, we can standardize these distances, just before combining them with the other feature distances (that is, for a₁ … a₅ and a₁₆ … a₄₅) to yield the final distance. This way, even small variations in D1 will count similarly to e.g. large variations in a₁.

> **Note:** This histogram standardization is strongly dependent on the quality of the extracted descriptors. If, for instance, we have a poor computation of D1, which yields more or less the same values for all shapes, then the standardization above will artificially amplify tiny differences in D1 which likely mean nothing, leading to poor matching.
