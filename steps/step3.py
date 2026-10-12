import sys
import os
import glob
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alignment import align_and_flip, check_alignment


USE_FILTER = True   # set False for the full database run

SOURCE_DB = "ShapeDatabase_normalized"   # output of Step 2.5
ALIGNED_DB = "ShapeDatabase_aligned"     # output of Step 3.1

STATS_SOURCE_FOLDER = "stats/"
alignment_check_filename = STATS_SOURCE_FOLDER + "alignment_check.csv"


#Step 3.1: align (PCA) + flip (moment test) every shape
files = glob.glob(f"{SOURCE_DB}/*/*.obj")

if USE_FILTER:
    files = files[:30]   #test

print(f"Aligning {len(files)} shapes")
for path in tqdm(files):
    out = path.replace(SOURCE_DB, ALIGNED_DB)
    align_and_flip(path, out)


#check that off-diagonal covariance is close to 0 for the aligned shapes
print("Verifying alignment...")
aligned_files = glob.glob(f"{ALIGNED_DB}/*/*.obj")

import pandas as pd
rows = []
for p in tqdm(aligned_files):
    off_diag = check_alignment(p)
    rows.append({"file": p, "off_diag_covariance": off_diag})

df_check = pd.DataFrame(rows)
os.makedirs(STATS_SOURCE_FOLDER, exist_ok=True)
df_check.to_csv(alignment_check_filename, index=False)

print(df_check.describe())