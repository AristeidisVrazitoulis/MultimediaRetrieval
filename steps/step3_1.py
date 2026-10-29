import sys
import os
import glob
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbsr.normalization import align_and_flip, check_alignment
from cbsr import config


USE_FILTER = False   #set True to test on the first 30 shapes only

SOURCE_DB = config.NORMALIZED_DB   # output of Step 2.5
ALIGNED_DB = config.ALIGNED_DB     # output of Step 3.1

alignment_check_filename = config.ALIGNMENT_CHECK_CSV

#Step 3.1: align (PCA) + flip (moment test) every shape
files = glob.glob(f"{SOURCE_DB}/*/*.obj")

if USE_FILTER:
    files = files[:30]   #test

print(f"Aligning {len(files)} shapes")
for path in tqdm(files):
    out = path.replace(SOURCE_DB, ALIGNED_DB)
    align_and_flip(path, out)


#check that off-diagonal covariance is close to 0 for the aligned shapes
aligned_files = glob.glob(f"{ALIGNED_DB}/*/*.obj")

import pandas as pd
rows = []
for p in tqdm(aligned_files):
    off_diag = check_alignment(p)
    rows.append({"file": p, "off_diag_covariance": off_diag})

df_check = pd.DataFrame(rows)
os.makedirs(config.STEP3_DIR, exist_ok=True)
df_check.to_csv(alignment_check_filename, index=False)

print(df_check.describe())