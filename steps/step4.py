import sys
import os
import itertools
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbsr import config
from cbsr.retrieval import load_database, prepare_database, save_prepared, normalized_feature_table
from analysis.retrieval_checks import compare_configurations


STORE_NORMALIZED_FEATURES = True   # results/step4/normalized_features_<method>.csv (the assignment asks to store them)
STORE_PREPARED_PARAMS = True       # results/step4/prepared_params.json, loaded by the app (rerun after new features!)
COMPARE_CONFIGURATIONS = True      # results/step4/distance_comparison.csv: precision@5 of every configuration (~2 min)
K = 5


# every combination the app offers, with the log transform and distance weighting on
APP_CONFIGS = [dict(norm_method=n, scalar_distance=s, hist_distance=h)
               for n, s, h in itertools.product(config.NORMALIZATION_METHODS, config.SCALAR_DISTANCES,
                                                config.HIST_DISTANCES)]
# the same without the log transform / without distance weighting, to show what each contributes
ABLATION_CONFIGS = [dict(norm_method=n, scalar_distance=config.DEFAULT_SCALAR_DISTANCE, hist_distance=h, **off)
                    for n, h in itertools.product(config.NORMALIZATION_METHODS, config.HIST_DISTANCES)
                    for off in (dict(use_log=False), dict(weighting=False))]


def store_normalized_features(db):
    os.makedirs(config.STEP4_DIR, exist_ok=True)
    for method in config.NORMALIZATION_METHODS:
        prep = prepare_database(db, norm_method=method)
        out = config.NORMALIZED_FEATURES_CSV.format(method=method)
        normalized_feature_table(db, prep["params"]).to_csv(out, index=False)
        print("saved", out)


def store_prepared_params(db):
    preps = [prepare_database(db, **c) for c in APP_CONFIGS]
    save_prepared(preps)
    print(f"saved {config.PREPARED_PARAMS_JSON} ({len(preps)} configurations)")


def compare(db):
    table = compare_configurations(db, APP_CONFIGS + ABLATION_CONFIGS, prepare_database, K)
    table.to_csv(config.DISTANCE_COMPARISON_CSV, index=False)
    print(table.to_string(index=False))
    print("saved", config.DISTANCE_COMPARISON_CSV)


if __name__ == "__main__":
    db = load_database()
    if STORE_NORMALIZED_FEATURES:
        store_normalized_features(db)
    if STORE_PREPARED_PARAMS:
        store_prepared_params(db)
    if COMPARE_CONFIGURATIONS:
        compare(db)
