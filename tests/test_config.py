from src.config import (
    PROJECT_ROOT,
    RAW_DIR,
    PROCESSED_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    TRAIN_CSV,
    TEST_CSV,
    RANDOM_STATE,
    N_SPLITS,
)


def test_paths_exist():
    assert PROJECT_ROOT.exists()
    assert RAW_DIR.exists()
    assert PROCESSED_DIR.exists()
    assert FIGURES_DIR.exists()
    assert TABLES_DIR.exists()


def test_dataset_files_exist():
    assert TRAIN_CSV.exists()
    assert TEST_CSV.exists()


def test_constants():
    assert RANDOM_STATE == 42
    assert N_SPLITS == 4
