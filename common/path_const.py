import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
APP_ROOT = os.path.join(ROOT, "rstudio")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
ORG_TRAIN = os.path.join(INPUT_DIR, "train.csv")
ORG_TEST = os.path.join(INPUT_DIR, "test.csv")
ORG_STORE = os.path.join(INPUT_DIR, "store.csv")
PROPER_TRAIN = os.path.join(INPUT_DIR, "train_proper.csv")
PROPER_TEST = os.path.join(INPUT_DIR, "test_proper.csv")
TRAIN1 = os.path.join(INPUT_DIR, "train1.csv")
TEST1 = os.path.join(INPUT_DIR, "test1.csv")
SUB = os.path.join(OUTPUT_DIR, "sub.csv")

TRAIN2 = os.path.join(INPUT_DIR, "train2.csv")
TEST2 = os.path.join(INPUT_DIR, "test2.csv")

TRAIN_SMALL = os.path.join(INPUT_DIR, "train_small.csv")
TEST_SMALL = os.path.join(INPUT_DIR, "test_small.csv")
TRAIN_SMALL2 = os.path.join(INPUT_DIR, "train_small2.csv")
TEST_SMALL2 = os.path.join(INPUT_DIR, "test_small2.csv")