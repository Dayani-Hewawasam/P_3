# Guessing an animal's species from its skeleton

I wanted to see how far I could get predicting an animal's species using only its body posture, with no photo pixels. Each animal in the data is a set of 20 body keypoints (eyes, nose, ears, elbows, knees, paws, tail base), and the task is to say whether it is a dog, cat, sheep, horse or cow.

This is a practice project I built while preparing for a junior data scientist assessment. I wrote it step by step so that I could explain every choice.

## The data

I used `keypoints.json`, which looks like the Animal Pose keypoint dataset.

- 6,117 annotated animals across 4,608 photos
- 5 species: dog (1,771), cat (1,466), sheep (1,078), horse (960), cow (842)
- Each animal has a bounding box and 20 keypoints, each stored as x, y and a visibility flag
- Missing keypoints are stored as `[0, 0, 0]`

Things I noticed when I explored it:

- **Many keypoints are missing.** The median animal has only 12 of 20 keypoints marked, and a quarter have 7 or fewer. Some have none at all. The `num_keypoints` field in the file did not match the number of marked points, so I counted the visible ones myself.
- **Photos can contain many animals.** One photo has up to 16. This matters for how I split the data (see below).
- **The bounding box is corner-to-corner** (`x1, y1, x2, y2`), not x, y, width, height. I checked this by testing which reading puts the keypoints inside the box, and the script picks the right one automatically.
- **The classes are only mildly imbalanced**, roughly 2 to 1 between dogs and cows.

## What I did

### 1. Cleaning
I dropped animals with fewer than 5 visible keypoints, because there is almost nothing to recognize from a couple of dots. Out of 6,117 animals, [fill in: N] were kept. I chose 5 after checking how much data different thresholds would remove: a threshold of 10 would have thrown away roughly 40% of the animals, which seemed too costly.

### 2. Normalization
Raw coordinates mix up posture with how big the animal looks in the photo and where it stands. A close-up dog and a far-away dog have the same posture but very different numbers. So for every animal I:

1. subtracted the center of its bounding box, so every animal is centered at zero
2. divided by the longer side of the box, so every animal has the same size

After this the model only sees the shape.

### 3. Missing keypoints
A missing keypoint is set to 0, but after centering, 0 is also a valid position (the middle of the box). To avoid confusing the two, I added 20 extra 0/1 flags that say whether each keypoint is real. Each animal ends up as 60 numbers: 40 coordinates plus 20 flags.

### 4. Train/test split
I used an 80/20 split that is stratified by species and grouped by photo. Animals from the same photo look alike (same lighting, same herd), so if some ended up in training and some in testing, the model would partly be tested on things it had already seen. Grouping by photo prevents that. The script also has a check that fails if any photo appears on both sides. The split uses a fixed random seed (42), so anyone running the code gets the same numbers. The test set has 1,141 animals.

## Models

1. **Majority class baseline.** Always guesses "dog", the most common species. It is a floor that any real model should beat.
2. **Logistic regression.** A simple linear model that gives each species a score from the 60 numbers.
3. **Random forest.** Hundreds of decision trees that vote. It can pick up patterns that are not straight lines.

For the last two I used balanced class weights, so mistakes on the smaller species count for more.

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Majority class | 0.297 | 0.092 |
| Logistic regression | 0.570 | 0.565 |
| Random forest | 0.642 | 0.641 |

Per species for the random forest:

| Species | Precision | Recall | F1 | Test animals |
|---|---|---|---|---|
| cat | 0.67 | 0.70 | 0.68 | 280 |
| cow | 0.66 | 0.58 | 0.62 | 159 |
| dog | 0.57 | 0.67 | 0.62 | 339 |
| horse | 0.79 | 0.71 | 0.75 | 182 |
| sheep | 0.60 | 0.48 | 0.53 | 181 |

**What I take from this**

- Both real models clearly beat the baseline, so posture does carry information about species.
- The random forest beats logistic regression by about 7 points, which suggests the differences between species are not purely linear.
- Horses were the easiest to recognize (F1 0.75), which fits their distinctive long-legged shape.
- Sheep were the hardest (F1 0.53). The model found fewer than half of the real sheep (recall 0.48). Sheep also have the fewest marked keypoints on average (10.5, against 13.4 for horses), so there is less to work with.
- The most confused pair was [fill in from the confusion matrix, e.g. "sheep predicted as ___"], probably because [fill in your own reason after looking at the matrix]. The full matrix is in `confusion_matrix.png`.
- 64% is a reasonable result for this problem: five fairly similar four-legged animals, 20 keypoints, and many of them missing.

**How I read the metrics**

- Precision: when the model says "sheep", how often is it right?
- Recall: of all the real sheep, how many did the model find?
- F1: one number combining the two.
- Macro F1: the average F1 over the five species, counting each equally, so a weak small class cannot hide behind a strong big one.

## Checking for a shortcut

Horses have more keypoints marked than sheep on average. That could reflect how the labellers worked rather than how the animals look, and a model could pick up on it. To test this, I compared the random forest using different feature sets:

| Features | Accuracy | Macro F1 |
|---|---|---|
| Coordinates + flags (60) | [fill in] | [fill in] |
| Coordinates only (40) | [fill in] | [fill in] |
| Flags only (20) | [fill in] | [fill in] |

[Fill in your conclusion. If "flags only" scores well above the 0.297 baseline, the pattern of which points were labelled reveals species by itself, which is a shortcut worth reporting. Note that "coordinates only" is not a perfectly clean test, because a missing point still shows up as (0, 0).]

## Limitations

- **One split.** With about 1,100 test animals, differences of 1 to 3 points could be noise. I did not use cross-validation or repeat runs with different seeds.
- **Missing keypoints.** Many keypoints are missing, and the pattern of which ones are labelled may leak information about the species.
- **Keypoints only.** The model never sees the image, so it cannot use colour, texture, or context, which a person would use easily.
- **Default settings.** I did not tune the random forest's settings, and I did not try engineered features such as limb-length ratios.

## What I would try next

- Add features like leg length relative to body length, or the angle between limb segments.
- Flip each animal horizontally, swapping left and right keypoints, to get more training data.
- Use cross-validation to get a steadier estimate of the scores.
- Compare dropping animals below 5 visible keypoints with a stricter threshold of 10.

## How to run it

```bash
python -m venv venv
venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib

python step3_prepare.py     # cleans, normalizes, splits, saves data/processed.npz
python step4_train.py       # trains the three models, prints scores, saves confusion_matrix.png
python step5_ablation.py    # compares feature sets
```

Run `step3_prepare.py` before `step4_train.py`, since the training script reads the file the preparation script saves.

## Files

| File | What it does |
|---|---|
| `step1b_keys.py` | Prints the structure of `keypoints.json` |
| `step2_load.py` | Loads the data into a table and prints summary statistics |
| `step3_prepare.py` | Cleaning, normalization, train/test split, saving |
| `step4_train.py` | Trains and evaluates the three models |
| `step5_ablation.py` | Compares coordinates, flags, and both |
| `step_checks.py` | Prints the evidence behind each cleaning and normalization choice |