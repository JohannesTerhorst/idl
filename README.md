# Introduction to Deep Learning (IN2346) — WS 26/27

Working copy for the i2dl exercises. Course material (slides, exam prep) stays in
OneDrive under `30 WiSe 26_27/20 Introduction to Deep Learning`; the pristine
exercise zips stay in `10 Exercise/` there. This repo is only the working copy.

## Layout

The course's notebooks assume this exact sibling structure — `datasets/` and
`output/` next to the exercise folders. Do not rename `exercise_NN`:

    idl/                        <- repo root == the course's "i2dl" root
    ├── datasets -> ~/data/idl  <- symlink, gitignored, auto-populated
    ├── output/                 <- submission zips, gitignored
    ├── exercise_01/
    │   ├── 1_introduction.ipynb
    │   ├── exercise_code/      <- goes into the submission
    │   ├── images/
    │   ├── models/             <- save_model() writes here, gitignored
    │   └── requirements/
    └── exercise_02 … 11        <- unzip alongside as they are released

## Running an exercise

Always run with the **exercise folder as cwd** — the notebook opens
`images/sample_image.jpg` and writes `models/` by relative path:

    source .venv/bin/activate
    cd exercise_01
    jupyter notebook 1_introduction.ipynb   # or just open it in VSCode

The last cell calls `submit_exercise('../output/exercise_01')`, which zips
`exercise_code/`, `models/` and the notebooks to `output/exercise_01.zip`.
Upload that at https://i2dl.vc.in.tum.de/ (register with your matriculation
number). Exercise 01 threshold: 60. Re-submission allowed until the deadline;
best score counts.

## On a fresh machine

    git clone <this repo> && cd idl && ./bootstrap.sh

`bootstrap.sh` recreates the two things git deliberately does not carry: the
`datasets` symlink and `.venv`. Datasets are auto-downloaded by the notebooks,
so nothing large ever moves between machines.

## Notes

- **No GPU needed until exercise 06.** Exercise 01's requirements contain no
  torch at all — just jupyter, numpy, matplotlib, pandas, seaborn. The README
  states exercises 1–5 need no acceleration.
- The course prescribes conda with python 3.11. `bootstrap.sh` uses uv (or a
  plain venv) against the same pinned `requirements.txt`, which is equivalent
  for grading — the submission server only ever sees your zip.
- The shipped `requirements.txt` files are **UTF-16 encoded**. pip decodes them
  fine (verified), but they will look like binary to grep and most editors.
- The notebook's version check prints "we expect python 3.10.x" while the README
  says 3.11. Harmless — it only warns.
