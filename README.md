# ML Practice

## Requirements

Install **Python**, then install the required libraries:

```bash
pip install pandas
pip install scikit-learn
```

## How to Use

Each experiment generally contains Python files for **using** and **experimenting with** the models.

### Using Your Own Data

There is generally a file with a name such as:

```text
useOnly.py
use-only.py
use_only.py
```

The exact name may vary between experiments.

This file is used to **load/use the trained model with your own data and get predictions**.

Run it with:

```bash
python <use-file-name>.py
```

### Training / Experimenting

There is generally another file such as:

```text
main.py
dev.py
```

or a similarly named development/experiment file.

Use this file to:

* Train models
* Experiment with different models
* Improve model performance
* Compare results
* Save trained models

Run it with:

```bash
python <filename>.py
```

## Typical Workflow

1. Install Python.
2. Install the required libraries.
3. Open the experiment folder.
4. Run the main/development file to train or experiment with the models.
5. Run the `useOnly`-type file to use the trained model with your own data.
6. Follow the inputs requested by the script.
7. Check the predictions/results.

> **Note:** The exact filenames can differ between experiments, but the `useOnly`-type file is generally intended for using the trained model with your own data.
