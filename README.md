# Sound Source Tracking (SST) Evaluation Pipeline

## Overview

This repository provides a Python pipeline for evaluating Sound Source Tracking (SST) systems. Given a multichannel audio recording, a SST system estimates the time-varying positions of sound sources.
This toolkit enables the quantitative evaluation of such systems using standard tracking metrics covering detection, localization, and association objectives.

The repository also includes utilities to read and process items from the [LibriJump SST evaluation dataset](https://zenodo.org/records/15791948) [1], as well as a Python adaptation of the [LOCATA challenge metrics](https://github.com/cevers/sap_locata_eval) [3].

---

## Scope and Assumptions

- Source positions are represented as time-varying Direction of Arrival (DOA) using azimuth and elevation angles.
- Silent source frames are represented by NaN values.
- Distances between predictions and ground truths are computed using angular distance.

---

## Installation

```bash
pip install git+https://github.com/Orange-OpenSource/sstracking_metrics
```

---

## Metrics Overview

Given ground-truth and predicted tracks, the following SST objectives are evaluated:

### Detection

- Detection accuracy
- Precision
- Recall

### Localization

- Mean localization error
- Median localization error
- Localization accuracy

### Association

- Association accuracy
- Precision
- Recall

Additionally, a Python implementation of the [LOCATA challenge metrics](https://github.com/cevers/sap_locata_eval) [3] is provided in:

```
./tracking_metrics/locata.py
```

Readers are referred to [1], [2], and [3] for detailed theoretical background on the metrics.

---

## Notes on SST Metrics

- All metrics rely on a matching step between ground-truth and predicted tracks to identify:
  - True Positives (TPs)
  - False Positives (FPs)
  - False Negatives (FNs)
- An optional gating mechanism can be applied during matching to discard predictions that are too far from ground-truth positions. Predictions exceeding a distance threshold are counted as FPs.
- Localization and association metrics are computed only on TPs.
- Localization accuracy is defined as the percentage of predictions located within a given threshold from the ground truth.
- Association metrics correspond to a subset of the Higher-Order Tracking Accuracy (HOTA) framework [2].

---

## Evaluation Dataset

This repository includes tools to read and process items from the LibriJump SST evaluation dataset [1].

The dataset can be downloaded from the [official LibriJump repository](https://zenodo.org/records/15791948).

---

## Repository Structure

```
.
├── src/tracking_metrics/    # SST metrics implementations
├── dummy_examples/          # Example of YAML metric outputs
├── dummy_example_metrics.py # Minimal fabricated example
├── dummy_example.py         # Example using the LibriJump dataset
```

---

## Usage

Two example scripts are provided to demonstrate how to run the metrics:

- `dummy_example_metrics.py`
  - Minimal example using fabricated data
- `dummy_example.py`
  - Similar evaluation pipeline applied to LibriJump dataset items

Both scripts generate YAML files containing metric results in the `./dummy_examples` directory.

## Environment Setup

It is recommended to use a dedicated virtual environment before installing the package, either using conda/mamba

```bash
conda create -n sstracking_metrics python=3.12
conda activate sstracking_metrics
```

or a classic venv

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## Citation

If you use this code or the provided metrics in your research, please cite:

[1] Iatariene, T., Guérin, A., & Serizel, R. (2025). Tracking of Intermittent and Moving Speakers: Dataset and Metrics. Proceedings of the 11th Convention of the European Acoustics Association, Forum Acusticum 2025.

---

## References

[2] Luiten, J., Osep, A., Dendorfer, P., Torr, P., Geiger, A., Leal-Taixé, L., & Leibe, B. (2021). HOTA: A higher order metric for evaluating multi-object tracking. International Journal of Computer Vision, 129(2), 548–578.

[3] Evers, C., Löllmann, H. W., Mellmann, H., Schmidt, A., Barfuss, H., Naylor, P. A., & Kellermann, W. (2020). The LOCATA challenge: Acoustic source localization and tracking. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28, 1620–1643.

---

## Licenses

This project is licensed under the MIT License.
It uses third-party libraries and datasets under their respective licenses.
See THIRD_PARTY_LICENSES.txt for details.
