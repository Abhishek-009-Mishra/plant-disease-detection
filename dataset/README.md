# Dataset Folder

This folder is intentionally empty in version control (see `.gitignore`).

Download the **PlantVillage** dataset and point the project at it in one of two ways:

**Option A - environment variable (recommended):**
Create a `.env` file in the project root (copy `.env.example`) and set:

```
PLANTVILLAGE_DATASET_ROOT=D:/PlantVillage
```

**Option B - edit the default directly:**
Open `src/config.py` and change the `DATASET_ROOT` default.

## Expected structure

```
<DATASET_ROOT>/
├── color/
│   ├── Apple___Apple_scab/
│   ├── Apple___healthy/
│   ├── Tomato___Early_blight/
│   └── ...
├── grayscale/
│   └── (same class folders)
└── segmented/
    └── (same class folders)
```

You do not need all three representations - the system automatically
detects which ones are present.
