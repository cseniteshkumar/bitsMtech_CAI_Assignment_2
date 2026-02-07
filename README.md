# ADL Assignment 2

This is the assignment project for Advanced Deep Learning (ADL) Assignment 2.

## Project Overview

This project implements deep learning techniques for image analysis using the CelebA (Celebrity Attributes) dataset. The assignment explores various aspects of deep learning including model architecture, training, and evaluation on facial attribute recognition tasks.

## Dataset

The project uses the **CelebA Dataset** (Large-scale CelebFaces Attributes Dataset):
- **Images**: Located in `CelebA/img_align_celeba/`
- **Attributes**: `list_attr_celeba.csv` - Binary attributes for each celebrity image
- **Bounding Boxes**: `list_bbox_celeba.csv` - Face bounding box coordinates
- **Landmarks**: `list_landmarks_align_celeba.csv` - Facial landmark positions
- **Partitions**: `list_eval_partition.csv` - Train/validation/test split information

## Project Structure

```
bitsMtech_ADL_Assignment_2/
├── ADL_assign2_v4_VT_Jan30.ipynb    # Main assignment notebook
├── requirement.txt                   # Project dependencies
├── README.md                         # Project documentation
├── bckUp/                           # Backup files
│   └── ADL_assign2_v4_VT_Jan30.ipynb
└── CelebA/                          # Dataset directory
    ├── list_attr_celeba.csv
    ├── list_bbox_celeba.csv
    ├── list_eval_partition.csv
    ├── list_landmarks_align_celeba.csv
    └── img_align_celeba/            # Celebrity images
```

## Requirements

Install the required dependencies using:

```bash
pip install -r requirement.txt
```

## Usage

1. **Setup Environment**: Ensure all dependencies are installed
2. **Dataset**: Verify the CelebA dataset is present in the `CelebA/` directory
3. **Run Notebook**: Open and execute `ADL_assign2_v4_VT_Jan30.ipynb` in Jupyter Notebook or JupyterLab

```bash
jupyter notebook ADL_assign2_v4_VT_Jan30.ipynb
```

## Assignment Details

- **Course**: Advanced Deep Learning (ADL)
- **Assignment**: Assignment 2
- **Institution**: BITS Mtech Program

## Notes

- Ensure sufficient GPU memory is available for training deep learning models
- The CelebA dataset contains 202,599 celebrity images with 40 binary attributes each
- Backup files are maintained in the `bckUp/` directory

## License

This project is for educational purposes as part of the BITS Mtech ADL course.
