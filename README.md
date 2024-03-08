[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cMYIlpV6J6Nbng_cYLu4KUTI1AWQq5nz#scrollTo=rnbrFPzQXX3w)

# Semantic Segmentation using U-Net

This repository contains code for performing semantic segmentation using the U-Net architecture. Semantic segmentation is a computer vision task where the goal is to classify each pixel in an image into a specific class. U-Net is a popular convolutional neural network architecture commonly used for image segmentation tasks.

## Getting Started

### Prerequisites

- Python 3.11.x
- TensorFlow
- Keras
- NumPy
- Matplotlib

You can install the required Python packages using pip:

```bash
pip install tensorflow keras numpy matplotlib
```

### Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/semantic-segmentation.git
```

2. Navigate to the project directory:

```bash
cd semantic-segmentation
```

3. Optionally, you may want to set up a virtual environment before installing dependencies.

### Usage

1. **Prepare CamVid dataset**:
   - Download the CamVid dataset from [here](https://www.kaggle.com/datasets/carlolepelaars/camvid).
   - Extract the downloaded dataset, which includes training, validation, and test sets, along with their corresponding label images.
   - Organize the dataset into the following directories:
     - `train`: Contains training images.
     - `train_labels`: Contains labeled images corresponding to the training set.
     - `val`: Contains validation images.
     - `val_labels`: Contains labeled images corresponding to the validation set.
     - `test`: Contains test images.
     - `test_labels`: Contains labeled images corresponding to the test set.
   
2. **Modify paths in the code**:
   - Update the paths in the code to point to the directories where you have stored the CamVid dataset.
   - Modify the `data_dir` variable to specify the root directory of the dataset.

```python
# Define paths to dataset directories
data_dir = '/path/to/camvid_dataset'
train_dir = os.path.join(data_dir, 'train')
train_labels_dir = os.path.join(data_dir, 'train_labels')
val_dir = os.path.join(data_dir, 'val')
val_labels_dir = os.path.join(data_dir, 'val_labels')
test_dir = os.path.join(data_dir, 'test')
test_labels_dir = os.path.join(data_dir, 'test_labels')
class_dict_file = os.path.join(data_dir, 'class_dict.csv')
```

3. **Run the notebook in Google Colab**:
   - Click the button above to open the notebook in Google Colab and execute the code.

4. **Customize the model**:
   - You can modify the U-Net architecture or experiment with different hyperparameters to improve performance.
   - Adjust the input shape, number of classes, and other parameters based on the CamVid dataset requirements.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments
- Special thanks to [CARLO LEPELAARS](https://www.kaggle.com/datasets/carlolepelaars/camvid) for providing the CamVid dataset used in this project.
