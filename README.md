
# T2T-BinFormer Pre-trained Model

This repository hosts the pre-trained model for T2T-BinFormer, an experimental approach to image processing using Transformer models. The project is a beginner's attempt to delve into computer science and image processing. As I am still new to this field, there might be some errors in the implementation.

## Model Link

You can download the pre-trained model from Google Drive: [T2T-BinFormer Model](https://drive.google.com/file/d/1e6cmnE_7z9hwJe3A_xYsoXYOP2cbAJtH/view).

## Installation

Clone this repository to your local machine using:

```bash
git clone https://github.com/RisabBiswas/T2T-BinFormer.git
cd T2T-BinFormer
```

## Usage

### Pre-processing Data

To preprocess the data, run the following command with the appropriate parameters:

```bash
python process_dibco2.py --data_path /root/autodl-tmp/T2T-BinFormer/ --split_size 256 --testing_dataset 2018 --validation_dataset 2016
```

### Training the Model

To train the model with specified parameters, use the command below:

```bash
python train2.py --data_path /root/autodl-tmp/T2T-BinFormer/ --batch_size 32 --vit_model_size base --vit_patch_size 16 --epochs 1000 --split_size 256 --validation_dataset 2016
```

## Results

The PSNR (Peak Signal-to-Noise Ratio) results can be viewed in the `Figure_1.png` file located in this directory.

## Acknowledgements

Special thanks to [Risab Biswas](https://github.com/RisabBiswas) for the initial setup and model configuration provided in his GitHub repository.

## Note

This model training is a preliminary attempt at entering the field of computer science. The process and results might contain errors due to my novice level in this field.
