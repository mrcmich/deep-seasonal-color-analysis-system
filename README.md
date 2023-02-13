# Deep Seasonal Color Analysis System (DSCAS)
## Introduction
DSCAS is a comprehensive pipeline based on classical and AI techniques for assigning a color palette to a user according to color harmony principles, given a selfie of the user. The assigned palette, together with configurable filters, is then used to retrieve compatible clothing items from a database of clothes photos which can be suggested to the user.The project is currently being developed as part of the course on Computer Vision and Cognitive Systems.

## Datasets
{insert description of datasets used and where to find them}

## Usage
For information about the composition and execution of the pipeline, please see python notebook pipeline_demo.ipynb. If you want to get a grasp
of how the system works, you're welcome to read our paper: Deep_Seasonal_Color_Analysis_System__DSCAS_.pdf.

## Project Structure
A brief overview of the most relevant files and directories of the project is presented as a tree structure, where elements are accompanied by a comment
identified by the hashtag character:
```bash
root
│
├───dresscode_test_dataset/
│   # directory containing the test partition of the dress code dataset - please note that you'll have to download 
│   # the dataset yourself from the provided link.
│
├───headsegmentation_dataset_ccncsa/
│   # directory containing the face segmentation dataset - please note that you'll have to download 
│   # the dataset yourself from the provided link.
│
├───class_weights_computation.ipynb
│   # python notebook for computation of class weights used by weighted mIoU and weighted loss when evaluating 
│   # face segmentation models.
│
├───clothes_segmentation_test.py
│   # 
│
├───extract_weights_from_checkpoint.py
│   # python script for extraction of weights of pre-trained face segmentation models from their corresponding checkpoints.
│
├───mean_std.ipynb
│   # python notebook for computation of mean and standard deviation of face segmentation dataset images.
│
├───models_loss_selection_FastSCNN.ipynb
├───models_loss_selection_UNet.ipynb
│   # python notebooks for comparison of standard and weighted loss for selected face segmentation models.
│
├───models_preprocessing_FastSCNN.ipynb
├───models_preprocessing_UNet.ipynb
│   # python notebooks for comparison of different preprocessing and data augmentation transforms for selected 
│   # face segmentation models.
│
├───models_test_CGNet_demo.ipynb
├───models_test_Deeplabv3_demo.ipynb
├───models_test_FastSCNN_demo.ipynb
├───models_test_LEDNet_demo.ipynb
├───models_test_UNet_demo.ipynb
│   # python notebooks for evaluation of mIoU of base face segmentation models on test partition of face segmentation dataset.
│
├───models_test_FastSCNN_best.ipynb
├───models_test_UNet_best.ipynb
│   # python notebooks for evaluation of mIoU of selected face segmentation models after hpo on test partition of 
│   # face segmentation dataset.
│
├───models_training_best_UNet_complete.ipynb
├───models_training_best_UNet_validation.ipynb
│   # python notebooks for training of UNet face segmentation model after hpo on Google Colab.
│
├───models_training_or_hpo.py
│   # python script for training or hpo of face segmentation models.
│
├───palette_classification_cloth_mappings_computation.ipynb
│   # python notebook for computation and storage of mappings assigning each clothing item of dress code dataset 
|   # to its corresponding palette for retrieval.
│
├───palette_classification_demo_user.ipynb
│   # python notebook showcasing the entire user palette classification process.
│
├───palette_classification_user_thresholds_computation.ipynb
│   # python notebook for computation of user thresholds on training partition of face segmentation dataset, 
│   # used when thresholding metrics contrast, intensity and value during the user palette classification process.
│
├───pipeline_demo.ipynb
│   # python notebook showcasing usage and results of our pipeline on a real image.
│
├───retrieval_test.py
│   # 
│
├───retrieval_use_case.py
│   # 
│
├───metrics_and_losses/
│   ├───metrics.py
│       # python file containing evaluation metrics.
│
├───models/
│   # directories and files related to selection, training, hpo and other experiments of face segmentation models.
│   │
│   ├───config.py
│   │   #
│   │
│   ├───dataset.py
│   │   # python file containing Dataset classes used to load images from face segmentation and dress code datasets.
│   │
│   ├───training_and_testing.py
│   │   #
│   │
│   ├───cloud/
│   ├───demo/
│   ├───hpo/
│   ├───local/
│   ├───loss_selection/
│   ├───outputs/
│   ├───plots/
│   ├───preprocessing/
│   ├───training_best/
│   └───weights/
│   
├───palette_classification/
│   ├───color_processing.py
│   ├───palette.py
│   ├───clothing_palette_mappings/
│   ├───example_images/
│   └───palettes/
│   
├───pipeline/
│   
├───retrieval/
│   ├───clothes_segmentation.py
│   ├───training_and_testing_retrieval.py
│   ├───retrieved_images/
│   └───segmented_clothes/
│   
├───slurm_scripts/
│   ├───slurm_config.py
│   ├───hpo/
│   ├───training_best/
│   └───training_demo/
│   
└───utils/
    ├───custom_transforms.py
    ├───model_names.py
    ├───segmentation_labels.py
    ├───utils.py
```

## Authors

- [Francesco Baraldi](https://github.com/francescobaraldi)
- [Matteo Pagliani](https://github.com/MatteoPagliani)
- [Me](https://github.com/mrcmich)

## References

- [Fast-SCNN: Fast Semantic Segmentation Network](https://github.com/Tramac/Fast-SCNN-pytorch), [Tramac](https://github.com/Tramac).
- [CGNet: A Light-weight Context Guided Network for Semantic Segmentation](https://github.com/wutianyiRosun/CGNet), [wutianyiRosun](https://github.com/wutianyiRosun).
- [LEDNet: A Lightweight Encoder-Decoder Network for Real-time Semantic Segmentation](https://github.com/xiaoyufenfei/LEDNet), [xiaoyufenfei](https://github.com/xiaoyufenfei).
- [U-Net for brain segmentation](https://github.com/mateuszbuda/brain-segmentation-pytorch), [mateuszbuda](https://github.com/mateuszbuda).
- [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915), [Pytorch](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/).
