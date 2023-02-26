# Deep Seasonal Color Analysis System (DSCAS)
## Introduction
DSCAS is a comprehensive pipeline based on classical and Deep Learning techniques for assigning a color palette to a user according to color harmony principles, given a selfie of the user. The assigned palette, together with a query describing the desired type of clothing, is then used to retrieve compatible clothing items from a database of clothes photos which can be suggested to the user. The project was developed as part of the course on Computer Vision and Cognitive Systems.

## Datasets
+ __Face Segmentation Dataset__: dataset used for segmenting face salient features. The dataset contains 16557 fully pixel-level labeled segmentation images. Facial images are included from different ethnicities, ages and genders making it a well balanced dataset. Also, there is a wide variety of facial poses and different camera angles to provide a good coverage from -90 to 90 degrees face orientations. Click [here](https://store.mut1ny.com/product/face-head-segmentation-dataset-community-edition?v=cd32106bcb6d) to get access to this dataset.
+ __Dress Code Dataset__: dataset for image-based virtual try-on composed of image pairs coming from different catalogs of YOOX NET-A-PORTER, used for the retrieval component of our system. The dataset contains more than 50k high resolution model clothing images pairs divided into three different categories (i.e. dresses, upper-body clothes, lower-body clothes). For our project, we extracted clothing item images from the test partition of the dataset, obtaining 5400 high resolution clothing item images equally divided into the three categories described previously. Click [here](https://github.com/aimagelab/dress-code) to get access to this dataset.

## Usage
For information about composition and execution of the pipeline, see python notebook *pipeline_demo.ipynb*. Please note that you'll need to download the dress code dataset if you want to run the notebook. If you want to get a grasp
of how the system works, you're welcome to read the project report *Deep_Seasonal_Color_Analysis_System__DSCAS.pdf*.

## Project Structure
A brief overview of the most relevant files and directories of the project is presented as a tree structure, where elements are accompanied by a comment
identified by the hashtag character:
```bash
root
├───Deep_Seasonal_Color_Analysis_System__DSCAS.pdf
│   # project paper detailing scope and inner workings of our pipeline.
│
├───dresscode_test_dataset/
│   # directory containing the test partition of the dress code dataset - please note that you'll have  
│   # to download the dataset from the link provided in the Datasets section if you want to run 
│   # the system pipeline.
│
├───headsegmentation_dataset_ccncsa/
│   # directory containing the face segmentation dataset - please note that you'll have to download 
│   # the dataset from the link provided in the Datasets section if you want to train 
│   # face segmentation models yourself.
│
├───class_weights_computation.ipynb
│   # python notebook for computation of class weights used by weighted mIoU and weighted loss when 
│   # evaluating face segmentation models.
│
├───clothes_segmentation_test.py
│   # python file that segments cloth's images from test partition of the dress code dataset and 
│   # visualizes intermediate steps to evaluate the goodness of the segmentation.
│
├───extract_weights_from_checkpoint.py
│   # python script for extraction of weights of pre-trained face segmentation models from their 
│   # corresponding checkpoints.
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
│   # python notebooks for comparison of different preprocessing and data augmentation transforms for 
│   # selected face segmentation models.
│
├───models_test_CGNet_demo.ipynb
├───models_test_Deeplabv3_demo.ipynb
├───models_test_FastSCNN_demo.ipynb
├───models_test_LEDNet_demo.ipynb
├───models_test_UNet_demo.ipynb
│   # python notebooks for evaluation of mIoU of base face segmentation models on test partition of 
│   # face segmentation dataset.
│
├───models_test_FastSCNN_best.ipynb
├───models_test_UNet_best.ipynb
│   # python notebooks for evaluation of mIoU of selected face segmentation models after hpo on 
│   # test partition of face segmentation dataset.
│
├───models_training_best_UNet_complete.ipynb
├───models_training_best_UNet_validation.ipynb
│   # python notebooks for training of UNet face segmentation model after hpo on Google Colab.
│
├───models_training_or_hpo.py
│   # python script for training or hpo of face segmentation models.
│
├───palette_classification_cloth_mappings_computation.ipynb
│   # python notebook for computation and storage of mappings assigning each clothing item of  
|   # dress code dataset to its corresponding palette for retrieval.
│
├───palette_classification_demo_user.ipynb
│   # python notebook showcasing the entire user palette classification process.
│
├───palette_classification_user_thresholds_computation.ipynb
│   # python notebook for computation of user thresholds on training partition of 
│   # face segmentation dataset, used when thresholding metrics contrast, intensity and value during 
│   # the user palette classification process.
│
├───pipeline_demo.ipynb
│   # python notebook showcasing usage and results of our pipeline on a real image.
│
├───retrieval_test.py
│   # python file that tests a given open CLIP model, passed as parameter, over the test partition 
│   # of the dress code dataset.
│
├───retrieval_use_case.py
│   # python file that simulates a retrieval process for a user.
│
├───metrics_and_losses/
│   # python package for metrics and losses defined by us.
│   │
│   ├───metrics.py
│       # python file containing evaluation metrics.
│
├───models/
│   # python package for face segmentation models.
│   │
│   ├───config.py
│   │   # python configuration file for training, evaluation and hpo of face segmentation models.
│   │
│   ├───dataset.py
│   │   # python file containing Dataset classes used to load images from face segmentation and 
│   │   # dress code datasets.
│   │
│   ├───training_and_testing.py
│   │   # python file containing functions for training and evaluation of face segmentation models.
│   │
│   ├───cloud/
│   ├───local/
│   │   # directories containing models' implementation for both local and cloud face segmentation models.
│   │   
│   ├───demo/
│   │   # directory containing checkpoints from training of demo (base) face segmentation models.
│   │
│   ├───training_best/
│   │   # directory containing checkpoints from training of selected face segmentation models after hpo.
│   │
│   ├───hpo/
│   │   # directory containing checkpoints from hpo of face segmentation models.
│   │
│   ├───preprocessing/
│   ├───loss_selection/
│   │   # directories containing checkpoints from preprocessing and loss comparison experiments 
│   │   # of selected face segmentation models.
│   │
│   ├───outputs/
│   │   # directory containing output and error files from execution of scripts to run 
│   │   # training and hpo of face segmentation models on SLURM.
│   │
│   ├───plots/
│   │   # directory containing plots tracking loss and score of face segmentation models during the 
│   │   # training process.
│   │
│   └───weights/
│       # directory containing the weights of all face segmentation models trained by us.
│   
├───palette_classification/
│   # python package for palette classification of user and clothing images.
│   │
│   ├───clothing_palette_mappings/
│   │   # directory containing mappings between clothing item images and their corresponding season 
│   │   # palettes as JSON files,  divided into categories dresses, upper-body and lower-body.
│   │
│   └───palettes/
│       # directory containing season palette CSV files and mappings assigning a unique id 
│       # to each season palette.
│   
├───pipeline/
│   # python package for implementation of system pipeline and included components.
│   
├───retrieval/
│   # python package for clothing segmentation and retrieval.
│   │
│   ├───training_and_testing_retrieval.py
│       # python file containing functions for training and evaluation of retrieval component.
│   
├───slurm_scripts/
│   # python package for scripts to run training and hpo on SLURM.
│   │
│   ├───slurm_config.py
│       # python file containing dictionaries for configuration of training and hpo processes.
│   
└───utils/
    # python package for utility functions and classes.
    │
    ├───custom_transforms.py
    │   # python file containining custom data augmentation transforms for face segmentation models.
    │   
    ├───utils.py
        # python file containing generic functions used throughout the project.
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
