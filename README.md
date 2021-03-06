# Tabular data generation using Generative Adversarial Networks (GANs) and Wasserstein GANs (WGANs)

GANs and WGANs are two types of state-of-the-art algorithms to generate data from scratch. Generating data allows us, to some extend, training machine learning (ML) models results in better performance when suffering from a lack of original data. While GANs are widely used for images data, here we'll utilize GANs power to handle tabular data. This project aims to provide a clear and simple implementation (Pytorch) to GANs and WGANs. The former presents the basic idea that presented in the paper ["Generative adversarial networks"](https://arxiv.org/abs/1406.2661) (Goodfellow et al., 2014). The latter, ["Wasserstein generative adversarial networks"](http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf) (Arjovsky et al., 2017), stands as an extension of the GAN while demonstrating better training stability, less sensitivity for hyperparameters and model architecture, and handle categorical data (as in our case). As an example, we will use the ["Pima Indians Diabetes Database"](https://www.kaggle.com/uciml/pima-indians-diabetes-database) collected from Kaggle. 

## Getting Started

Clone the repository to your local machine.

### Prerequisites

Install the requirements file with 

```
pip install -r requirements.txt
```

### Running example

Here we will use as an example the popular dataset "diabetes" downloaded from Kaggle. Run the main.py file with the following configuration via terminal or the IDE configuration:

#### Running the GAN model

```
--algorithm GAN --data-set data/diabetes_dataset/diabetes.csv --epochs 500 --train
```

#### Running the WGAN model

```
--algorithm WGAN --data-set data/diabetes_dataset/diabetes.csv --epochs 200 --train
```

## Results and further discussions

As mentioned above, WGAN manages to handle categorical variables where GAN is not originally designed for - and suffers from what's called ["Mode Collapse"](https://developers.google.com/machine-learning/gan/problems#mode-collapse). In the following histograms we can see both algorithm performance where the WGAN succeded to reproduce the categorical feature "Outcome", where the simple GAN fails.

GAN                        |  WGAN
:-------------------------:|:-------------------------:
![](./images/GAN/Age_distribution.png)                      |  ![](./images/WGAN/Age_distribution.png)
![](./images/GAN/BMI_distribution.png)                      |  ![](./images/WGAN/BMI_distribution.png)
![](./images/GAN/BloodPressure_distribution.png)            |  ![](./images/WGAN/BloodPressure_distribution.png)
![](./images/GAN/DiabetesPedigreeFunction_distribution.png) |  ![](./images/WGAN/DiabetesPedigreeFunction_distribution.png)
![](./images/GAN/Glucose_distribution.png)                  |  ![](./images/WGAN/Glucose_distribution.png)
![](./images/GAN/Insulin_distribution.png)                  |  ![](./images/WGAN/Insulin_distribution.png)
![](./images/GAN/SkinThickness_distribution.png)            |  ![](./images/WGAN/SkinThickness_distribution.png)
![](./images/GAN/Outcome_distribution.png)                  |  ![](./images/WGAN/Outcome_distribution.png)

