# Science-Fair-24-25
This repository contains all code files used in Arnav's 2024-25 science fair project. A more updated version is avalible here(I will add link later).
## The Ground Truth: Novel Approach to Landslide Dynamics and Prediction with Machine Learning 
This project was inspired by watching a bolder fall on a car and completly crushing it during a landslide. I wanted to create a better way to predict landslides and understand how the were triggered so they could be prevented. This project explores two ways to monitor landslides, one using remote sensing and the other using local sensing methods. Explore the section below for more information.

## Code Dependencies
This code is written in python and requires the following libraries:
```bash
rasterio
gdal
numpy
tensorflow
sklearn
earthengine-api
geemap
``` 
These libraries can all be installed through the command `pip install requirements.txt`.  However, GDAL will need to be installed seperatly. 
<details>
<summary style="font-size:17px;"><strong> Using pip for downloading GDAL</strong></summary>

The pre-complied wheel for this library was sourced from [Christoph Gohlke](https://github.com/cgohlke/geospatial-wheels) 's repository for precompiled geospatial wheels for python. Don't worry, I have included it in my repository.
To download, just open up a normal terminal, cd to this repository, and run `pip install gdal.whl`. That should do the trick.

</details>
<br>
Other than that this code was complied and created completly on Windows 11 Home OS, so there could possibly be some compatability issues with other OS.
<br>
<strong>Disclaimer: The Local Sensing Arduino Code Was Created In Visual Studio Using Platformio. I do not know how it will run on other OS or editors.</strong>

## Science Fair Explanation Information:
[Link to my project board](https://google.com)
<br><br>
### Abstract
Landslides represent a significant geological hazard characterized by the rapid movement of rock, soil, and debris down a slope. These result in over 4,000 fatalities and about $20 billion worth of property damage annually worldwide. This project aims to use deep learning methods to predict landslide vulnerability values precisely. An expandable framework was developed in Python to obtain data from remote sensing methods to train deep-learning models for landslide prediction. This framework leverages a combination of data types, including Normalized Difference Vegetation Index (NDVI), rainfall, slope, topography, and satellite imagery data to provide highly accurate and scalable predictions. Trained on the USGS Landslide Susceptibility Inventory, the models incorporate diverse terrain and environmental factors to enhance prediction reliability. Multiple CNN models were created for each geological feature in consideration, and the best out of their category were then further enhanced for accuracy and efficiency. The final models consisted of satellite imagery with 99% accuracy, topography with 98% accuracy, slope with 97% accuracy, and NDVI with 66% accuracy. All models except NDVI had a loss of less than 0.2, with NDVI having 1.07. The accuracy of validation runs on the models was always within 5% of the training, usually performing better than the training run, which indicates good generalization by the models. Future work for this project will include the evaluation of a multi-input CNN and the development of an API that will use real-time satellite data to predict the probability of a landslide in any location in the world.



## Execution of Code
Here is where it all gets messy. I have used this repository as a way to dump all of my science fair code files, so getting this to run is something even I haven't streamlined yet. You should probably go to this repository for a better experience. 
However, I did take the time to comment all of the code files so other people can understand what is happening.
### Remote Sensing
<hr style="margin-top:1px;">
To run this code you are going to need Python 3.10+, the libraries mentioned earlier, and a decent computer.
<strong>You will need a google earth engine developer account to get you data.</strong> 
<br><br>

Most of the code is ready for you, so all you really have to is clone the repository and: `cd "Science-Fair-24-25/Remote Sensing/Data"` in a terminal. After that, open up the `coordinates.txt` file in the `Collection` folder and edit it to your desired coordinates. (Or just use my datasets). 
<strong> You will have to specify in decimal form (e.g 31, -112) and it will have to be in the contiguous United States.</strong>
Once you enter your desired number of coordinates (e.g 200), `cd ../` and run `./collectDatasets.sh`. This bash script runs all
of the python programs that collect and download the satellite data, and process it. After this script finishes execution, you are good to go to start 
training the AI models.
### Note that this repository is a depository for my science fair files. So the instructions might not be clear and stuff
### AI Models Information
The two main models used in this project are CNN and XGBoost. CNN from Tensorflow. The code for all of the AI models can be found under the `AI Systems` directory, 
however I strongly reccomend going to the updated repository if you want to run the code. If you insist on using this repository, the notebooks for all the final models can be found under the Results section. They might be missing a few components as I had ran into many LFS errors and had to re-intialize the repository several times. The notebooks train the model on 400 datasets, 2400 after augmentation.
