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
training the AI models. I have used XGboost and CNN for numerical and image analysis, respectively. 