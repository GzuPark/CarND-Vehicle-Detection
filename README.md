# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


Steps
---

The goals / steps of this project are the following:

1. Exploratory Data Analysis (EDA) with Cars and Non-Cars dataset.
2. Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
3. Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
4. Create a heatmap of recurring detections to reject outliers and follow detected vehicles.
5. Estimate a bounding box for vehicles detected.
6. Run the pipeline on a video stream.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier. These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the [project video itself](./project_video.mp4).

Files
---

* `writeup.md`: Describe the project what I apply tools and techniques
* `vehicle_detection.ipynb`: Code following by steps of pipeline
* `project_video.mp4`: Video to make result for the project
* `project_video_result.mp4`: Result video
* `output_images`: images for writeup
* `test_images`: Highway images for testing pipline
