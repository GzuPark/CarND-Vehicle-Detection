# Vehicle Detection Project

The goals / steps of this project are the following:

1. Exploratory Data Analysis (EDA) with Cars and Non-Cars dataset.
2. Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
3. Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
4. Create a heatmap of recurring detections to reject outliers and follow detected vehicles.
5. Estimate a bounding box for vehicles detected.
6. Run the pipeline on a video stream.

[//]: # (Image References)
[image1]: ./output_images/both_examples.png
[image2]: ./output_images/selected_parameters.png
[image3]: ./output_images/simple_rectangle.png
[image4]: ./output_images/various_rectangles.png
[image5]: ./output_images/heatmap.png
[image6]: ./output_images/heatmap_with_threshold.png
[image7]: ./output_images/labels.png
[image8]: ./output_images/draw_rectangles_in_image.png
[image9]: ./output_images/project_result.gif
[video1]: ./project_video_result.mp4

#### 1. Exploratory Data Analysis (EDA)

For the Vehicle Detection project, the example images of Cars and Non-Cars from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the [project video itself](./project_video.mp4).

If I can find a car in an image, I have to train images both cars and non-cars, whose ratio is 1:1 (recommend). Thus, I should explore numbers of dataset that the number of cars is `8792` and the number of non-cars is `8968`. These are random samples like below:

 ![][image1]

#### 2. Histogram of Oriented Gradients (HOG)

The code for this step is contained from the 1st code cell to the 10th code cell of the IPython notebook. Extracting HOG features from images is defined by the function `get_hog_features()`, and aggregating features, which depends on the color space and the number of channel, is defind by the function `extract_features()`.

I started by extracting features in all the Cars and Non-Cars images. I then aggregated all and normalized using by `StandardScaler()`. After making dataset for training, I split training dataset and test dataset, whose percentile is 75% and 25%. Also, I set up `random_state=777` for reproducibility and fairness among various parameters. I chose `LinearSVC`, since the results are mostly similar, but the runtime is significantly less ([refer from: SVC Classification in scikit-learn](http://scikit-learn.org/stable/modules/svm.html#classification)).

I tried to select parameters simply for HOG feature extraction. As you can see below, I made lists and they showed 144 different combinations. Yes, I can extend orient, pix_per_cell, and cell_per_block, but the runtime would be very large: running time is almost 2 hour without GPU. Anyway, I could find parameter combination with best accuracy on 140th.

```python
cspace = ["RGB", "YUV", "LUV", "HLS", "YUV", "YCrCb"]
orient = [7, 9, 11]
pix_per_cell = [8, 16]
cell_per_block = 2
hog_channel = [0, 1, 2, "ALL"]
```

```
+-------+------------+--------+-----------------+-----------------+-------------+--------+--------+-------+
| Steps | ColorSpace | Orient | Pixels Per Cell | Cells Per Block | HOG Channel |  Time1 |  Score | Time2 |
+-------+------------+--------+-----------------+-----------------+-------------+--------+--------+-------+
|     1 |        RGB |      7 |               8 |               2 |           0 |  28.66 | 93.468 |  7.86 |
|     2 |        RGB |      7 |               8 |               2 |           1 |  40.08 | 94.685 |  6.17 |
|     3 |        RGB |      7 |               8 |               2 |           2 |  41.99 | 94.257 |  6.33 |
|     4 |        RGB |      7 |               8 |               2 |         ALL |  88.56 | 96.419 | 13.86 |

    ...

|   137 |      YCrCb |     11 |               8 |               2 |           0 |  28.92 | 94.707 |  7.29 |
|   138 |      YCrCb |     11 |               8 |               2 |           1 |  28.67 | 92.613 | 11.72 |
|   139 |      YCrCb |     11 |               8 |               2 |           2 |  28.69 | 90.473 |  17.6 |
|   140 |      YCrCb |     11 |               8 |               2 |         ALL |  69.95 | 98.266 | 18.13 |
|   141 |      YCrCb |     11 |              16 |               2 |           0 |  20.62 | 94.369 |  5.78 |
|   142 |      YCrCb |     11 |              16 |               2 |           1 |  20.86 | 94.212 |   7.5 |
|   143 |      YCrCb |     11 |              16 |               2 |           2 |  20.83 | 91.824 |  9.23 |
|   144 |      YCrCb |     11 |              16 |               2 |         ALL |  45.43 | 97.207 |  2.58 |
|   140 |      YCrCb |     11 |               8 |               2 |         ALL |  69.95 | 98.266 | 18.13 |
+-------+------------+--------+-----------------+-----------------+-------------+--------+--------+-------+
```

 Here is an example of each of the Cars and Non-cars classes with `cspace="YCrCb"`, `orient=11`, `pix_per_Cell=(8, 8)`, `cell_per_block=(2, 2)`, and `hog_channel="ALL"`:

 ![][image2]

#### 3. Find Cars with Sliding-Windows

The code for this step is contained from the 11th code cell to the 14th code cell of the IPython notebook. First, I decided to search windows with `find_cars()` function with fixed size of window and scale. Next, I applied all potential search areas in an image:

|Simple Box   |All potential Boxes  |
|:-----------:|:-------------------:|
|![][image3]  |![][image4]          |

Yes, it looks ugly, but I can change more readable like magic.

#### 4. Heatmap and Labels

Okay! Now, I wanted to recognize interested rectangles in an image with heatmap methods.

![][image5]

As you can see above, it still looks ugly due to left side vehicle and undefined object in a center. How about apply a threshold with heatmap? The result is below:

![][image6]

Much better! I can recognize two vehicles in the image. Let's make it to look simple using by `labels()`, which can identify individual blobs in the heatmap.

![][image7]

#### 5. Draw Boxes

Finally, I can apply a bounding box for vehicles detected on the original image. I applied the `draw_label_bboxes()` function for making representative box. Look at the result below:

![][image8]


#### 6. Video Implementation

Here's a [link to my video result](video1) and a GIF file below:

![][image9]


Discussion
---

As you can see GIF image, my results have irregular rectangles in every frame. If boxes in the previous some frame memorize, I then show more smoothly. Also, it contains three main issues. First, it cannot recognize a vehicle on 20 sec. Second, it looks like one vehicle when two vehicles overlap from 33 sec to 35 sec. Last, an undidentified object shows on the center of image when 40 sec. First problem can upgrade if I apply more data. Third issue maybe improve performance when I choose another parameters with HOG. But, Second should occur when heatmap transform to labels. If I apply more variety size of rectangles and tune threshold, I expect to seperate two object.

I apply only traditional computer vision methods not modern, such as machine learning. Next step, I will use the convolutional nueral network with tensorflow.
