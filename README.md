# Detect BTS faces
 
This project is inspired by: https://towardsdatascience.com/build-a-taylor-swift-detector-with-the-tensorflow-object-detection-api-ml-engine-and-swift-82707f5b4a56

![](https://github.com/irenechang1510/detect-bts-faces/blob/main/images/BTS%20Announced%20as%20Louis%20Vuitton....jpg)

Most of the applications I have seen of Tensorflow Object Detection API is recognizing objects of general classes (persons, cats, cars, etc). However, the above article uses this API to detect a specific person, which I find really interesting. I wonder if this model performs well with recognizing not one, but distinguishing between many different people. That's the motivation behind my attempt to try this model out on the boy band BTS. I'm curious if the model can label correctly the 7 members who have frequently changed their hairstyle, makeup, and who my family members usually commented: "They look the same!" (jokingly). Even though projects on this API has been quite popular and the API itself has proved to be pretty easy to use, I want to explore the difference between several common pretrained checkpoints, compare their performance, and see if I can find a way to tune these models to perform better.

## 1st attempt: SSD Resnet50 + tighten the pounding boxes

Being a newbie in this Object Detection playground, I downloaded only 200 images on this first attempt, and all of these images are group photos. After reading and following a lot of examples on this Object Detection API, and many failed attempts later, I was able to set the training in motion. The first model that I used is SSD Resnet50, which yielded an okay result, but it did poorly on new images. After some quick research, I found out the problem is that my labels in the images weren't tight enough. This introduced a lot of noise into the model, which explains why it performed not really well. After re-labeling the faces, making the pounding boxes to fit just right the faces, I improved the metrics to:

| Metrics |Results | 
| :---: | :---: |
| Average Precision  (AP) | @[ IoU=0.50:0.95 - area=   all - maxDets=100 ] = 0.593 |
| Average Precision  (AP) | @[ IoU=0.50      - area=   all - maxDets=100 ] = 0.926 |
| Average Precision  (AP) | @[ IoU=0.75      - area=   all - maxDets=100 ] = 0.731 |
| Average Precision  (AP) | @[ IoU=0.50:0.95 - area= small - maxDets=100 ] = 0.725 |
| Average Precision  (AP) | @[ IoU=0.50:0.95 - area=medium - maxDets=100 ] = 0.571 |
| Average Precision  (AP) | @[ IoU=0.50:0.95 - area= large - maxDets=100 ] = 0.659 |
| Average Recall     (AR) | @[ IoU=0.50:0.95 - area=   all - maxDets=  1 ] = 0.630 |
| Average Recall     (AR) | @[ IoU=0.50:0.95 - area=   all - maxDets= 10 ] = 0.653 |
| Average Recall     (AR) | @[ IoU=0.50:0.95 - area=   all - maxDets=100 ] = 0.653 |
| Average Recall     (AR) | @[ IoU=0.50:0.95 - area= small - maxDets=100 ] = 0.750 |
| Average Recall     (AR) | @[ IoU=0.50:0.95 - area=medium - maxDets=100 ] = 0.637 |
| Average Recall     (AR) | @[ IoU=0.50:0.95 - area= large - maxDets=100 ] = 0.684 |

![](https://github.com/irenechang1510/detect-bts-faces/blob/main/images/result.png)

*_The model only mislabels RM as Jhope!_*

At this point, my model was able to tell some members of BTS apart in new images, but on average, it still makes some really bad mislabelling. Thus, I go on to explore the mistakes the model usually makes, and explore different ways to improve it. The most obvious problem is that the model performs really bad if the picture doesn't have all 7 members.

## 2nd attempt: More individual/subunit photos and Faster-RCNN Resnet

I added around 200 more images in total of subunit/solo photos into my dataset and try fitting using another checkpoint, Faster-RCNN. For this checkpoint, I trained the model using Resnet50 and Resnet152. The latter is said to result in better results since this neural network contains more layers. The process is similar, and I obtained a slightly better results than the previous model for both of Faster RCNN models:

| Metrics | Results for Resnet152| Results for Resnet50 | 
| :---: | :---: |  :---: |
| Average Precision  (AP) @[ IoU=0.50:0.95 - area=   all - maxDets=100 ] | = 0.606 | = 0.603|
| Average Precision  (AP) @[ IoU=0.50      - area=   all - maxDets=100 ] | = 0.931 | = 0.949|
| Average Precision  (AP) @[ IoU=0.75      - area=   all - maxDets=100 ] | = 0.702 | = 0.699|
| Average Precision  (AP) @[ IoU=0.50:0.95 - area= small - maxDets=100 ] | = 0.758 | = 0.800|
| Average Precision  (AP) @[ IoU=0.50:0.95 - area=medium - maxDets=100 ] | = 0.598 | = 0.593|
| Average Precision  (AP) @[ IoU=0.50:0.95 - area= large - maxDets=100 ] | = 0.638 | = 0.643|
| Average Recall     (AR) @[ IoU=0.50:0.95 - area=   all - maxDets=  1 ] | = 0.654 | = 0.625|
| Average Recall     (AR) @[ IoU=0.50:0.95 - area=   all - maxDets= 10 ] | = 0.675 | = 0.674|
| Average Recall     (AR) @[ IoU=0.50:0.95 - area=   all - maxDets=100 ] | = 0.687 | = 0.682|
| Average Recall     (AR) @[ IoU=0.50:0.95 - area= small - maxDets=100 ] | = 0.767 | = 0.817|
| Average Recall     (AR) @[ IoU=0.50:0.95 - area=medium - maxDets=100 ] | = 0.676 | = 0.666|
| Average Recall     (AR) @[ IoU=0.50:0.95 - area= large - maxDets=100 ] | = 0.699 | = 0.696|

The Resnet50 model also performs better, though just by a little. Despite having similar precision and still making mistakes on a lot of new images, one improvement when more individuals photos are included is that now the model labels pictures with fewer than 7 members more accurately.

![](https://github.com/irenechang1510/detect-bts-faces/blob/main/images/result2.png)

*_The model correctly labels both of them!_*

For the next step, I'm going to increase the dataset by downloading more images as well as by employing some data augmentation techniques

## 3rd attempt: increase the number of images to 2000+ images

I added more images to improve the model. It's said that each class should have at least 1000 images to achieve the optimal results. Due to the limitation in time, I could only manage to obtain around 300+ images for each member, plus a few hundred group/subunit images. Contrary to what I believed, the model's performance worsens. For the same set of images, it's unable to label the images as correctly. I suspect this is because I included more individual images than group images, so most of the time the model just corrects the weights to one class, rather than all of them, thus the variation between pictures of different members is huge.

## Conclusion 
Object detection API can do well in identifying objects of different types, or distinguishing between human face and other objects. However, it doesn’t perform as well on recognizing more subtle features like the one illustrated in this task: face. If we train the model to recognize one person’s as the only class (as in this [article](https://towardsdatascience.com/build-a-taylor-swift-detector-with-the-tensorflow-object-detection-api-ml-engine-and-swift-82707f5b4a56), then the model will perform well; but not if we are tasking it to differentiate 7 faces. The training data, which plays a huge role in the model’s accuracy, is also a tricky problem that requires more time and effort for multiple trials (which I shall study more in near future).
