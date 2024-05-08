## Project Title :
Bounding boxes

## Overview :
```
*The project allows users to annotate images with bounding boxes, defining regions of interest within the images.
*It provides functionalities to process images by cropping them based on the annotated bounding boxes.
*The project facilitates visualizing bounding boxes on images, which aids in understanding the spatial relationships between objects in the images
*It assists in preparing datasets for machine learning tasks, as bounding boxes are commonly used to train object detection models.
````
## Usage Of Bounding Box :
* Object Detection: Bounding boxes are extensively used in object detection tasks, where the goal is to detect and localize objects within an image
* Image Segmentation: In semantic segmentation tasks, bounding boxes are often used as annotations to define regions of interest within images
``
## Importing Libraries :
```
import os [Provides functions for interacting with the operating system]
import csv [Allows reading and writing CSV files]
from PIL import Image, ImageDraw [Part of the Python Imaging Library (PIL)]
```
 ## Defining File Paths :
 csv_file = "/home/afreen-mohammad/Downloads/7622202030987_bounding_box.csv"
 
image_dir = "/home/afreen-mohammad/Downloads/7622202030987/" [ image copypath in downloads ]

output_dir = "/home/afreen-mohammad/Downloads/7622202030987/_with_boxes" [output copypath in downloads]
``
## Creating Output Directory:
```
os.makedirs(output_dir, exist_ok=True)
```
## Processing Images code:
```
def draw_boxes(image, boxes):

    draw = ImageDraw.Draw(image)

    for box in boxes:

        left = int(box['left'])

        top = int(box['top'])

        right = int(box['right'])

        bottom = int(box['bottom'])

        draw.rectangle([left, top, right, bottom], outline="red")

    return image


def crop_image(image, boxes):

    cropped_images = []

    for box in boxes:

        left = int(box['left'])

        top = int(box['top'])

        right = int(box['right'])

        bottom = int(box['bottom'])

        cropped_img = image.crop((left, top, right, bottom))

        cropped_images.append(cropped_img)

    return cropped_images


with open(csv_file, 'r') as file:

    csv_reader = csv.DictReader(file)

    for row in csv_reader:

        image_name = row['filename']

        image_path = os.path.join(image_dir, image_name)

        output_path = os.path.join(output_dir, image_name)

        image = Image.open(image_path)

        boxes = [{'left': row['xmin'], 'top': row['ymin'], 'right': row['xmax'], 'bottom': row['ymax']}]

        cropped_images = crop_image(image, boxes)

        for i, cropped_img in enumerate(cropped_images):

            cropped_img.save(os.path.join(output_dir, f"{i}_{image_name}"))

        full_image_with_boxes = draw_boxes(image, boxes)

        full_image_with_boxes.save(os.path.join(output_dir, f"full_{image_name}"))

```
## input Of The Code :
![image](https://github.com/afreen87awesome/ashu/assets/169051698/bf14a484-b8fb-4f06-8445-4bdc58791b19)

## Output Of The Code:
![image](https://github.com/afreen87awesome/ashu/assets/169051698/fcf933f3-6a01-46ba-9b6b-3ec5c91df01f)



## Project Title :
   Histogram

## Importing Libraries:

import numpy as np [NumPy is imported to handle numerical operations and arrays efficiently.]
import cv2 as cv [OpenCV is imported for image processing tasks]
from matplotlib import pyplot as plt [Matplotlib is imported for plotting histograms]
``
## Loading an Image:

img = cv.imread("/home/afreen-mohammad/Downloads/flowers.jpeg")
Note :- cv.imread(): This is a function from the OpenCV library (cv2 module) used to read an image from a file.

## Writing the Image:

cv.imwrite("/home/afreen-mohammad/Downloads/__pycache__/sana.jpg",img)
Note :-cv.imwrite()is used to write the loaded image (img) to a new file at the specified path ("/home/afreen-mohammad/Downloads/__pycache__/sana.jpg").
*This step saves the image with the same content as the original one.

## Assert Statement:
assert img is not None, "file could not be read, check with os.path.exists()"
Note :-An assert statement is used to check if the image was loaded successfully
     * If the loaded image is None, meaning it could not be read, the script raises an AssertionError with the message "file could not be read, check with os.path.exists()".

## Plotting Histograms:
color = ('b','g','r')
for i,col in enumerate(color):
Note :-    For each color channel, the script plots the histogram values (histr) using plt.plot().
  *  The color for each channel is specified ('b' for blue, 'g' for green, and 'r' for red).
    The x-axis limits are set to range from 0 to 256 using plt.xlim([0,256]).

## Calculating Histogram:

histr = cv.calcHist([img],[i],None,[256],[0,256])
* This line calculates the histogram of the image img for a specific color channel indicated by the variable i, using 256 bins and a pixel value range from 0 to 255.
  
 plt.plot(histr,color = col)
*This line plots the histogram values stored in the variable histr with the specified color col.

  plt.xlim([0,256])
  *This line sets the x-axis limits of the plot to range from 0 to 255, ensuring that the histogram covers the entire intensity range of the color channel.
 
 ## Displaying the Plot:
 plt.show()

## Oral Histogram code :
import numpy as np

import cv2 as cv

from matplotlib import pyplot as plt
 
img = cv.imread("/home/afreen-mohammad/Downloads/flowers.jpeg")

cv.imwrite("/home/afreen-mohammad/Downloads/__pycache__/sana.jpg",img)

assert img is not None, "file could not be read, check with os.path.exists()"

color = ('b','g','r')

for i,col in enumerate(color):

 histr = cv.calcHist([img],[i],None,[256],[0,256])
 
 plt.plot(histr,color = col)
 
 plt.xlim([0,256])
 
plt.show()



## Histogram Code Input :
![sana](https://github.com/afreen87awesome/ashu/assets/169051698/47c6cdb5-b605-4fa1-9bff-35ead764851f)

## Histogram Code Output :
![histgram scnsht](https://github.com/afreen87awesome/ashu/assets/169051698/c92248b2-efc4-44db-bb70-fafd48dac285)


## 3Project: 
iteration

An iteration refers to the process of repeating a set of instructions or a block of code multiple times. 
In programming, iterations are commonly used within loops, which are structures that allow you to execute a block of code repeatedly until a certain condition is met or for a specified number of times.
```
num = list(range(10))
```
This line creates a list named num containing numbers from 0 to 9 (inclusive). range(10) generates numbers from 0 up to (but not including) 10, and list() converts it into a list.
```
previousNum = 0
```
This line initializes a variable previousNum to 0. This variable is used to keep track of the previous number in each iteration of the loop
```
for i in num:
```
This line starts a for loop that iterates over each element (i) in the num list.
```
 sum = previousNum + i
```
In each iteration of the loop, this line calculates the sum of the current number (i) and the previous number (previousNum) and stores it in a variable named sum.
```
    print('Current Number '+ str(i) + 'Previous Number ' + str(previousNum) + 'is ' + str(sum)) # <- This is the issue.
```
This line prints the current number (i), the previous number (previousNum), and their sum. However, there's an issue here: the strings are concatenated without any spaces or punctuation between them, which can make the output difficult to read.
```
    previousNum=i
```
This line updates the value of previousNum to the current number (i) for the next iteration of the loop.

## Iteration Oral Code :
num = list(range(10))
previousNum = 0
for i in num:
    sum = previousNum + i
    print('Current Number '+ str(i) + 'Previous Number ' + str(previousNum) + 'is ' + str(sum)) # <- This is the issue.
    previousNum=i

## Iteration Code Output :
Current number 0 Previous Number 0 is 0

Current number 1 Previous Number 0 is 1

Current number 2 Previous Number 1 is 3

Current number 3 Previous Number 2 is 5

Current number 4 Previous Number 3 is 7

Current number 5 Previous Number 4 is 9

Current number 6 Previous Number 5 is 11

Current number 7 Previous Number 6 is 13

Current number 8 Previous Number 7 is 15

Current number 9 Previous Number 8 is 17
