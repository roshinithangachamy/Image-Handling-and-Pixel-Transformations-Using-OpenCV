# Image-Handling-and-Pixel-Transformations-Using-OpenCV 

## AIM:
Write a Python program using OpenCV that performs the following tasks:

1) Read and Display an Image.  
2) Adjust the brightness of an image.  
3) Modify the image contrast.  
4) Generate a third image using bitwise operations.

## Software Required:
- Anaconda - Python 3.7
- Jupyter Notebook (for interactive development and execution)

## Algorithm:
### Step 1:
Load an image from your local directory and display it.

### Step 2:
Create a matrix of ones (with data type float64) to adjust brightness.

### Step 3:
Create brighter and darker images by adding and subtracting the matrix from the original image.  
Display the original, brighter, and darker images.

### Step 4:
Modify the image contrast by creating two higher contrast images using scaling factors of 1.1 and 1.2 (without overflow fix).  
Display the original, lower contrast, and higher contrast images.

### Step 5:
Split the image (boy.jpg) into B, G, R components and display the channels

## Program Developed By:
### NAME: PREETHI A K 
### REGISTER NUMBER :212223230156

  ### Ex. No. 01

#### 1. Read the image ('Eagle_in_Flight.jpg') using OpenCV imread() as a grayscale image.
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
img =cv2.imread('Eagle_in_Flight.jpg',cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

#### 2. Print the image width, height & Channel.
```
img.shape
```

#### 3. Display the image using matplotlib imshow().
```
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
plt.imshow(img_gray,cmap='grey')
plt.show()
```

#### 4. Save the image as a PNG file using OpenCV imwrite().
```
img=cv2.imread('Eagle_in_Flight.jpg')
cv2.imwrite('Eagle.png',img)
```

#### 5. Read the saved image above as a color image using cv2.cvtColor().
```
img=cv2.imread('Eagle.png')
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
```




#### 6. Crop the image to extract any specific (Eagle alone) object from the image.
```
crop = img_rgb[0:450,200:550] 
plt.imshow(crop[:,:,::-1])
plt.title("Cropped Region")
plt.axis("off")
plt.show()
crop.shape
```

#### 7. Resize the image up by a factor of 2x.
```
res= cv2.resize(crop,(200*2, 200*2))
```

#### 8. Flip the cropped/resized image horizontally.
```
flip= cv2.flip(res,1)
plt.imshow(flip[:,:,::-1])
plt.title("Flipped Horizontally")
plt.axis("off")
```


#### 9. Read in the image ('Apollo-11-launch.jpg').
```
img=cv2.imread('Apollo-11-launch.jpg',cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb.shape
```

#### 10. Add the following text to the dark area at the bottom of the image (centered on the image):
```
text = 'Apollo 11 Saturn V Launch, July 16, 1969'
font_face = cv2.FONT_HERSHEY_PLAIN

```

#### 11. Draw a magenta rectangle that encompasses the launch tower and the rocket.
```
rcol= (255, 0, 255)
cv2.rectangle(img_rgb, (400, 100), (800, 650), rcol, 3)  
```

#### 13. Display the final annotated image.
```
plt.title("Annotated image")
plt.imshow(img_rgb)
plt.show()
```


#### 14. Read the image ('Boy.jpg').
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

#### 15. Adjust the brightness of the image.
```
boy = cv2.imread("boy.jpg")   # make sure boy.jpg is uploaded in Colab
if boy is None:
    raise FileNotFoundError("boy.jpg not found! Please upload it using left sidebar > Files.")

boy_rgb = cv2.cvtColor(boy, cv2.COLOR_BGR2RGB)
```

#### 16. Create brighter and darker images.
```
matrix = np.ones(boy_rgb.shape, dtype="uint8") * 50   # brightness adjustment value

img_brighter = cv2.add(boy_rgb, matrix)
img_darker   = cv2.subtract(boy_rgb, matrix)
```

#### 17. Display the images (Original Image, Darker Image, Brighter Image).
```
plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(boy_rgb); plt.title("Original"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(img_darker); plt.title("Darker"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(img_brighter); plt.title("Brighter"); plt.axis("off")
plt.show()
```

#### 18. Modify the image contrast.
```
matrix1 = np.ones(boy_rgb.shape, dtype="float32") * 1.1
matrix2 = np.ones(boy_rgb.shape, dtype="float32") * 1.2

img_higher1 = cv2.multiply(boy_rgb.astype("float32"), matrix1)
img_higher2 = cv2.multiply(boy_rgb.astype("float32"), matrix2)
```

#### 19. Display the images (Original, Lower Contrast, Higher Contrast).
```
plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(boy_rgb); plt.title("Original"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(img_higher1); plt.title("Higher Contrast (1.1x)"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(img_higher2); plt.title("Higher Contrast (1.2x)"); plt.axis("off")
plt.show()

```

#### 20. Split the image (boy.jpg) into the B,G,R components & Display the channels.
```
b, g, r = cv2.split(boy_rgb)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(r, cmap="Reds"); plt.title("Red Channel"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(g, cmap="Greens"); plt.title("Green Channel"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(b, cmap="Blues"); plt.title("Blue Channel"); plt.axis("off")
plt.show()
```

#### 21. Merged the R, G, B , displays along with the original image
```
merged_rgb = cv2.merge([r, g, b])

plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.imshow(boy_rgb); plt.title("Original RGB"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(merged_rgb); plt.title("Merged RGB"); plt.axis("off")
plt.show()
```

#### 22. Split the image into the H, S, V components & Display the channels.
```
boy_hsv = cv2.cvtColor(boy_rgb, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(boy_hsv)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(h, cmap="hsv"); plt.title("Hue Channel"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(s, cmap="gray"); plt.title("Saturation Channel"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(v, cmap="gray"); plt.title("Value Channel"); plt.axis("off")
plt.show()
```
#### 23. Merged the H, S, V, displays along with original image.
```
merged_hsv = cv2.merge([h, s, v])
merged_hsv_rgb = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2RGB)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.imshow(boy_rgb); plt.title("Original RGB"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(merged_hsv_rgb); plt.title("Merged HSV→RGB"); plt.axis("off")
plt.show()
```
 
## Output:
<img width="385" height="516" alt="image" src="https://github.com/user-attachments/assets/da74c2f6-69d0-4eae-abb8-e2f9800d3b16" />
<img width="380" height="502" alt="image" src="https://github.com/user-attachments/assets/c01bf3c4-31ab-4909-8940-cbc50ad95f73" />
<img width="381" height="498" alt="image" src="https://github.com/user-attachments/assets/be45b8ca-4b39-4f36-9072-22bc2b6d9ecc" />
<img width="382" height="501" alt="image" src="https://github.com/user-attachments/assets/ab4f144c-6e51-4d49-8779-845a487b8f7f" />
<img width="376" height="498" alt="image" src="https://github.com/user-attachments/assets/f00f61b4-98c9-4ca9-b415-d960e38f8fd8" />
<img width="375" height="485" alt="image" src="https://github.com/user-attachments/assets/407cd394-35d2-4f5d-a53e-6b54c0060d12" />
<img width="377" height="498" alt="image" src="https://github.com/user-attachments/assets/a849d328-8887-40df-a606-b02b7bde9309" />
<img width="371" height="497" alt="image" src="https://github.com/user-attachments/assets/09d4da6f-5a74-4000-8869-c4db53ae44a8" />
<img width="377" height="497" alt="image" src="https://github.com/user-attachments/assets/ed2ff481-ebba-4cc3-97c2-ef47e7153cca" />
<img width="598" height="495" alt="image" src="https://github.com/user-attachments/assets/029efba9-ee82-416c-b621-f0af799d1a13" />
<img width="464" height="500" alt="image" src="https://github.com/user-attachments/assets/eaaefb30-0fca-44fb-a552-bac46ff724a3" />
<img width="376" height="494" alt="image" src="https://github.com/user-attachments/assets/197a15ab-59f0-4902-9d24-7d1d67b39083" />
<img width="378" height="502" alt="image" src="https://github.com/user-attachments/assets/57c84bb0-1640-4215-bd84-fe191783e7f7" />


## Result:
Thus, the images were read, displayed, brightness and contrast adjustments were made, and bitwise operations were performed successfully using the Python program.

