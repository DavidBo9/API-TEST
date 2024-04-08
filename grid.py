import matplotlib.pyplot as plt
import cv2

# Load the image
image_path = 'nueva_img.jpg'
image = cv2.imread(image_path)

# Convert the image from BGR to RGB for displaying with matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image
plt.figure(figsize=(10,5))
plt.imshow(image_rgb)
plt.axis('on') # to turn on axis numbers
plt.show()
