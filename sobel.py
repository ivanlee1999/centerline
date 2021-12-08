import cv2

# Read the original image
img = cv2.imread('test2.png')
# Display original image
# cv2.imshow('Original', img)
# cv2.waitKey(0)

# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (9,9), 0)

#Sobel detection
# sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
# sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
# sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=40, threshold2=40) # Canny Edge Detection
# Display Canny Edge Detection Image

filename = 'Sobel.jpg'
cv2.imwrite(filename, edges)
#cv2.imshow('Canny Edge Detection', edges)
#cv2.waitKey(0)
# cv2.destroyAllWindows()
