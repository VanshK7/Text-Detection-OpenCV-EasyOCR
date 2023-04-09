import cv2
import easyocr
import matplotlib.pyplot as plt

# Read image
image_path="C:/Users/Vansh/Desktop/Coding/Python/Computer Vision 16 Projects/Text Detection/road-closed-sign.jpg"
img=cv2.imread(image_path)

# Instantiate Text Detector
reader=easyocr.Reader(['en'], gpu=False)            # As CUDA is not available, gpu=False

# Detect Text on Image
text_=reader.readtext(img)

# Draw bbox and text
for t in text_:
    print(t)
    bbox, text, confidence=t
    print(text)
    if confidence>0.7:
        cv2.rectangle(img, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1])), (0, 255, 0), 5)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)        # (image, text, bottom left corner, font, font size, color, thickness)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
