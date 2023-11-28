# %% [markdown]
# ## Imports
import numpy as np
import torch
from torchvision import models
from PIL import Image
import pytesseract
import cv2
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' or any other interactive backend
import matplotlib.pyplot as plt
# import tesseract

# %% [markdown]
# ## Functionalities


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
class TextDetector:


    def __init__(self, img, structuring_el_size=(17, 3)):
        self.__img = img
        self.__bound_rects = self.__detect_text(structuring_el_size)

    def __detect_text(self, strucuring_el_size=(17, 3)):

        boundRect = []

        # Converting image to grayscale.
        if (len(self.__img.shape) == 3):
            img_gray = cv2.cvtColor(self.__img, cv2.COLOR_BGR2GRAY)
        elif (len(self.__img.shape) == 2):
            img_gray = self.__img
        else:
            raise TypeError("Invalid shape for image array.")
        img_sobel = cv2.Sobel(img_gray, cv2.CV_8U, 1, 0, None, 3, 1, 0,
                              cv2.BORDER_DEFAULT)

        
        _, img_threshold = cv2.threshold(img_sobel, 0, 255,
                                         cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        
        element = cv2.getStructuringElement(cv2.MORPH_RECT, strucuring_el_size)
        img_threshold = cv2.morphologyEx(cv2.UMat(img_threshold),
                                         cv2.MORPH_CLOSE, element)

        
        contours, _ = cv2.findContours(img_threshold, 0, 1)

        
        contoursPoly = []
        for c in contours:
            contoursPoly.append(cv2.approxPolyDP(c, 3, True))
            rect = cv2.boundingRect(contoursPoly[-1])
            x, y, w, h = rect
            if (w > h):
                boundRect.append(rect)
        return boundRect

def recognize_text(self):
        """Recognize text from the bounding boxes found in self.__detect_text,
            using each bounding box as input to the Tesseract neural network.
            This method uses pytesseract to recognize the characters in each
            area.

        Note:
            To properly use the Tesseract, we utilize the Pillow library to
            convert self.__img to a Pillow Image
        Returns:
            strs (list): list of strings recognized by the Tesseract.
            bboxes (list): list of bounding boxes recognized in the previous
                method.
            img_bboxes (np.ndarray): Array representation of the input image,
                with red (white) rectangles drawn over each bounding box.
        """
        strs = []
        img_bboxes = self.__img.copy()

        # For each bounding box
        for box in self.__bound_rects:
            x, y, w, h = box
            # Draw rectangle
            if (len(self.__img.shape) == 3):
                cv2.rectangle(img_bboxes, (x, y), (x + w, y + h), (0, 0, 255),
                              3, 8, 0)
            elif (len(self.__img.shape) == 2):
                cv2.rectangle(img_bboxes, (x, y), (x + w, y + h), 255, 3, 8, 0)
            else:
                raise TypeError("Invalid shape for image array.")
            # Find text
            img_pil = Image.fromarray(self.__img[y:y + h + 1, x:x + w + 1])
            strs.append(pytesseract.image_to_string(img_pil))

        return strs, self.__bound_rects, img_bboxes


# %% [markdown]
# ## Tests
if __name__ == "__main__":
    import imageio
    from corner_detection import CornerDetector
    from perspective_transform import PerspectiveTransform
    import matplotlib.pyplot as plt
    import os

    # Listing example files
    example_files = [
        './images/' + f for f in os.listdir('./images')
        if os.path.isfile(os.path.join('./images', f))
    ]
    # Selecting random file for testing
    file_img = example_files[np.random.randint(0, len(example_files))]
    # file_img = './images/806123698_321554.jpg'  # Good file for testing
    img = imageio.imread(file_img)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.show()

    # Finding corners from input image
    corner_points = CornerDetector(img).find_corners4().astype(np.float32)
    corner_points[:, [0, 1]] = corner_points[:, [1, 0]]

    # Computing the perspective transform
    img_p = PerspectiveTransform(img, corner_points).four_point_transform()

    # Finding text areas
    img_cv = cv2.cvtColor(img_p, cv2.COLOR_RGB2BGR)
    # Testing with different structuring element sizes
    sizes = [(17, 3), (30, 10), (5, 5), (9, 3)]
    for size in sizes:
        strs, bound_rects, img_bboxes = TextDetector(img_cv,
                                                     size).recognize_text()
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img_bboxes, cv2.COLOR_BGR2RGB))
        plt.show()
        print(size)
        print(*strs, sep='\n')
