import numpy as np
import cv2
from matplotlib import pyplot as plt
from glob import glob

images = glob("../Stringed instruments (10981)/*")
for i, img_name in enumerate(images):
    print(img_name)
    if i > 10:
        break
    img = cv2.imread("../Stringed instruments (10981)/CMIM000012502.jpg")
    mask = np.zeros(img.shape[:2],np.uint8)
    h, w, _ = img.shape
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect = (0,0,w-1,h-1)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    print(mask2)
    img = img*mask2[:,:,np.newaxis]

    plt.imshow(img)
    plt.colorbar()
    plt.show()
