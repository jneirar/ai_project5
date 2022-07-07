import numpy as np
import cv2
from mss import mss
from PIL import Image

mon = {'left': 160, 'top': 160, 'width': 1000, 'height': 600}

with mss() as sct:
    while True:
        screenShot = sct.grab(mon)
        img = Image.frombytes(
            'RGB', 
            (screenShot.width, screenShot.height), 
            screenShot.rgb, 
        )
        cv2.imshow('test', np.array(img))

        if cv2.waitKey(33) & 0xFF in (
            ord('q'), 
            27, 
        ):
            break

        if cv2.waitKey(33) == ord('s'):
            cv2.imwrite("out.png", np.array(img))
            print("Guardado")
        
        '''
        if cv2.waitKey(33) == 100:
            mon["width"] = mon["width"]+10
            print("right")
        elif cv2.waitKey(33) == 119:
            mon["width"] = mon["width"]-10
            print("left")
        elif cv2.waitKey(33) == 115:
            mon["height"] = mon["height"]-10
            print("Up")
        elif cv2.waitKey(33) == 100:
            mon["height"] = mon["height"]+10
            print("Down")
        '''
        
