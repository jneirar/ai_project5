import cv2
import time

def initiate_camera(seconds, out_name):
    st = time.time()
    iteration = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        # frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow('Input', frame)
        cur_time = time.time()
        elapsed_time = cur_time - st

        if elapsed_time > iteration * seconds:
            cv2.imwrite(out_name, frame)
            iteration += 1
            #Aquí debería ir el código que manda la imagen (frame) al separador de letras.
            #frame contiene la imagen como valor, y la imagen capturada como archivo se guarda en out_name
        c = cv2.waitKey(1)
        if c == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


#initiate_camera(5, "out.jpg")