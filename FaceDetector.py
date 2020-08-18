import cv2

# files with machine learning to detect faces
face_detector = cv2.CascadeClassifier('face.xml')

# loads the webcam, number represents stream (0 for first camera and 1 for second etc)
webcam = cv2.VideoCapture(1)

# loop that runs to detect the faces
while True:
    # read the current frame from the webcam
    successful_frame_read, frame = webcam.read()

    # stops the program if there is an error
    if not successful_frame_read:
        break

    # changes the frame to grayscale so it works faster as there is only one channel used (brightness channel) rather than the 3 other colour channels + the brightness channel
    frame_greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detects the faces in grayscale
    faces = face_detector.detectMultiScale(frame_greyscale)

    # draws a box around each face, rgb is invertes in this module
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

    # shows the image, imshow=image show, shows the caption and the frame
    cv2.imshow('Face Detector', frame)

    # waits for a key to be pressed until the frame is replaced with the latest frame, the number indicates how many miliseconds until it force refreshes
    cv2.waitKey(1)

# releases the program from using the webcam
webcam.release()
# closes all of the windows that the program uses
cv2.destroyAllWindows()
