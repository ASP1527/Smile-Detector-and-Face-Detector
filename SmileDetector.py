import cv2

# files with machine learning to detect faces
face_detector = cv2.CascadeClassifier('face.xml')
smile_detector = cv2.CascadeClassifier('smile.xml')

# loads the webcam, number represents stream (0 for first camera and 1 for second etc)
webcam = cv2.VideoCapture(1)

# loop that runs to detect the faces
while True:
    # read the current frame from the webcam
    successful_frame_read, frame = webcam.read()

    # stops the program if there is an error
    if not successful_frame_read:
        break

    # changes the frame to grayscale so it works faster as there is only one channel used (brightness channel) rather than the 3 colour channels + the brightness channel
    frame_greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detects the faces in grayscale
    faces = face_detector.detectMultiScale(frame_greyscale)

    # draws a box around each face, rgb is invertes in this module/ runs face detection
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 200, 50), 4)

        # crop the face so it detects smiles within a face uses numpy N-dimensional slicing
        the_face = frame[y:y+h, x:x+w]
        # changes the face to grayscale
        face_greyscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # detects the smiles in grayscale, scaleFactor blurs the image to make it easier to detect smiles, higher the number=more blur, minNeighbors is the minimum amount of rectangles to count it as a smile
        smiles = smile_detector.detectMultiScale(
            face_greyscale, scaleFactor=1.7, minNeighbors=20)

        # draws a box around each smile, rgb is invertes in this module/ runs smile detection for the smile in the face
        # for (x_, y_, w_, h_) in smiles:
        #cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_),(50, 50, 200), 4)
        if len(smiles) > 0:
            cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale=3,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    # shows the image, imshow=image show, shows the caption and the frame
    cv2.imshow('Smile Detector', frame)

    # waits for a key to be pressed until the frame is replaced with the latest frame, the number indicates how many miliseconds until it force refreshes
    cv2.waitKey(1)

# releases the program from using the webcam
webcam.release()
# closes all of the windows that the program uses
cv2.destroyAllWindows()
