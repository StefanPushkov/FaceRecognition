import config as cf
import os
import face_recognition
import imutils
import time
import cv2
import pickle
import datetime

VideoOutFile = '/home/spushkov/FaceRecognition/CompVision/Main/Data/out.avi'


if not os.path.exists(cf.base_dir + '/DB_csv'):
    os.makedirs(cf.base_dir + '/DB_csv')
    with open(cf.base_dir + '/DB_csv/records.csv', 'a') as f:
        f.write("Person; Time")
        f.write("\n")


def Recognition():
    data = pickle.loads(open(cf.base_dir+'/EncodedFaces/EncodedFaces.pickle', "rb").read())
    known_encodings, known_names = data['encodings'], data['names']
    print("[INFO] starting video stream...")
    # vs = VideoStream('rtsp://80.254.24.22:554').start()  # rtsp://80.254.24.22:554  rtsp://192.168.10.165:554
    cap = cv2.VideoCapture()
    cap.open('rtsp://80.254.24.22:554')
    writer = None
    time.sleep(2.0)

    num_frames = 100

    # loop over frames from the video file stream
    while True:

        # grab the frame from the threaded video stream
        for i in range(0, num_frames):
            ret, frame = cap.read() # for VideoCapture()
            print(ret)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            rgb_resize = imutils.resize(rgb, width=1050)
            # cv2.imshow('Frame3', rgb_resize)
            r = frame.shape[1] / float(rgb_resize.shape[1])


            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input frame, then compute
            # the facial embeddings for each face
            boxes = face_recognition.face_locations(rgb_resize,
                                                    model='hog')

            encodings = face_recognition.face_encodings(rgb_resize, boxes)
            names = []

            # loop over the facial embeddings
            for encoding in encodings:
                # attempt to match each face in the input image to our known
                # encodings
                matches = face_recognition.compare_faces(known_encodings,
                                                         encoding)
                detection_at = datetime.datetime.now()
                name = "Unknown"

                # check to see if we have found a match
                if True in matches:
                    # find the indexes of all matched faces then initialize a
                    # dictionary to count the total number of times each face
                    # was matched
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for i in matchedIdxs:
                        name = known_names[i]
                        counts[name] = counts.get(name, 0)+1

                    # determine the recognized face with the largest number
                    # of votes (note: in the event of an unlikely tie Python
                    # will select first entry in the dictionary)
                    name = max(counts, key=counts.get)



                    # update the list of names
                names.append(name)

                csv_line = name + ";" + str(detection_at)
                with open(cf.base_dir + '/DB_csv/records.csv', 'a') as outfile:
                    outfile.write(csv_line + "\n")
            for ((top, right, bottom, left), name) in zip(boxes, names):
                  # rescale the face coordinates
                top = int(top)
                right = int(right)
                bottom = int(bottom)
                left = int(left)

                        # draw the predicted face name on the image
                cv2.rectangle(rgb_resize, (left, top), (right, bottom),
                                (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(rgb_resize, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)



            if writer is None and VideoOutFile is not None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(VideoOutFile, fourcc, 20,
                                        (frame.shape[1], frame.shape[0]), True)

                # if the writer is not None, write the frame with recognized
                # faces to disk
            if writer is not None:
                writer.write(frame)
                    # check to see if we are supposed to display the output frame to
                    # the screen
            '''
            if len(boxes) >= 1:
                cv2.imshow("Frame", rgb_resize)
            key = cv2.waitKey(1) & 0xFF

                    # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
            '''
        break

    #cv2.destroyAllWindows()
    cap.release()
    writer.release()



if __name__ == '__main__':
    Recognition()
