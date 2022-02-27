import cv2
import numpy as np
import imutils
import os
import time

def playSound(filename):
    ''' using the os module, plays a sound of the filename passed in'''
    os.system("afplay " + filename)

def drawContour(frame, c, word):
    ''' draws a contour on the blob with a marker, word, passed in'''
    cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

    M = cv2.moments(c)
    if (M["m00"] != 0):
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(frame, word, (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

def findLargestContour(cnts_red):
    ''' if the countour list isn't empty then returns the largest contour'''
    # goes through each blob and makes an array out of the areas
    cntsAreas = []
    if (len(cnts_red) != 0): # check to see if the list is empty or not
        for c in cnts_red:
            cntsAreas.append(cv2.contourArea(c))
        # finds the contour with the largest area
        cntsMaxArea = cnts_red[cntsAreas.index(max(cntsAreas))]
        return cntsMaxArea
    else:
        return ""

def findContours(frame, hsv, lower_red, upper_red):
    '''returns all the contours in the framw of a specified color'''
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    cnts_red = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts_red = imutils.grab_contours(cnts_red)

    return cnts_red


# names of the piano note audio files
# os.system("afplay " + file)
noteFiles = ["FA.wav", "SO.wav", "RA.wav", "SHI.wav", "DO.wav", "RE.wav", "MI.wav"]
noteNames = ["FA", "SO", "RA", "SHI", "DO", "RE", "MI"]
# yellow, green, red, blue
noteColorBounds = [
    [[25, 70, 120], [30, 255, 255]],
    [[40, 70, 80], [70, 255, 255]],
    [[0, 50, 120], [10, 255, 255]],
    [[90, 60, 0], [121, 255, 255]],
]

cap = cv2.VideoCapture(0)

scale = 1.3
width, height = int(scale * 480), int(scale * 270)

while True:
    # get what is on the camera
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lower and upper bounds for the colors
    lower_red = np.array([0, 50, 120])
    upper_red = np.array([10,255,255])

    coveredNote = [] # by the end should have 7 entries 0 or 1 depending on which
    # find all the contours of the red
    for i in range(len(noteColorBounds)):
        # get the lower and upper bound arrays
        lower_bound = np.array(noteColorBounds[i][0])
        upper_bound = np.array(noteColorBounds[i][1])

        # get all the contours
        cnts = findContours(frame, hsv, lower_bound, upper_bound)

        # finds the largest contour
        cntsMaxArea = findLargestContour(cnts)

        # draws contour and adds 0 or 1
        if cntsMaxArea != "":
            drawContour(frame, cntsMaxArea, noteNames[i])
            coveredNote.append(1)
        else:
            coveredNote.append(0)

    print(coveredNote) # checking

    # play piano sound
    coveredNoteNumber = coveredNote.count(0)
    if coveredNoteNumber == 1:
        playSound(noteFiles[coveredNote.index(0)])

    # resize and display
    frameResized = cv2.resize(frame, (width, height))
    cv2.imshow("frame", frameResized)

    key = cv2.waitKey(1)
    if (key == 27):
        break
cap.release()
cv2.destroyAllWindows()
