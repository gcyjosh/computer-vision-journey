import os

import cv2

#read video
video_path = os.path.join('.','data','waves.mp4')

video = cv2.VideoCapture(video_path)

#visualise video

ret = True 
while ret:  
    ret, frame = video.read() #True if frame is read
    
    if ret:
        cv2.imshow('frame', frame) #Opens a window titled 'frame'. Shows the current video frame.
        cv2.waitKey(40) #ask cv2 to wait 40ms while it recognises every frame in a 25fps video

# Clean up
video.release() #Closes the video file. Frees memory and resources
cv2.destroyAllWindows() #Closes the video display window.

if cv2.waitKey(40) & 0xFF == ord('q'): #This lets you stop the video early by pressing q.
    break
