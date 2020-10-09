import cv2
def read_video_stream():
	#capture the input device from which we will take i/p stream
	cap=cv2.VideoCapture(0)
	#0 for capturing your default webcam
	#created a classifier object (harcascade classifier)
	face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	while True:
		#cap.read() returns a tuple one is bool value and other is reqd frame(img)
		#if bool values is false means image is not properly captured
		ret,frame=cap.read()
		if ret==False:
			continue
		faces=face_cascade.detectMultiScale(frame,1.3,5)
		#in detect multi scale we give the frame, scaling factor, number of neighbours
		#scaling factor-> how much image size is reduced at each capture
		#1.3-> shrink image by 30%
		#we shrink image as harcascade works for a fixed size image
		# detect multi scale returns a list of tuples
		# each tuple represents each face in the frame
		# each tuple is of form (x,y,w,h) where (x,y) is top left coordinate of face and w and h are width and height
		# draw a rectangle over each face
		for (x,y,w,h) in faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0))
			# this rectangle method takes frame and coordinates of top left and bottom right and also the color in bgr(255,0,0 means blue)
		#shows the frame with window name as face recognition
		cv2.imshow("face recognition",frame)
		#wait for user input
		# let we want to quit when q is pressed
		# cv2.waitkey(1) means wait for 1ms and it returns a 32 bit integer
		#0xFF is 8 bits all ones(11111111)
		#and of both converts 32 bits num into 8 bit num(0-255)
		key_pressed=cv2.waitKey(1) & 0xFF
		#stop capturing frames when q is pressed(ord gives ascii of q)
		if key_pressed==ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__=='__main__':
	#reads video stream frame by frame and recognises faces
	read_video_stream()
