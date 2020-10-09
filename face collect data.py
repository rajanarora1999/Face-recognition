#-- generating selfie training data for face recognition
#1. read video stream
#2. detect largest face and flatten the pixels and store in a numpy array
#3. repeat for multiple people and make face data


import cv2
import numpy as np

#capture the input device from which we will take i/p stream
cap = cv2.VideoCapture(0)
#0 for capturing your default webcam
# Face Detection(#created a classifier object (harcascade classifier))
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
n=int(input('enter number of persons whose data is to be collected='))
for i in range(n):
	#list for storing the facial data
	face_data=[]
	skip=0 
	file_name = input("Enter the name of the person : ")
	while True:
		#cap.read() returns a tuple one is bool value and other is reqd frame(img)
		#if bool values is false means image is not properly captured
		ret,frame = cap.read()

		if ret==False:
			continue
		
		#in detect multi scale we give the frame, scaling factor, number of neighbours
				#scaling factor-> how much image size is reduced at each capture
				#1.3-> shrink image by 30%
				#we shrink image as harcascade works for a fixed size image
				# detect multi scale returns a list of tuples
				# each tuple represents each face in the frame
				# each tuple is of form (x,y,w,h) where (x,y) is top left coordinate of face and w and h are width and height
				
				
		faces = face_cascade.detectMultiScale(frame,1.3,5)
		#if no face is detected
		if len(faces)==0:
			continue
		#sort for largest face (sort by area(w*h))	
		faces = sorted(faces,key=lambda f:f[2]*f[3])

		# Pick the last face (because it is the largest face acc to area(f[2]*f[3]))
		for face in faces[-1:]:
			#coordinates of top left and width and height
			x,y,w,h = face
			# draw a rectangle over each face
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
			# this rectangle method takes frame and coordinates of top left and bottom right and also the color in bgr(255,0,0 means blue)
			#now extract the area reqd i.e dont take exact boundary of face take extra spacing of 10 on each side
			#Extract (Crop out the required face) : Region of Interest
			offset = 10
			face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
			face_section = cv2.resize(face_section,(100,100))

			skip += 1
			#we store capture after every 10 ms
			if skip%10==0:
				face_data.append(face_section)
				print(len(face_data),end=' ')
				print("samples of {} collected".format(file_name))

		#shows the frame with window name as frame
		cv2.imshow("Frame",frame)
		cv2.imshow('face',face_section)
		#wait for user input
				# let we want to quit when q is pressed
				# cv2.waitkey(1) means wait for 1ms and it returns a 32 bit integer
				#0xFF is 8 bits all ones(11111111)
				#and of both converts 32 bits num into 8 bit num(0-255)
		key_pressed = cv2.waitKey(1) & 0xFF
		#stop capturing frames when q is pressed(ord gives ascii of q)
		if key_pressed == ord('q'):
			#close the window(will open again for new person)
			cv2.destroyAllWindows()
			break

	# Convert our face list array into a numpy array
	face_data = np.asarray(face_data)
	#flatten the array
	face_data = face_data.reshape((face_data.shape[0],-1))
	print(face_data.shape)

	# Save this data into file system
	np.save(file_name+'.npy',face_data)
	print("Data Successfully save at "+file_name+'.npy')
#release the device
cap.release()

