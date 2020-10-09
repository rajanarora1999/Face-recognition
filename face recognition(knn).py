# 1. load the training data (numpy arrays of all the persons)
		# x- values are stored in the numpy arrays
		# y-values we need to assign for each person
# 2. Read a video stream using opencv
# 3. extract faces out of it
# 4. use knn to find the prediction of face (int)
# 5. map the predicted id to name of the user 
# 6. Display the predictions on the screen - bounding box and name

import cv2
import numpy as np 
import os 

########## KNN CODE ############
def distance(v1, v2):
	return np.sqrt(((v1-v2)**2).sum())

def knn(x,y,qp,k=5):
	#vals list to store distances and their labels
    vals=[]
    #iterate over the training data and append distance and label for each point
    for i in range(x.shape[0]):
        vals.append((distance(x[i],qp),y[i]))
    #sort the list so that k nearest can be taken
    vals=sorted(vals)
    #take first k nearest
    vals=vals[:k]
    #convert list into numpy array
    vals=np.array(vals)
    # now take how many unique labels are there in k nearest(use only 1st column for labels)
    #and also return their count
    #new vals is a list with two tupls
    #first tuple has labels and second has their freq.
    new_vals=np.unique(vals[:,1],return_counts=True)
    #take the index of max freq label
    max_freq_index=new_vals[1].argmax()
    #return the label with max freq
    return new_vals[0][max_freq_index]

################################


#capture the input device from which we will take i/p stream
cap = cv2.VideoCapture(0)
#0 for capturing your default webcam
# Face Detection(#created a classifier object (harcascade classifier))
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
#prepare a training data set and store in face_data and store its labels in a labels list
face_data = [] 
labels = []
#map each name/label with a class_id
class_id = 0 
names = {} #Mapping btw id - name


# Data Preparation
#data we generated using face collevt.py file will be now used as training data set
#os.listdir() -> put path inside brackets leave empty for same directory
for fx in os.listdir():
	#extracts all files ending with .npy
	if fx.endswith('.npy'):
		#Create a mapping btw class_id and name
		#slic upto -4 to remove .npy
		names[class_id] = fx[:-4]
		print("Loaded "+fx)
		#load the np array and append in face_data
		data_item = np.load(fx)
		face_data.append(data_item)

		#Create Labels for the class
		#let data item has 100 faces of a person(i.e 100 rows) so each row has to be given same class id
		# prepare a array of ones of size 100 and multiply by class id and store in labels
		target = class_id*np.ones((data_item.shape[0],))
		class_id += 1
		labels.append(target)
#now add all the lists( because face_data and labels are lists of lists)
face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))
#make a 1d array of labels
face_labels=face_labels.flatten()
#print(face_dataset.shape)
#print(face_labels.shape)
#print(face_labels)


#testing

while True:
	ret,frame = cap.read()
	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(frame,1.3,5)
	if(len(faces)==0):
		continue

	for face in faces:
		x,y,w,h = face

		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		#Predicted Label (out)
		out = knn(face_dataset,face_labels,face_section.flatten())

		#Display on the screen the name and rectangle around it
		pred_name = names[int(out)]
		#puttext takes frame, name to be displayed, coordinates,front,color,thickness of font, type of line
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

	cv2.imshow("Faces",frame)

	key = cv2.waitKey(1) & 0xFF
	if key==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()









