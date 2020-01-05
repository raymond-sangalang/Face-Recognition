import tkinter as tk
import os
import cv2
import numpy as np
from PIL import Image
import re
import pickle
import sys
from HashTable import HashTable

colors = {"blue": (255,0,0), "red": (0,0,255), "green": (0, 255, 0), "white": (255,255,255), "black": (0,0,0), 'yellow': (0, 255,255)}

class Person:
    def __init__(self, lastName = "", firstName = "", ID = "", email = ""):
        self.lastName = lastName
        self.firstName = firstName
        self.ID = ID
        self.email = email
    def __str__(self):
        getframe_expr = 'sys._getframe({}).f_code.co_name'
        caller = eval(getframe_expr.format(2))
        if caller is "insert":
            return self.ID
        return (self.firstName + ", " + self.lastName + ", " + self.ID)
        
    def __eq__(self, other):
        return self.ID == other


class FaceRecognitionApplication():
    def __init__(self, fileName, table):
        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.id_from_path_pattern = re.compile(r'[^.]*\.?([^.]*)')
        self.scaleFactor = 1.3
        self.minNeighbour = 5
        self.samples_directory = 'Data'
        self.recognizer_directory = 'Recognizer'
        self.sample_size = 100
        self.confidence_max = 18
        self.rectangle_width = 6
        self.fontType = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.fileName = fileName
        self.table = table
        self.root = tk.Tk()
        self.read_from_file()
        self.GUI()
        
    def __del__(self):
        self.root.quit()
        
    def generate_database(self, ID, img_num, gray_img):
        cv2.imwrite(self.samples_directory + "/Sample." + ID + "." + str(img_num) + ".jpg", gray_img)
        
    def capture_images(self, data):
        webcam = cv2.VideoCapture(0)
        if '' in data:
            print('Incomplete Field(s)!')
            return
         
        self.table.insert(Person(data[0], data[1], data[2], data[3]))
        self.write_to_file()
        
        for i in range(self.sample_size):
            ret,img = webcam.read()
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray_img, self.scaleFactor, self.minNeighbour)
            
            for x,y,w,h in faces:
                self.generate_database(data[2], i, gray_img[y:y+h, x:x+w])
                cv2.rectangle(img, (x,y), (x+w, y+h), colors['red'], self.rectangle_width)
                cv2.putText(img, "ID-" + data[2] + " | img #" + str(i), (x, y+h+20), self.fontType, 1, colors['yellow'], 2)                
                cv2.waitKey(10)
            cv2.imshow("Collecting Samples", img)
        
        webcam.release()
        cv2.destroyAllWindows()        
        self.train_classifier()
            
    def train_classifier(self):
        path_list = [os.path.join(self.samples_directory, f) for f in os.listdir(self.samples_directory)]
        img_list,ID_list = [],[]
        
        for path in path_list:
            try:
                img = Image.open(path).convert('L')
                np_img = np.array(img, 'uint8')
                ID = int(re.search(self.id_from_path_pattern, path).group(1))
                img_list.append(np_img)
                ID_list.append(ID)              
            except:
                pass
        self.recognizer.train(img_list, np.array(ID_list))
        self.recognizer.save(self.recognizer_directory + '/trainingData.yml')
    
    def recognize(self):
        msg = '?'
        webcam = cv2.VideoCapture(0)
        self.recognizer.read("Recognizer/trainingData.yml")        
        while True:
            ret, img = webcam.read()
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray_img, self.scaleFactor, self.minNeighbour)
            for x,y,w,h in faces:
                ID, conf = self.recognizer.predict(gray_img[y:y+h, x:x+w])
                print(conf)
                if conf < self.confidence_max:
                    cv2.rectangle(img, (x,y), (x+w, y+h), colors['green'], self.rectangle_width)
                    vertical_position = y+h
                    data = self.table.search(str(ID))
                    if data is not None:
                        cv2.putText(img, 'Confidence: High', (x+60, y-10), self.fontType, 1, colors['white'], 2)
                        data_list = [data.firstName + ' ' + data.lastName, data.ID, data.email]
                        for i in data_list:
                            vertical_position += 20
                            cv2.putText(img, i, (x, vertical_position), self.fontType, 1, colors['white'], 2)
                            
                elif conf < self.confidence_max+5:
                    cv2.rectangle(img, (x,y), (x+w, y+h), colors['yellow'], self.rectangle_width)
                    vertical_position = y+h
                    data = self.table.search(str(ID))
                    cv2.putText(img, 'Confidence: Moderate', (x+40, y-10), self.fontType, 1, colors['white'], 2)
                    if data is not None:
                        data_list = [data.firstName + ' ' + data.lastName, data.ID, data.email]
                        for i in data_list:
                            vertical_position += 20
                            cv2.putText(img, i, (x, vertical_position), self.fontType, 1, colors['white'], 2) 
                            
                elif conf < self.confidence_max+10:
                    cv2.rectangle(img, (x,y), (x+w, y+h), colors['red'], self.rectangle_width)
                    vertical_position = y+h
                    data = self.table.search(str(ID))
                    cv2.putText(img, 'Confidence: Low', (x+40, y-10), self.fontType, 1, colors['white'], 2)
                    if data is not None:
                        data_list = [data.firstName + ' ' + data.lastName, data.ID, data.email]
                        for i in data_list:
                            vertical_position += 20
                            cv2.putText(img, i, (x, vertical_position), self.fontType, 1, colors['white'], 2)                
                elif conf < self.confidence_max+40:
                    cv2.rectangle(img, (x,y), (x+w, y+h), colors['black'], self.rectangle_width)
                    cv2.putText(img, msg, (x+int(w/2)-50, y-30), self.fontType, 6, colors['red'], 2)
            cv2.imshow("Press 'esc' to quit", img)
            k = cv2.waitKey(100)
            if k is 27: #esc key
                break
        webcam.release()
        cv2.destroyAllWindows()
            
    def write_to_file(self):
        pickle.dump(self.table, open(self.fileName, "wb"))
    
    def read_from_file(self):
        try:
            self.table = pickle.load(open(self.fileName, "rb"))
            self.table.display()
        except:
            print("Empty File!")
        
    def GUI(self):
        information = [None for i in range(4)] # list to hold first name, last name, ID, and email
        W_HEIGHT = 180
        W_WIDTH = 350
          
        self.root.title('CIS 22C Project')
        self.root.configure(background='white')
        self.root.geometry(str(W_WIDTH)+'x'+str(W_HEIGHT))
        self.root.bind('<Escape>', lambda e: self.__del__()) 
        
        # Labels
        tk.Label(self.root, text="First Name").grid(row=1)
        tk.Label(self.root, text="Last Name").grid(row=2)
        tk.Label(self.root, text="ID").grid(row=3)
        tk.Label(self.root, text="Email").grid(row=4)        
        
        #Entry boxes
        firstName_E = tk.Entry(self.root, textvariable=information[0])
        firstName_E.grid(row=1, column=1)
        lastName_E = tk.Entry(self.root, textvariable=information[1])
        lastName_E.grid(row=2, column=1)
        ID_E = tk.Entry(self.root, textvariable=information[2])
        ID_E.grid(row=3, column=1)
        email_E = tk.Entry(self.root, textvariable=information[3])
        email_E.grid(row=4, column=1)
                
        # Buttons
        recognition_button = tk.Button(self.root, text = "Start Recognition", command = lambda: self.recognize()).grid(row = 0, column = 1)
        add_B = tk.Button(self.root, text = "Add", command = lambda: map(lambda: x, [self.capture_images([lastName_E.get(), firstName_E.get(), ID_E.get(), email_E.get()]), firstName_E.delete(first=0,last=30), lastName_E.delete(first=0,last=30), ID_E.delete(first=0,last=30), email_E.delete(first=0,last=30)])).grid(row=5, column = 1)
        quit_B = tk.Button(self.root, text = "Exit", command = lambda:self.__del__()).grid(row = 9, column = 1)        
        
        tk.mainloop()
        
if __name__ is "__main__":
    fileName = 'students.csv'
    table = HashTable(100)
    app = FaceRecognitionApplication(fileName, table)