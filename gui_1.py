import tkinter as tk
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import re
import pickle
import sys
from HashTable import HashTable

'''
Things to fix:
1) fix the img capturing speed issue(multithreading)
2) data structure
3) unfreeze tkinter
'''

colors = {"blue": (255,0,0), "red": (34,34,178), "green": (152, 251, 152), "white": (255,255,255), "black": (0,0,0), 'yellow': (210,250,250)}

class Person:
    def __init__(self, lastName = "", firstName = "", ID = "", email = ""):
        self.lastName = lastName
        self.firstName = firstName
        self.ID = ID
        self.email = email
    def __str__(self):
        return str(self.ID)
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
        self.confidence_max = 50
        self.rectangle_width = 4
        self.fontType = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.fileName = fileName
        self.table = table
        self.webcam = cv2.VideoCapture(0)
        self.root = tk.Tk()
        self.read_from_file()
        self.GUI()
        
    def __del__(self):
        self.root.quit()
        self.webcam.release()
        cv2.destroyAllWindows()          
        
    def generate_database(self, ID, img_num, gray_img):
        cv2.imwrite(self.samples_directory + "/Sample." + ID + "." + str(img_num) + ".jpg", gray_img)
        
    def capture_images(self, data):
        if '' in data:
            print('Incomplete Field(s)!')
            return
         
        self.table.insert(Person(data[0], data[1], data[2], data[3]))
        self.write_to_file()
        
        for i in range(self.sample_size):
            gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray_img, self.scaleFactor, self.minNeighbour)
            
            for x,y,w,h in faces:
                self.generate_database(data[2], i, gray_img[y:y+h, x:x+w])
                cv2.rectangle(self.img, (x,y), (x+w, y+h), colors['red'], self.rectangle_width)
                cv2.putText(self.img, "ID-" + data[2] + " | img #" + str(i), (x, y+h+20), self.fontType, 1, colors['yellow'], 2)                
                cv2.waitKey(10)
              
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
        msg = "?"
        self.recognizer.read("Recognizer/trainingData.yml")        
        while True:
            gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray_img, self.scaleFactor, self.minNeighbour)
            for x,y,w,h in faces:
                ID, conf = self.recognizer.predict(gray_img[y:y+h, x:x+w])
                if conf < self.confidence_max:
                    cv2.rectangle(self.img, (x,y), (x+w, y+h), colors['green'], self.rectangle_width)
                    vertical_position = y+h
                    data = self.table.search(str(ID))
                    if data is not None:
                        data_list = [data.firstName + ' ' + data.lastName, data.ID, data.email]
                        for i in data_list:
                            vertical_position += 20
                            cv2.putText(self.img, i, (x, vertical_position), self.fontType, 1, colors['yellow'], 2)
                else:
                    cv2.rectangle(self.img, (x,y), (x+w, y+h), colors['black'], self.rectangle_width)
                    cv2.putText(self.img, msg, (x, y+h+20), self.fontType, 1, colors['red'], 2)
            cv2.imshow("Press 'esc' to quit", self.img)
            k = cv2.waitKey(100)
            if k is 27: #esc key
                break
            
    def write_to_file(self):
        pickle.dump(self.table, open(self.fileName, "wb"))
    
    def read_from_file(self):
        try:
            self.table = pickle.load(open(self.fileName, "rb"))
        except:
            print("Empty File!")
        
    def GUI(self):
        information = [None for i in range(4)] # list to hold first name, last name, ID, and email
        W_HEIGHT = 480
        W_WIDTH = 1000  
        self.root.title('CIS 22C Project')
        self.root.configure(background='white')
        self.root.geometry(str(W_WIDTH)+'x'+str(W_HEIGHT))
        self.root.bind('<Escape>', lambda e: self.__del__()) 
        
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, W_WIDTH*2/3)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, W_HEIGHT)          
        self.cam_frame = tk.Label(self.root, borderwidth=0, bg = 'black')
        self.cam_frame.grid(row = 0, column = 0)
        
        frame = tk.Frame(self.root, width = W_WIDTH/2, height = W_HEIGHT, bg = 'white')
        frame.grid(row = 0, column = 1)              
        
        # Labels
        tk.Label(frame, text="First Name", bg = 'white').grid(row=1)
        tk.Label(frame, text="Last Name", bg = 'white').grid(row=2)
        tk.Label(frame, text="ID", bg = 'white').grid(row=3)
        tk.Label(frame, text="Email", bg = 'white').grid(row=4)        
        
        #Entry boxes
        firstName_E = tk.Entry(frame, textvariable=information[0])
        firstName_E.grid(row=1, column=1)
        lastName_E = tk.Entry(frame, textvariable=information[1])
        lastName_E.grid(row=2, column=1)
        ID_E = tk.Entry(frame, textvariable=information[2])
        ID_E.grid(row=3, column=1)
        email_E = tk.Entry(frame, textvariable=information[3])
        email_E.grid(row=4, column=1)
                
        # Buttons
        recognition_button = tk.Button(frame, fg = 'black', text = "Start Recognition", command = lambda: self.recognize()).grid(row = 0, column = 1)
        add_B = tk.Button(frame, fg = 'black', text = "Add", command = lambda: map(lambda: x, [self.capture_images([lastName_E.get(), firstName_E.get(), ID_E.get(), email_E.get()]), firstName_E.delete(first=0,last=30), lastName_E.delete(first=0,last=30), ID_E.delete(first=0,last=30), email_E.delete(first=0,last=30)])).grid(row=5, column = 1)
        quit_B = tk.Button(frame, fg = 'red', text = "Exit", command = lambda:self.__del__()).grid(row = 9, column = 1)
        self.show_frame()
        tk.mainloop()
        
    def show_frame(self):
        self.ret, self.img = self.webcam.read()
        frame = cv2.flip(self.img, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)  
        self.cam_frame.imgtk = imgtk
        self.cam_frame.configure(image=imgtk)
        self.cam_frame.after(10, self.show_frame)
        
        
if __name__ is "__main__":
    fileName = 'students.csv'
    table = HashTable(10)
    app = FaceRecognitionApplication(fileName, table)