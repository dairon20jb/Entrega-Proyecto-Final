import os
from tkinter import *
import cv2
from PIL import Image, ImageTk
import imutils
import shutil
window = Tk()  # Se crea la ventana de la interfaz
window.title("Reconocimiento Facial")
window.geometry('200x165') #Dimensiones iniciales de la ventana
width, height = 200, 165
cap = cv2.VideoCapture(1) #Seleccionando la cámara que se usará para la implementación
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
lbl = Label(window, text="Digite cantidad de fotos:")
dataPath = 'C:\PRUEBA\Data'  # Cambia a la ruta donde quieras que se almacene Data
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)
face_recognizer = cv2.face.EigenFaceRecognizer_create() #Método seleccionado
# face_recognizer = cv2.face.FisherFaceRecognizer_create()
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# Leyendo el modelo
face_recognizer.read('modeloEigenFace.xml')
# face_recognizer.read('modeloFisherFace.xml')
# face_recognizer.read('modeloLBPHFace.xml')
#cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# cap = cv2.VideoCapture('prueba.mp4') #En caso de realizar pruebas con videos
# Se hace uso de haarcascade para encontrar los rostros en la imagen
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def show_frame():
    #Redimensión de la ventana una vez se abre la Cámara
       window.geometry('325x400')
       width, height = 325,400
       lbl4 = Label(window, text="Digite cantidad de fotos:")
       lbl.grid(column=0, row=0)  #Ubicación de los botones y texto de la interfaz
       btn.grid(column=0, row=4)
       btn1.grid(column=0, row=8)
       btn3.grid(column=0, row=12)
       lbl4.grid(column=0, row=16)
       txt.grid(column=0, row=20)
       txt1.grid(column=0, row=28)
       lbl2.grid(column=0, row=24)
       ret, frame = cap.read()
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       auxFrame = gray.copy()
       faces = faceClassif.detectMultiScale(gray, 1.15, 5) #Sensibilidad del detecto de rostros
       for (x, y, w, h) in faces:
              rostro = auxFrame[y:y + h, x:x + w]
              rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
              result = face_recognizer.predict(rostro)
              cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
              # Condiciones para encontrar un rostro conocido
              if result[1] < 5000:
                     cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1,(0, 255, 0), 1, cv2.LINE_AA)
                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
              else:
                     cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Abrir la camara en la interfaz
       cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
       img = Image.fromarray(cv2image)
       imgtk = ImageTk.PhotoImage(image=img)
       lbl.imgtk = imgtk
       lbl.configure(image=imgtk)
       lbl.after(10, show_frame)
def fotos():
       personName = txt1.get()
       personPath = dataPath + '/' + personName
       if not os.path.exists(personPath):
           print('Carpeta creada: ', personPath)
           os.makedirs(personPath)
       cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
       # cap = cv2.VideoCapture('Video.mp4')
       faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
       count = 0
       while True:
           ret, frame = cap.read()
           if ret == False: break
           frame = imutils.resize(frame, width=640)
           gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
           auxFrame = frame.copy()
           faces = faceClassif.detectMultiScale(gray, 1.15 , 5) #Sensibilidad del detector de rostros
           # Recorta el rostro de cada frame y lo almacena
           for (x, y, w, h) in faces:
               cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) #Dibujar rectangulo alrededor del rostro
               rostro = auxFrame[y:y + h, x:x + w]
               rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
               cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count), rostro)
               count = count + 1
           cv2.imshow('frame', frame)
           k = cv2.waitKey(1)   #Se repite hasta que se capturen la cantidad de fotos deseadas por el usuario
           if k == 27 or count >= (int(txt.get())):
               break
       cap.release()
       cv2.destroyAllWindows()
def entrenar():
    peopleList = os.listdir(dataPath)
    print('Lista de personas: ', peopleList)
    labels = []
    facesData = []
    label = 0
    for nameDir in peopleList:
        personPath = dataPath + '/' + nameDir
        print('Leyendo las imágenes')
        for fileName in os.listdir(personPath):
            print('Rostros: ', nameDir + '/' + fileName)
            labels.append(label)
            facesData.append(cv2.imread(personPath+'/'+fileName,0))
            #image = cv2.imread(personPath+'/'+fileName,0)
            #cv2.imshow('image',image)
            #cv2.waitKey(10)
        label = label + 1
    print('labels= ',labels)
    print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
    print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))
    print('Número de etiquetas 2: ',np.count_nonzero(np.array(labels)==2))
    print('Número de etiquetas 3: ',np.count_nonzero(np.array(labels)==3))
    print('Número de etiquetas 4: ',np.count_nonzero(np.array(labels)==4))
    print('Número de etiquetas 5: ',np.count_nonzero(np.array(labels)==4))
    print('Número de etiquetas 6: ',np.count_nonzero(np.array(labels)==4))
    print('Número de etiquetas 7: ',np.count_nonzero(np.array(labels)==4))
    print('Número de etiquetas 8: ',np.count_nonzero(np.array(labels)==4))
    print('Número de etiquetas 9: ',np.count_nonzero(np.array(labels)==4))
    print('Número de etiquetas 10: ',np.count_nonzero(np.array(labels)==4))
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    # Entrenando el reconocedor de rostros
    print("Entrenando...")  #Se entrena el método y una vez entrenado se almacena
    face_recognizer.train(facesData, np.array(labels))
    face_recognizer.write('modeloEigenFace.xml')
    #face_recognizer.write('modeloFisherFace.xml')
    #face_recognizer.write('modeloLBPHFace.xml')
    print("Modelo almacenado...")
txt = Entry(window,width=10)
txt1 = Entry(window,width=10)
btn = Button(window, text="Iniciar reconocimiento", command=show_frame)
btn1 = Button(window, text="Tomar fotos para entrenar método", command=fotos)
btn3 = Button(window, text="Entrenar método", command=entrenar)
lbl2 = Label(window, text="Nombre:")
btn.grid(column=0, row=0)
btn1.grid(column=0, row=4)
btn3.grid(column=0, row=8)
lbl.grid(column=0, row=12)
txt.grid(column=0, row=16)
txt1.grid(column=0, row=24)
lbl2.grid(column=0, row=20)
window.mainloop()

