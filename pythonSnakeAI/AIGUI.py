
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QFileDialog, QLabel, QLineEdit, QWidget, QPushButton, qApp
from AI import train
from PyQt5.QtCore import QFileSystemWatcher
from os.path import isfile
import sys



class FilePopup(QWidget):
    '''
    Klasa tworząca okienko do tworzenia plików.
    '''
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Create New File")
        self.setGeometry(400,200,400,200)

        self.text = QLineEdit(self)
        self.text.move(20,20)
        self.text.resize(360,30)
        self.text.setText("./model/")

        self.b1 = QPushButton(self)
        self.b1.setText("Create")
        self.b1.move(20,60)
        self.b1.clicked.connect(self.createCheck)


    def createCheck(self):
        '''
        Funkcja sprawdza poprawność nazwy tworzonego pliku i tworzy plik.
        '''
        filename = self.text.text()
        if filename[-4:] != ".pth":
            filename1 = filename+".pth"
            filename2 = filename+".txt"
            filename2 = filename2.replace('./model/', './files/')
        #print(filename2)
        if not isfile(filename1):
            with open(filename1, 'w') as f:
                f.close()
        if not isfile(filename2):
            with open(filename2, 'w') as f:
                f.close()



class AIController(QMainWindow):
    '''
    Main Window.
    '''
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Controller")
        self.setGeometry(200,200,200,200)
        self.ai_train = None
        self.filename = "./files/model1.txt"
        self.watcher = QFileSystemWatcher(self)
        self.watcher.addPath(self.filename)
        self.l_score = QLabel(self)
        self.l_record = QLabel(self)
        self.l_nog = QLabel(self)
        self.l_name = QLabel(self)
        self.watcher.fileChanged.connect(self.updateInfo)
        self._createMenu()
        self._createLabels()




    def _createMenu(self):
        '''
        Funkcja tworząca pasek menu.
        '''
        menuBar = self.menuBar()
        menuBar.setNativeMenuBar(False)
        self.setMenuBar(menuBar)

        openAct = QAction('&Open',self)
        openAct.triggered.connect(self.open_ai)

        newAct = QAction('&New',self)
        newAct.triggered.connect(self.new_file)

        fileMenu = menuBar.addMenu("&File")
        fileMenu.addAction(newAct)
        fileMenu.addAction(openAct)


    def _createLabels(self):
        '''
        Funkcja dodająca opis AI.
        '''
        self.l_nog.setText("Number of games: ")
        self.l_record.setText("Record: ")
        self.l_score.setText("Score: ")
        self.l_nog.move(25,40)
        self.l_record.move(25,55)
        self.l_score.move(25,70)
        self.l_nog.adjustSize()
        self.l_record.adjustSize()
        self.l_score.adjustSize()


    def open_ai(self):
        """
        Funkcja otwierająca i startująca AI, otwiera model, określa na podstawie nazwy pliku nazwe modelu
        Ustawia label z nazwą AI.
        """

        self.filename = QFileDialog.getOpenFileName()
        if not self.filename[0] =="":
            filename = self.filename[0]
            self.filename = self.filename[0].split('.')
            self.filename = self.filename[0].split('/')

            self.l_name.setText(f'Nazwa AI: {self.filename[-1]}')
            self.l_name.move(25, 25)
            self.l_name.adjustSize()

            self.filename = "./files/" + self.filename[-1] + ".txt"
            self.watcher.addPath(self.filename)
            print(self.filename)
            # filename = filename.split('/')
            # filename = filename[-1]
            self.ai_train = train(filename)


    def new_file(self):
        '''
        Funkcja wywołuje okienko tworzące plik.
        '''
        self.nf = FilePopup()
        self.nf.show()



    def updateInfo(self, path):
        '''
        Funkcja zmieniająca informacje wyświetlające się w GUI.
        '''
        l = []
        with open(path, 'r') as f:
            l = f.readlines()
            f.close()
        print(l)
        self.l_nog.setText(f'Number of games: {l[0]}')
        self.l_score.setText(f'Score: {l[1]}')
        self.l_record.setText(f'Record: {l[2]}')
        self.l_nog.adjustSize()
        self.l_score.adjustSize()
        self.l_record.adjustSize()





if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AIController()
    win.show()
    sys.exit(app.exec_())