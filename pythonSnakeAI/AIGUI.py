from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QMenuBar, QAction, qApp, QFileDialog
import AI
import sys


#dwa layouty zasysające wykres i okienko gry
#menu bar



class AIController(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Controller")
        self.setGeometry(700,700,700,700)
        self._createMenu()


    def _createMenu(self):
        '''
        Funkcja tworząca pasek menu
        :return:
        '''
        menuBar = self.menuBar()
        menuBar.setNativeMenuBar(False)
        self.setMenuBar(menuBar)

        exitAct = QAction('&Save and Exit',self)
        exitAct.triggered.connect(qApp.quit) #to trzeba będzie zmienić jeszcze żeby zapisywało

        openAct = QAction('&Open',self)
        openAct.triggered.connect(self.open)

        fileMenu = menuBar.addMenu("&File")
        fileMenu.addAction("&New")
        fileMenu.addAction(openAct)
        fileMenu.addAction(exitAct)


    def open(self):
        """
        Funkcja otwierająca i startująca AI, otwiera model, określa na podstawie nazwy pliku nazwe modelu
        Jeśli nie istnieje plik o danej nazwie popup window pyta się czy go utworzyć
        :return:
        """
        filename = QFileDialog.getOpenFileName()
        filename = filename[0]
        filename = filename.split('/')
        filename = filename[-1].split('.')
        filename = filename[0]
        AI.train(filename)
        print(filename)



    def save(self):
        """
        Funkcja zapisująca AI
        :return:
        """
        pass

    def loadPlot(self):
        '''
        Funkcja wczytująca wykres dla AI
        :return:
        '''
        pass

    def loadGameWindow(self):
        '''
        Funkcja wczytująca główne okienko gry
        :return:
        '''
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AIController()
    win.show()
    sys.exit(app.exec_())