from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QMenuBar, QAction, qApp, QVBoxLayout
import AI
import sys

#TODO: 30.01 lista pod przyciskiem QVBoxLayout może
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

        fileMenu = menuBar.addMenu("&File")
        fileMenu.addAction("&New")
        fileMenu.addAction("&Open")
        fileMenu.addAction(exitAct)


    def open(self):
        """
        Funkcja otwierająca i startująca AI, otwiera model, określa na podstawie nazwy pliku nazwe modelu
        :return:
        """
        return

    def save(self):
        """
        Funkcja zapisująca AI
        :return:
        """
        return

    def generateButtons(self):
        '''
        Funkcja tworząca odpowiednią ilość przycisków na start programu (po jednym dla każdego wczytanego AI)
        Zasysa ilość plików i tworzy je w forze https://stackoverflow.com/questions/54927194/python-creating-buttons-in-loop-in-pyqt5/54929235
        Do usunięcia
        :return:
        '''
        pass

    def newButton(self):
        '''
        Funkcja generująca nowy przycisk przy stworzeniu nowego AI
        Używane tylko w sytuacji użycia New w menu bar
        Do usunięcia
        :return:
        '''
        pass

    def openList(self):
        '''
        Funkcja otwierająca liste atrybutów pod wciśniętym przyciskiem
        Do usunięcia
        :return:
        '''
        self.label1.show()
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