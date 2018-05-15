import sys

from PyQt5.QtWidgets import QWidget, QApplication

from main_ui import Ui_MainWindow


class Router_Crack(QWidget,Ui_MainWindow):
    def __init__(self):
        super(Router_Crack, self).__init__()
        self.setupUi(self)
        self.setFunc()
        self.pause = False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    router_crack = Router_Crack()
    router_crack.show()
    app.exec_()
