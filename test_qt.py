from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
import sys

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test Uygulama")
        self.setGeometry(100, 100, 300, 200)
        
        button = QPushButton("Test Butonu", self)
        button.setGeometry(100, 80, 100, 40)
        
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TestWindow()
    sys.exit(app.exec_()) 