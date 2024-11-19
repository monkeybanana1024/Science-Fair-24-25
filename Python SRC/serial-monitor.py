import sys
import serial
import serial.tools.list_ports
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
                             QLabel, QComboBox, QTextEdit, QWidget, QLineEdit)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt, QTimer

class SerialMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.serial_port = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.read_serial)

    def initUI(self):
        self.setWindowTitle('PyQt Serial Monitor')
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2c3e50;
                color: #ecf0f1;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px;
                margin: 5px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QLabel, QComboBox, QTextEdit, QLineEdit {
                font-size: 14px;
                color: #ecf0f1;
            }
            QComboBox, QLineEdit {
                background-color: #34495e;
                border: 1px solid #3498db;
                padding: 5px;
            }
            QTextEdit {
                background-color: #34495e;
                border: 1px solid #3498db;
            }
        """)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Port selection
        port_layout = QHBoxLayout()
        self.port_combo = QComboBox()
        self.refresh_ports()
        port_layout.addWidget(QLabel('Select Port:'))
        port_layout.addWidget(self.port_combo)
        refresh_button = QPushButton('Refresh')
        refresh_button.clicked.connect(self.refresh_ports)
        port_layout.addWidget(refresh_button)
        layout.addLayout(port_layout)

        # Connect button
        self.connect_button = QPushButton('Connect')
        self.connect_button.clicked.connect(self.toggle_connection)
        layout.addWidget(self.connect_button)

        # Monitor
        self.monitor = QTextEdit()
        self.monitor.setReadOnly(True)
        layout.addWidget(self.monitor)

        # Input
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.returnPressed.connect(self.send_data)
        input_layout.addWidget(self.input_field)
        send_button = QPushButton('Send')
        send_button.clicked.connect(self.send_data)
        input_layout.addWidget(send_button)
        layout.addLayout(input_layout)

    def refresh_ports(self):
        self.port_combo.clear()
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo.addItems(ports)

    def toggle_connection(self):
        if self.serial_port is None:
            try:
                port = self.port_combo.currentText()
                self.serial_port = serial.Serial(port, 9600, timeout=0.1)
                self.connect_button.setText('Disconnect')
                self.monitor.append(f"Connected to {port}")
                self.timer.start(10)
            except serial.SerialException as e:
                self.monitor.append(f"Error: {str(e)}")
        else:
            self.serial_port.close()
            self.serial_port = None
            self.connect_button.setText('Connect')
            self.monitor.append("Disconnected")
            self.timer.stop()

    def read_serial(self):
        if self.serial_port and self.serial_port.in_waiting:
            data = self.serial_port.readline().decode('utf-8').strip()
            self.monitor.append(data)

    def send_data(self):
        if self.serial_port:
            data = self.input_field.text() + '\n'
            self.serial_port.write(data.encode())
            self.input_field.clear()

    def closeEvent(self, event):
        if self.serial_port:
            self.serial_port.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SerialMonitor()
    ex.show()
    sys.exit(app.exec_())