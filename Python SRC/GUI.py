import sys
import serial
import serial.tools.list_ports
import time
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QSlider, QLabel, QVBoxLayout, QHBoxLayout, 
                             QDialog, QComboBox, QFrame, QWidget, QFileDialog, QMessageBox, QTextEdit, QLineEdit)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

class StyleHelper:
    @staticmethod
    def get_style():
        return """
        QWidget {
            background-color: #2c3e50;
            color: #ecf0f1;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        QPushButton {
            background-color: #3498db;
            border: none;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            font-size: 16px;
            margin: 4px 2px;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #2980b9;
        }
        QSlider::groove:horizontal {
            border: 1px solid #bbb;
            background: white;
            height: 10px;
            border-radius: 4px;
        }
        QSlider::sub-page:horizontal {
            background: #3498db;
            border: 1px solid #777;
            height: 10px;
            border-radius: 4px;
        }
        QSlider::add-page:horizontal {
            background: #bbb;
            border: 1px solid #777;
            height: 10px;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #3498db;
            border: 1px solid #777;
            width: 20px;
            margin-top: -5px;
            margin-bottom: -5px;
            border-radius: 10px;
        }
        QComboBox, QLineEdit {
            border: 1px solid #3498db;
            border-radius: 3px;
            padding: 5px;
            min-width: 6em;
            background-color: #34495e;
        }
        QTextEdit {
            background-color: #34495e;
            border: 1px solid #3498db;
            font-family: Consolas, Monaco, monospace;
            font-size: 14px;
        }
        QLabel {
            font-size: 16px;
        }
        """

class CombinedUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.arduino = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.csv_file = None
        self.csv_writer = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Arduino Water Flow Control and Serial Monitor')
        self.setGeometry(100, 100, 1000, 800)
        self.setStyleSheet(StyleHelper.get_style())

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel for Arduino Control
        left_panel = QVBoxLayout()

        # Port selection
        port_layout = QHBoxLayout()
        self.port_combo = QComboBox()
        self.refresh_ports()
        port_layout.addWidget(QLabel('Select Port:'))
        port_layout.addWidget(self.port_combo)
        refresh_button = QPushButton('Refresh')
        refresh_button.clicked.connect(self.refresh_ports)
        port_layout.addWidget(refresh_button)
        left_panel.addLayout(port_layout)

        # Connect button
        self.connect_button = QPushButton('Connect')
        self.connect_button.clicked.connect(self.toggle_connection)
        left_panel.addWidget(self.connect_button)

        # Start/Stop buttons
        self.start_button = QPushButton('Start Monitoring')
        self.start_button.clicked.connect(self.start_monitoring)
        self.stop_button = QPushButton('Stop Monitoring')
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.stop_button.setEnabled(False)
        left_panel.addWidget(self.start_button)
        left_panel.addWidget(self.stop_button)

        # Flow control
        self.flow_slider = QSlider(Qt.Horizontal)
        self.flow_slider.setMinimum(0)
        self.flow_slider.setMaximum(100)
        self.flow_slider.setValue(50)
        self.flow_slider.setTickPosition(QSlider.TicksBelow)
        self.flow_slider.setTickInterval(10)
        self.flow_label = QLabel('Water Flow Rate: 50%')
        self.flow_slider.valueChanged.connect(self.update_flow_label)
        left_panel.addWidget(self.flow_label)
        left_panel.addWidget(self.flow_slider)

        # Data display
        self.timestamp_label = QLabel('Timestamp: ')
        self.moisture_label = QLabel('Moisture: ')
        self.flow_rate_label = QLabel('Water Flow Rate: ')
        self.movement_x_label = QLabel('Movement X: ')
        self.movement_y_label = QLabel('Movement Y: ')
        self.movement_z_label = QLabel('Movement Z: ')
        
        for label in [self.timestamp_label, self.moisture_label, self.flow_rate_label,
                      self.movement_x_label, self.movement_y_label, self.movement_z_label]:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("background-color: #34495e; padding: 10px; border-radius: 5px;")
            left_panel.addWidget(label)

        # Event marking buttons
        self.slump_button = QPushButton('Mark Slump')
        self.slump_button.clicked.connect(lambda: self.mark_event('Slump'))
        self.debris_button = QPushButton('Mark Debris')
        self.debris_button.clicked.connect(lambda: self.mark_event('Debris'))
        self.slide_button = QPushButton('Mark Slide')
        self.slide_button.clicked.connect(lambda: self.mark_event('Slide'))
        left_panel.addWidget(self.slump_button)
        left_panel.addWidget(self.debris_button)
        left_panel.addWidget(self.slide_button)

        main_layout.addLayout(left_panel)

        # Right panel for Serial Monitor
        right_panel = QVBoxLayout()

        # Monitor
        self.monitor = QTextEdit()
        self.monitor.setReadOnly(True)
        right_panel.addWidget(self.monitor)

        # Input
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.returnPressed.connect(self.send_data)
        input_layout.addWidget(self.input_field)
        send_button = QPushButton('Send')
        send_button.clicked.connect(self.send_data)
        input_layout.addWidget(send_button)
        right_panel.addLayout(input_layout)

        main_layout.addLayout(right_panel)

    def refresh_ports(self):
        self.port_combo.clear()
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo.addItems(ports)

    def toggle_connection(self):
        if self.arduino is None:
            try:
                port = self.port_combo.currentText()
                self.arduino = serial.Serial(port, 9600, timeout=0.1)
                self.connect_button.setText('Disconnect')
                self.monitor.append(f"Connected to {port}")
                self.timer.start(10)
            except serial.SerialException as e:
                self.monitor.append(f"Error: {str(e)}")
        else:
            self.stop_monitoring()
            self.arduino.close()
            self.arduino = None
            self.connect_button.setText('Connect')
            self.monitor.append("Disconnected")
            self.timer.stop()

    def update_flow_label(self, value):
        self.flow_label.setText(f'Water Flow Rate: {value}%')
        if self.arduino:
            self.arduino.write(f"FLOW:{value}\n".encode())

    def start_monitoring(self):
        if not self.arduino:
            QMessageBox.warning(self, "Not Connected", "Please connect to a port first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.csv_file = open(file_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['Timestamp', 'Water Flow Rate', 'Moisture', 'Movement X', 'Movement Y', 'Movement Z', 'Event'])

            # Send START command after file is created
            try:
                self.arduino.write(b"START\n")
            except Exception as e:
                self.monitor.append(f"Failed to start monitoring: {e}")
                return

            self.timer.start(1000)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.monitor.append("Monitoring started")

    def stop_monitoring(self):
        self.timer.stop()
        if self.arduino:
            self.arduino.write(b"STOP\n")
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.monitor.append("Monitoring stopped")

    def update_data(self):
        if self.arduino and self.arduino.in_waiting:
            try:
                data = self.arduino.readline().decode('utf-8').strip()
                self.monitor.append(data)
                timestamp, flow_rate, moisture, movement_x, movement_y, movement_z = data.split(',')
                
                # Update labels
                self.timestamp_label.setText(f'Timestamp: {timestamp}')
                self.moisture_label.setText(f'Moisture: {moisture}')
                self.flow_rate_label.setText(f'Water Flow Rate: {flow_rate}')
                self.movement_x_label.setText(f'Movement X: {movement_x}')
                self.movement_y_label.setText(f'Movement Y: {movement_y}')
                self.movement_z_label.setText(f'Movement Z: {movement_z}')

                # Write to CSV
                if self.csv_writer:
                    self.csv_writer.writerow([timestamp, flow_rate, moisture, movement_x, movement_y, movement_z, ''])
                    self.csv_file.flush()

            except Exception as e:
                self.monitor.append(f"Error decoding data: {e}")

    def mark_event(self, event_type):
        if self.csv_writer:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self.csv_writer.writerow([timestamp, '', '', '', '', '', event_type])
            self.csv_file.flush()
            self.monitor.append(f"{event_type} event marked at {timestamp}")

    def send_data(self):
        if self.arduino:
            data = self.input_field.text() + '\n'
            self.arduino.write(data.encode())
            self.monitor.append(f"Sent: {data.strip()}")
            self.input_field.clear()

    def closeEvent(self, event):
        self.stop_monitoring()
        if self.arduino:
            self.arduino.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CombinedUI()
    ex.show()
    sys.exit(app.exec_())