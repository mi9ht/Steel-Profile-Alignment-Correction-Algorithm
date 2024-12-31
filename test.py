import sys
import traceback
import math
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFrame, QMainWindow
from PyQt5.QtCore import QTimer, Qt, QPointF
from PyQt5.QtGui import QColor, QPainter, QFont, QPen, QBrush
from net import *
from environment import *
from train import *


env = MyEnv(action_dim1, action_dim2)
agent = Agent(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim1=action_dim1,
        action_dim2=action_dim2,
        lr=lr,
        gamma=gamma,
        e=epsilon,
        target_update=target_update,
        model=model,
        num_layers=num_layers
    )
agent.load_net('net_Double DQN_0.2_0.2')


def get_angle(p1, p2):
    delta_x = p2.x() - p1.x()
    delta_y = p2.y() - p1.y()
    angle_radians = math.atan2(delta_y, delta_x)
    angle_degrees = math.degrees(angle_radians)
    return 90 - angle_degrees


class TextDisplay(QLabel):
    def __init__(self, text):
        super().__init__()
        self.setText(text)
        self.setFont(QFont('Arial', 30))
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black; color: white;")
        self.setMaximumSize(900, 50)


class Canvas(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("border: 2px solid black;")
        self.setMinimumSize(600, 500)
        self.points = []
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.run = False
        self.rise = False
        self.working = False
        self.changed = False
        self.isFalse = False
        self.FPS = 10
        self.steel_length = steel_length * 100

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(QColor(255, 0, 0), 10))
        if len(self.points) == 2:
            painter.drawLine(self.points[0], self.points[1])
            
        painter.setPen(QPen(QColor(0, 0, 255), 2))
        painter.drawLine(500, 0, 500, 500)
        painter.setPen(QPen(QColor(255, 0, 255), 2))
        painter.drawLine(100, 0, 100, 500)
        
        if self.rise:
            painter.setPen(QPen(QColor(0, 0, 255), 2))
            brush = QBrush(QColor(0, 0, 255))
            painter.setBrush(brush)
            painter.drawRect(500 - 5, self.points[self.machine].y() - 25, 10, 50)
        
        if self.working:
            painter.setPen(QPen(QColor(0, 0, 0), 5))
            brush = QBrush(QColor(0, 0, 0))
            painter.setBrush(brush)
            painter.drawRect(500 - 5, self.points[self.machine].y() - 25, 10, 50)

    def generate_points(self):
        self.timer.stop()
        self.changed = False
        self.working = False
        self.isFalse = False
        self.state = env.reset()
        action1, action2 = agent.best_action(self.state)
        self.next_state, _, _, _, arrive_t = env.step(self.state, action1, action2)
        if self.next_state[1] == -1:
            self.isFalse = True
        delta_t, _ = self.state
        v = L * 100 / arrive_t
        x1, y1 = 100, 150
        x2, y2 = 100, 350
        if delta_t <= 0:
            x2 -= (-delta_t) * v
            y1 = y2 - (self.steel_length ** 2 - (x1 - x2) ** 2) ** 0.5
        else:
            x1 -= delta_t * v
            y2 = y1 + (self.steel_length ** 2 - (x2 - x1) ** 2) ** 0.5
        self.points = [QPointF(x1, y1), QPointF(x2, y2)]
        self.angle = get_angle(*self.points)
        self.to_rise_t = env.dis_to_con(action1, action_dim1, a1_min, a1_max)
        self.rise_idx = 100 + self.to_rise_t * v
        self.working_idx = self.rise_idx + t_machine * v
        self.control_t = env.dis_to_con(action2, action_dim2, a2_min, a2_max)
        self.machine = 0 if self.control_t < 0 else 1
        if abs(self.angle) < 0.5 or self.control_t == 0:
            self.changed = True
        
        super().update()
    
    def start_moving(self):
        if not self.points or self.isFalse:
            return
        self.run = True
        self.timer.start(self.FPS)

    def update(self):
        if self.run:
            for i in range(2):
                if self.rise or self.working:
                    self.points[i] += QPointF(self.state[1], 0)
                else:
                    self.points[i] += QPointF(self.state[1] * self.FPS, 0)
                if self.points[i].x() >= 650:
                    self.run = False
                    self.timer.stop()
            if self.points[self.machine].x() >= self.rise_idx and not self.rise and not self.working and not self.changed:
                self.rise = True
            if self.points[self.machine].x() >= self.working_idx and not self.working and not self.changed:
                self.rise = False
                self.working = True
            if self.points[self.machine].x() >= 500 and not self.changed:
                if self.machine:
                    self.target_x = self.points[self.machine].x() - self.next_state[0] * self.next_state[1]
                else:
                    self.target_x = self.points[self.machine].x() + self.next_state[0] * self.next_state[1]
                self.run = False
        
        if self.points[self.machine].x() >= 500 and not self.changed:
            self.points[(self.machine + 1) % 2] += QPointF(self.state[1], 0)
            dy = (self.steel_length ** 2 - (self.points[self.machine].x() - self.points[(self.machine + 1) % 2].x()) ** 2) ** 0.5
            if self.machine:
                dy = -dy
            self.points[(self.machine + 1) % 2].setY(self.points[self.machine].y() + dy)
            if self.points[(self.machine + 1) % 2].x() >= self.target_x:
                self.state = self.next_state
                self.angle = get_angle(*self.points)
                self.working = False
                self.changed = True
                self.run = True
        super().update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("输送链型钢位姿纠正算法测试")
        self.setGeometry(100, 100, 700, 600)
        
        self.text_timer = QTimer(self)
        self.text_timer.timeout.connect(self.update_text)
        self.text_timer.start(10)

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        
        self.test_display1 = TextDisplay('test')
        main_layout.addWidget(self.test_display1)
        self.test_display2 = TextDisplay('')
        main_layout.addWidget(self.test_display2)
        
        self.canvas = Canvas(self)
        self.canvas.setFrameShape(QFrame.StyledPanel)
        main_layout.addWidget(self.canvas)
        
        generate_button = QPushButton("生成随机角度的型材", self)
        generate_button.clicked.connect(self.canvas.generate_points)
        main_layout.addWidget(generate_button)
        
        start_button = QPushButton("开始测试", self)
        start_button.clicked.connect(self.canvas.start_moving)
        main_layout.addWidget(start_button)
        
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def update_text(self):
        if self.canvas.points:
            self.test_display1.setText('Δt: %.2f, v: %.2f, θ: %.2f°' % 
                                    (self.canvas.state[0],
                                    self.canvas.state[1],
                                    self.canvas.angle))
            if not self.canvas.isFalse:
                self.test_display2.setText('%s: %.2f' % 
                                        ('Left' if self.canvas.control_t < 0 else 'Right',
                                        abs(self.canvas.control_t)))
            else:
                self.test_display2.setText('FALSE')


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    print("Uncaught exception", file=sys.stderr)
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit(1)


if __name__ == '__main__':
    sys.excepthook = handle_exception
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
