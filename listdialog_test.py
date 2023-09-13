import sys
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QItemDelegate


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('信息输出栏')

        # 创建一个表格控件
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(['x', 'y'])

        # 添加多行信息
        self.add_row(1, 2)
        self.add_row(3, 4)
        self.add_row(5, 6)

        # 将第一列设置为只读
        delegate = QItemDelegate()
        self.table_widget.setItemDelegateForColumn(0, delegate)

        # 连接单元格数据变化的信号
        self.table_widget.cellChanged.connect(self.handle_cell_changed)

        # 创建一个垂直布局，并将表格控件加入其中
        layout = QVBoxLayout()
        layout.addWidget(self.table_widget)
        self.setLayout(layout)

    def add_row(self, x, y):
        # 获取当前表格的行数
        row_count = self.table_widget.rowCount()

        # 在表格中插入一行
        self.table_widget.insertRow(row_count)

        # 设置每个单元格的数据
        item_x = QTableWidgetItem(str(x))
        item_y = QTableWidgetItem(str(y))
        self.table_widget.setItem(row_count, 0, item_x)
        self.table_widget.setItem(row_count, 1, item_y)

    def handle_cell_changed(self, row, column):
        if column == 0:  # 只处理第一列的变化
            item_x = self.table_widget.item(row, 0)
            item_y = self.table_widget.item(row, 1)
            if item_x is not None and item_y is not None:
                x = int(item_x.text())
                y = 2 * x  # 根据需要修改对应的公式
                item_y.setText(str(y))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
