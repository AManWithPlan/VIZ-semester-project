import json
import math
import sys

import networkx as nx
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QPen, QTransform, QPainter, QSurfaceFormat, QColor, QAction, QFont
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QFileDialog, \
    QMenuBar, QMenu
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.sparse import rand
from sklearn.cluster import KMeans

neighbours = {}


class VisGraphicsScene(QGraphicsScene):
    def __init__(self):
        super(VisGraphicsScene, self).__init__()
        self.selection = None
        self.wasDragg = False
        self.pen = QPen(Qt.black)
        self.selected = QPen(Qt.red)
        self.font = QFont('Arial', 10)

    def mouseReleaseEvent(self, event):
        if (self.wasDragg):
            return
        if (self.selection):
            self.selection.setPen(self.pen)
        item = self.itemAt(event.scenePos(), QTransform())
        if (item):
            item.setPen(self.selected)
            self.selection = item


class VisGraphicsView(QGraphicsView):
    def __init__(self, scene, parent):
        super(VisGraphicsView, self).__init__(scene, parent)
        self.startX = 0.0
        self.startY = 0.0
        self.distance = 0.0
        self.myScene = scene
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)

    def wheelEvent(self, event):
        zoom = 1 + event.angleDelta().y() * 0.001
        self.scale(zoom, zoom)

    def mousePressEvent(self, event):
        self.startX = event.pos().x()
        self.startY = event.pos().y()
        self.myScene.wasDragg = False
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        endX = event.pos().x()
        endY = event.pos().y()
        deltaX = endX - self.startX
        deltaY = endY - self.startY
        distance = math.sqrt(deltaX * deltaX + deltaY * deltaY)
        if (distance > 5):
            self.myScene.wasDragg = True
        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.reorderMatrix = None
        self.matrixMode = True
        self.loadData = self.loadFile
        self.setWindowTitle('Adjacency Matrix Visualization')
        self.createGraphicView()
        self.reorder_algorithms = [
            ("Random reorder", self.random_reorder, None),
            ("Alphabetical reorder", self.alphabetical_sort, None),
            ("Hierarchical reorder", self.cluster_adjacency_matrix, None),
            ("Sum reorder", self.order_adjacency_matrix, None),
            # Add more algorithms and their functions
        ]
        for i in range(1, 10):
            self.reorder_algorithms.append((f"kmeans, k={i}", self.kmeans_clustering, i)),
        # self.createMenu()
        # self.createReorderComboBox()
        # self.createConvertButton()  # Add the createConvertButton method
        self.createMenuBar()
        self.show()
        self.showMaximized()
        self.adjacency_matrix = None
        self.node_labels = None
        self.generateRandomData()
        self.updateVisualization()

    def createMenuBar(self):
        menu_bar = QMenuBar(self)

        # File menu
        # file_menu = QMenu("File", self)
        # menu_bar.addMenu(file_menu)

        # Load data action
        load_action = QAction("Load Data", self)
        load_action.triggered.connect(self.loadData)
        menu_bar.addAction(load_action)

        view_menu = QMenu("View", self)
        menu_bar.addMenu(view_menu)
        # menu_bar.addMenu(file_menu)
        # Convert to Matrix action
        convert_matrix_action = QAction("Convert", self)
        convert_matrix_action.triggered.connect(self.handleConvertButtonClick)
        view_menu.addAction(convert_matrix_action)

        order_menu = QMenu("Reorder", self)
        menu_bar.addMenu(order_menu)
        # Reorder combo box

        for name, func, parm in self.reorder_algorithms:
            reorder_action = QAction(name, self)
            if parm:
                reorder_action.triggered.connect(lambda: func(parm))
            else:
                reorder_action.triggered.connect(func)
            order_menu.addAction(reorder_action)

        self.setMenuBar(menu_bar)

    def handleConvertButtonClick(self):
        self.matrixMode = not self.matrixMode
        self.updateVisualization()

    def convertToForceDirectedGraph(self):
        scaler = len(self.adjacency_matrix) * 50

        def drawNodes(graph, pos):
            for node in graph.nodes():
                x = pos[node][0] * scaler + 250
                y = pos[node][1] * scaler + 250
                label = str(self.node_labels[node])
                item = self.scene.addEllipse(x - 5, y - 5, 10, 10)
                item.setBrush(QBrush(QColor(255, 255, 255)))
                item.setPen(QPen(Qt.black))
                item.setZValue(1)
                text = self.scene.addText(label)
                text.setDefaultTextColor(QColor(255, 0, 0))
                text.setPos(x + 5, y - 15)

        def drawEdges(graph, pos):
            for edge in graph.edges():
                start_node = edge[0]
                end_node = edge[1]
                start_pos = pos[start_node]
                end_pos = pos[end_node]
                line = self.scene.addLine(start_pos[0] * scaler + 250,
                                          start_pos[1] * scaler + 250,
                                          end_pos[0] * scaler + 250,
                                          end_pos[1] * scaler + 250)
                line.setPen(QPen(Qt.black))
                line.setZValue(0)

        def drawGraph(graph, pos):
            drawEdges(graph, pos)
            drawNodes(graph, pos)

        graph = nx.from_numpy_matrix(self.adjacency_matrix)
        pos = nx.spring_layout(graph, dim=2)
        drawGraph(graph, pos)

    def random_reorder(self):
        # Generate a random permutation of the node labels
        permutation = np.random.permutation(len(self.node_labels))

        # Reorder the adjacency matrix and node labels using the permutation
        self.adjacency_matrix = self.adjacency_matrix[permutation][:, permutation]
        self.node_labels = [self.node_labels[i] for i in permutation]

        self.updateVisualization()

    def alphabetical_sort(self):

        # Sort the node labels alphabetically
        sorted_indices = np.argsort(self.node_labels)
        self.node_labels = [self.node_labels[i] for i in sorted_indices]

        # Reorder the adjacency matrix using the sorted indices
        self.adjacency_matrix = self.adjacency_matrix[sorted_indices][:, sorted_indices]

        self.updateVisualization()

    def kmeans_clustering(self, n_clusters):
        # Apply kmeans clustering to the adjacency matrix
        kmeans = KMeans(n_clusters=n_clusters).fit(self.adjacency_matrix)

        # Get the cluster labels
        cluster_labels = kmeans.labels_

        # Sort the node labels by cluster
        sorted_indices = np.argsort(cluster_labels)
        self.node_labels = [self.node_labels[i] for i in sorted_indices]

        # Reorder the adjacency matrix using the sorted indices
        self.adjacency_matrix = self.adjacency_matrix[sorted_indices][:, sorted_indices]

        self.updateVisualization()

    def cluster_adjacency_matrix(self):
        Z = linkage(self.adjacency_matrix, 'ward')

        print(Z)

        # Get the node order from the linkage matrix
        idx = leaves_list(Z)
        print(idx)

        # Reorder the adjacency matrix using the linkage matrix
        self.adjacency_matrix = self.adjacency_matrix[idx, :][:, idx]
        self.node_labels = [self.node_labels[i] for i in idx]

        self.updateVisualization()

    def order_adjacency_matrix(self):
        row_sums = np.sum(self.adjacency_matrix, axis=1) + np.sum(self.adjacency_matrix, axis=0)

        # Sort the node labels by the row sums
        sorted_indices = np.argsort(row_sums)[::-1]
        self.node_labels = [self.node_labels[i] for i in sorted_indices]

        # Reorder the adjacency matrix using the sorted indices
        self.adjacency_matrix = self.adjacency_matrix[sorted_indices][:, sorted_indices]

        self.updateVisualization()

    def createGraphicView(self):
        self.scene = VisGraphicsScene()
        self.brush = [QBrush(Qt.yellow), QBrush(Qt.green), QBrush(Qt.blue)]

        format = QSurfaceFormat()
        format.setSamples(4)

        gl = QOpenGLWidget()
        gl.setFormat(format)
        gl.setAutoFillBackground(True)

        self.view = VisGraphicsView(self.scene, self)
        self.view.setViewport(gl)
        self.view.setBackgroundBrush(QColor(255, 255, 255))

        self.setCentralWidget(self.view)
        self.view.setGeometry(0, 0, 800, 600)

    def loadFile(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Json files (*.json)")
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]

            self.adjacency_matrix, self.node_labels = load(file_path)

            self.updateVisualization()

    def updateVisualization(self):
        # Clear previous visualization
        self.scene.clear()

        # if self.reorderMatrix:
        #     self.reorderMatrix()
        if self.matrixMode:
            self.convertToMatrix()
        else:
            self.convertToForceDirectedGraph()

    def convertToMatrix(self):
        data, node_labels = self.adjacency_matrix, self.node_labels
        maxv = np.max(data)
        count = len(node_labels)

        # map the data
        cell_size = 80
        # Add column labels
        col_labels = []
        for i in range(count):
            label_item = self.scene.addText(node_labels[i], self.scene.font)
            col_labels.append(label_item)

        # Adjust column label positions to prevent overlap
        for i, label_item in enumerate(col_labels):
            width = label_item.boundingRect().width()
            label_item.setPos(i * cell_size + cell_size / 2 - width / 2, -cell_size - 20)
            for j in range(i):
                prev_label_item = col_labels[j]
                prev_width = prev_label_item.boundingRect().width()
                if abs(label_item.x() - prev_label_item.x()) < (width + prev_width) / 2:
                    label_item.setPos(label_item.x(), label_item.y() - 20)

        # Add row and data items
        row_labels = []
        for i in range(count):
            # Add row label
            row_label_item = self.scene.addText(node_labels[i], self.scene.font)
            row_labels.append(row_label_item)

            for j in range(count):
                value = data[i][j]

                # Calculate color based on value range
                color_value = 1 - (value / maxv)
                color = QColor.fromHsvF(0, 0, color_value)
                brush = QBrush(color)

                rect = self.scene.addRect(j * cell_size, i * cell_size, cell_size, cell_size, self.scene.pen, brush)
                text_item = self.scene.addText(str(value), self.scene.font)
                text_item.setPos(j * cell_size + cell_size / 2 - text_item.boundingRect().width() / 2,
                                 i * cell_size + cell_size / 2 - text_item.boundingRect().height() / 2)

                # Adjust font color for better visibility
                if value == 0:
                    text_item.setDefaultTextColor(Qt.white)
                elif value < maxv * 0.6:
                    text_item.setDefaultTextColor(Qt.black)
                else:
                    text_item.setDefaultTextColor(Qt.white)

        # Adjust row label positions to prevent overlap
        for i, label_item in enumerate(row_labels):
            height = label_item.boundingRect().height()
            label_item.setPos(-cell_size - 20, i * cell_size + cell_size / 2 - height / 2)
            for j in range(i):
                prev_label_item = row_labels[j]
                prev_height = prev_label_item.boundingRect().height()
                if abs(label_item.y() - prev_label_item.y()) < (height + prev_height) / 2:
                    label_item.setPos(label_item.x() - 20, label_item.y())

    def generateRandomData(self):
        count = 15
        self.adjacency_matrix = np.floor(rand(15, 15, density=0.1, format='csr').toarray() * 10)
        self.node_labels = ['Node {}'.format(i) for i in range(count)]


def load(path):
    with open(path) as f:
        data3 = json.load(f)
    # with open('data\\\\raw.json') as f:
    #     data3 = json.load(f)
    vertices = {}
    for node in data3['vertices']:
        vertices[node['id']] = node
    vertices_to_id = list(vertices.keys())
    id_to_vertices = {v: k for k, v in enumerate(vertices_to_id)}
    edges = []
    for node in data3['edges']:
        edges.append(node)
    adjacency = [[None for i in range(len(vertices))] for j in range(len(vertices))]
    for edge in edges:
        first_id = edge['from']
        second_id = edge['to']
        first_position = id_to_vertices[first_id]
        second_position = id_to_vertices[second_id]
        if adjacency[first_position][second_position] is None:
            adjacency[first_position][second_position] = [edge]
        else:
            adjacency[first_position][second_position].append(edge)
    heatmap = np.array([[len(adjacency[i][j]) if adjacency[i][j] else 0 for i in range(len(adjacency))] for j in
                        range(len(adjacency))])

    try:
        node_labels = [vertices[node_id]['name'] for node_id in vertices_to_id]
    except:
        node_labels = [vertices[node_id]['title'] for node_id in vertices_to_id]

    return heatmap, node_labels


def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
