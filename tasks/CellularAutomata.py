from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import QTimer
import sys
import numpy as np
import pickle

# Conway's game of life

class SimulationData:
    def __init__(self, random_seed, iterations):
        self.random_seed = random_seed
        self.iterations = iterations

np.random.seed(420)
simulation_data = SimulationData(420, [])

class GridWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Game of Life")
        self.setGeometry(50, 50, 640, 640)
        self.iterationList = []

        self.grid_size = 128
        self.square_size = 4
        self.iterations = 999
       
        # Initalize the grid randomly with a density of 19%
        density = 0.19
        num_ones = int(self.grid_size**2 * density)
        self.grid = np.hstack([np.ones(num_ones), np.zeros(self.grid_size**2 - num_ones)])
        np.random.shuffle(self.grid)
        self.grid = self.grid.reshape([self.grid_size, self.grid_size])
        
        self.counter = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_grid)
        self.timer.start(100)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QColor(128, 128, 128))
        size = self.square_size * self.grid_size

        start_x = (self.width() - size) // 2
        start_y = (self.height() - size) // 2

        # Draw 128x128 grid of squares
        for x in range(128):
            for y in range(128):
                if self.grid[x, y] == 1:  # If the cell is 1, draw a rectangle
                    painter.fillRect(start_x + x*self.square_size, start_y + y*self.square_size, self.square_size, self.square_size, QColor(128, 128, 128))
    
    def update_grid(self):        
        """
        Conway's game of life has four rules:
            1. Any live cell with fewer than two live neighbours dies, as if by underpopulation.
            2. Any live cell with two or three live neighbours lives on to the next generation.
            3. Any live cell with more than three live neighbours dies, as if by overpopulation.
            4. Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        """
        self.iterationList.append(self.grid)

        kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
        self.grid = self.convolve_dead_or_alive(self.grid, kernel)
        if self.counter == self.iterations:
            self.closeWindow()
        self.counter += 1
        self.update()
    
    def convolve_dead_or_alive(self, grid, kernel):
        """
        Here we use convolution to find the state of the next frame in the game of life. 
        This is to make the performance better than with nested for-loops.
        """

        kernelH, kernelW = kernel.shape
        padH, padW = kernelH // 2, kernelW // 2
        gridH, gridW = grid.shape
        h, w = gridH + 1 - kernelH + 2*padH, gridW + 1 - kernelW + 2*padW

        # Add zero padding to the grid
        padded_grid = np.pad(grid, ((padH, padH), (padW, padW)), mode='constant', constant_values=0)

        filter1 = np.arange(kernelW) + np.arange(h)[:,np.newaxis]

        intemediate = padded_grid[filter1]
        intemediate = np.transpose(intemediate,(0,2,1))

        filter2 = np.arange(kernelH) + np.arange(w)[:, np.newaxis]

        intemediate = intemediate[:, filter2]
        intemediate = np.transpose(intemediate, (0,1,3,2))

        product = intemediate * kernel

        neighbor_sum = product.sum(axis = (2,3))

        conditions = [neighbor_sum == 3, np.logical_and(neighbor_sum == 4, grid == 1)]
        choices = [1, 1]

        return np.select(conditions, choices, default=0)
    
    def save_simulation_data(self, data, filename):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def closeWindow(self):
        simulation_data.iterations.extend(self.iterationList)
        self.save_simulation_data(simulation_data, './data/simulation_data.pkl')
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    grid_window = GridWindow()
    grid_window.show()
    sys.exit(app.exec_())
   