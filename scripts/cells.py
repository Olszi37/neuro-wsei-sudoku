import numpy as np
import cv2
import pickle
import os
from sklearn.externals import joblib
from skimage.feature import hog

from helpers import Helpers
from digit import Digit
from scripts.train import NeuralNetwork


class Cells(object):
    '''
    Extracts each cell from the sudoku grid obtained
    from the Extractor
    '''

    def __init__(self, sudoku):
        print "Loading recognition libraries...",
        clf, pp = joblib.load(os.path.abspath("scripts\\digits_cls.pkl"))

        sizes, biases, wts = pickle.load(open('networks\\net', 'r'))
        net = NeuralNetwork(customValues=(sizes, biases, wts))
        print 'done.'

        print 'Extracting cells...',
        self.helpers = Helpers()
        self.cells = self.extractCells(sudoku, clf, pp, net)
        print 'done.'

    def extractCells(self, sudoku, clf, pp, net):
        cells = []
        W, H = sudoku.shape
        cell_size = W / 9
        i, j = 0, 0
        for r in range(0, W, cell_size):
            row = []
            j = 0
            for c in range(0, W, cell_size):
                cell = sudoku[r:r + cell_size, c:c + cell_size]

                # powiekszenie komorki (28x28) dla sieci neuronowej do poznania
                cell = self.helpers.make_it_square(cell, 28)

                cell_1 = self.clean_1(cell)  # wyczyszczenie szumow
                print cell_1.mean()
# kolejny sposob na rozpoznanie pustych wierszy - obliczac jasnosc/ciemnosc poszczegolnych komorek
                if cell_1.mean() > 40.0:
                    cell = cv2.dilate(cell, (3, 3))
                    cell = self.clean_2(cell)
                    self.helpers.show(cell, 'After clean_2')
                    cell_hog = hog(cell, orientations=9, pixels_per_cell=(14, 14),
                                   cells_per_block=(1, 1), visualise=False)
                    cell_hog = pp.transform(np.array([cell_hog], 'float64'))
                    digit = clf.predict(cell_hog)  # rozpoznanie - 2 algorytm
                    row.append(digit)
                else:
                    row.append('.')  # pusta komorka
                j += 1
            cells.append(row)
            i += 1
        return cells

    def clean_1(self, cell):
        contour = self.helpers.largestContour(cell.copy())
        x, y, w, h = cv2.boundingRect(contour)
        cell = self.helpers.make_it_square(cell[y:y + h, x:x + w], 28)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, kernel)
        cell = 255 * (cell / 130)
        return cell

    def clean_2(self, cell):
        contour = self.helpers.largestContour(cell.copy())
        x, y, w, h = cv2.boundingRect(contour)
        cell = self.helpers.make_it_square(cell[y:y + h, x:x + w], 28)
        self.helpers.show(cell, "after cropping")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        # cell = cv2.dilate(cell, kernel, iterations=1)
        cell = cv2.morphologyEx(cell, cv2.MORPH_OPEN, kernel)
        # cell = 255 * (cell / 130)
        return cell

    def centerDigit(self, digit):
        digit = self.centerX(digit)
        digit = self.centerY(digit)
        return digit

    def centerX(self, digit):
        topLine = self.helpers.getTopLine(digit)
        bottomLine = self.helpers.getBottomLine(digit)
        if topLine is None or bottomLine is None:
            return digit
        centerLine = (topLine + bottomLine) >> 1
        imageCenter = digit.shape[0] >> 1
        digit = self.helpers.rowShift(
            digit, start=topLine, end=bottomLine, length=imageCenter - centerLine)
        return digit

    def centerY(self, digit):
        leftLine = self.helpers.getLeftLine(digit)
        rightLine = self.helpers.getRightLine(digit)
        if leftLine is None or rightLine is None:
            return digit
        centerLine = (leftLine + rightLine) >> 1
        imageCenter = digit.shape[1] >> 1
        digit = self.helpers.colShift(
            digit, start=leftLine, end=rightLine, length=imageCenter - centerLine)
        return digit
