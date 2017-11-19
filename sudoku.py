import os
import sys

from scripts.sudokuExtractor import Extractor
from scripts.sudoku_str import SudokuStr


def get_cells(image_path):
    # manipulacja obrazem, wyciagniecie komorek sudoku i rozpoznanie cyfr
    for row in Extractor(os.path.abspath(image_path)).cells:
        for cell in row:
            yield str(cell).replace('[', '').replace(']', '')


def snap_sudoku(image_path):
    # pobranie wyniku rozpoznania cyfr z komorek do obliczenia sudoku
    grid = ''.join(cell for cell in get_cells(image_path))
    s = SudokuStr(grid)
    print "Finded values: \n", s
    try:
        print('\nSolving...\n\n{}'.format(s.solve()))
    except ValueError:
        print('No solution found.  Please rescan the puzzle.')


if __name__ == '__main__':
    try:
        snap_sudoku(image_path=sys.argv[1])
    except IndexError:
        fmt = 'usage: {} image_path'
        print(fmt.format(__file__.split('/')[-1]))
