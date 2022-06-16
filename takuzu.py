# takuzu.py: Template para implementação do projeto de Inteligência Artificial
# 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as
# instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que
# considerem pertinentes.

# Grupo 28:
# 99271 Margarida Estrela
# 99323 Rui Moniz

import sys
import numpy as np
from utils import unique
from search import (
    Problem,
    Node,
    # astar_search,
    # breadth_first_tree_search,
    depth_first_tree_search,
    # greedy_search,
    # recursive_best_first_search,
)


class TakuzuState:
    state_id = 0

    def __init__(self, board, free_positions: int):
        self.board = board
        self.id = TakuzuState.state_id
        self.free = free_positions
        TakuzuState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de Takuzu."""

    def __init__(self, n):
        self.board_repr = np.full((n, n), 2, dtype=int)

    def size(self):
        return len(self.board_repr)

    def change_number(self, row, col, value):
        """Altera o valor da respetiva posição do tabuleiro"""
        self.board_repr[row][col] = value

    def get_number(self, row: int, col: int):
        """Devolve o valor na respetiva posição do tabuleiro."""
        n = self.size()
        if(row < 0 or row > n-1 or col < 0 or col > n-1):
            return None
        else:
            return self.board_repr[row][col]

    def get_row(self, row: int):
        """Devolve a linha correspondente do tabuleiro."""
        return self.board_repr[row].copy()

    def get_col(self, col: int):
        """Devolve a coluna correspondente do tabuleiro."""
        column = []
        n = self.size()

        for i in range(n):
            column.append(self.get_number(i, col))

        column = np.array(column)

        return column

    def get_first_free(self):
        """Devolve a primeira posicao livre, da direita para a esquerda, de
        cima para baixo."""
        n = self.size()

        for row in range(n):
            for col in range(n):
                if self.get_number(row, col) == 2:
                    return row, col
        return None, None

    def get_all_free(self):
        """Devolve o numero de posicoes livres do tabuleiro, da direita para a
         esquerda, de cima para baixo."""
        n = self.size()
        free = []
        for row in range(n):
            for col in range(n):
                if self.get_number(row, col) == 2:
                    free += [(row, col), ]
        return free

    def adjacent_vertical_numbers(self, row: int, col: int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""

        n = self.size()

        if row == 0:
            return (None, self.get_number(row + 1, col))
        elif row == n-1:
            return (self.get_number(row - 1, col), None)
        else:
            return (self.get_number(row - 1, col), self.get_number(row + 1, col))

    def adjacent_horizontal_numbers(self, row: int, col: int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""

        n = self.size()

        if col == 0:
            return (None, self.get_number(row, col + 1))
        elif col == n-1:
            return (self.get_number(row, col - 1), None)
        else:
            return (self.get_number(row, col - 1), self.get_number(row, col + 1))

    @staticmethod
    def parse_instance_from_stdin():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.
        Por exemplo:
            $ python3 takuzu.py < input_T01
            > from sys import stdin
            > stdin.readline()
        """

        n = int(sys.stdin.readline().rstrip('\n'))
        board = Board(n)
        i = 0

        input = sys.stdin.readlines()
        for line, i in zip(input, range(n)):
            board.board_repr[i] = (list(map(int, line.rstrip("\n").split("\t"))))

        return board

    # TODO: outros metodos da classe
    def __repr__(self):
        return self.board_repr

    def to_string(self):
        output = ""

        for row in self.board_repr:
            output += "\t".join(map(str, row))
            output += "\n"

        return output.rstrip()


class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        # TODO
        self.initial = TakuzuState(board, len(board.get_all_free()))

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO

        act = []

        if state.free == 0:
            return act

        else:
            position = state.board.get_first_free()

            row = position[0]
            col = position[1]

            for number in range(2):
                state.board.change_number(row, col, number)
                if self.verify_adjacent(state.board, position, number) and self.verify_col_row(state.board, position, number):
                    act.append((row, col, number),)

            state.board.change_number(row, col, 2)
            return act

    def verify_adjacent_horizontal(self, board: Board, pos, value):
        """ Retorna True caso não haja mais que 2 numeros iguais adjacentes
         horizontalmente um ao outro."""

        row, col = pos
        n = board.size()

        if (row == 0 and col in [0, n-1]) or (row == n-1 and col in [0, n-1]):
            return True

        adj_h = (value, ) + board.adjacent_horizontal_numbers(row, col)

        return len(unique(adj_h)) != 1

    def verify_adjacent_vertical(self, board: Board, pos, value):
        """ Retorna True caso não haja mais que 2 numeros iguais adjacentes
         horizontalmente um ao outro."""

        row, col = pos
        n = board.size()

        if (row == 0 and col in [0, n-1]) or (row == n-1 and col in [0, n-1]):
            return True

        adj_v = (value, ) + board.adjacent_vertical_numbers(row, col)

        return len(unique(adj_v)) != 1

    def verify_adjacent(self, board: Board, pos, value):
        """ Retorna True caso não haja mais que 2 numeros iguais adjacentes
         horizontalmente um ao outro."""

        row, col = pos

        up = board.get_number(row-1, col)
        down = board.get_number(row+1, col)
        left = board.get_number(row, col-1)
        right = board.get_number(row, col+1)

        if up is None:
            if left is None:
                return self.verify_adjacent_horizontal(board, (row, col+1), right) and \
                    self.verify_adjacent_vertical(board, (row+1, col), down)
            elif right is None:
                return self.verify_adjacent_horizontal(board, (row, col-1), left) and \
                    self.verify_adjacent_vertical(board, (row+1, col), down)
            else:
                return self.verify_adjacent_horizontal(board, (row, col-1), left) and \
                    self.verify_adjacent_horizontal(board, (row, col+1), right) and \
                    self.verify_adjacent_horizontal(board, pos, value) and \
                    self.verify_adjacent_vertical(board, (row+1, col), down)

        elif down is None:
            if left is None:
                return self.verify_adjacent_horizontal(board, (row, col+1), right) and \
                    self.verify_adjacent_vertical(board, (row-1, col), up)
            elif right is None:
                return self.verify_adjacent_horizontal(board, (row, col-1), left) and \
                    self.verify_adjacent_vertical(board, (row-1, col), up)
            else:
                return self.verify_adjacent_horizontal(board, (row, col-1), left) and \
                    self.verify_adjacent_horizontal(board, (row, col+1), right) and \
                    self.verify_adjacent_horizontal(board, pos, value) and \
                    self.verify_adjacent_vertical(board, (row-1, col), up)

        elif left is None:
            return self.verify_adjacent_horizontal(board, (row, col+1), right) and \
                self.verify_adjacent_vertical(board, pos, value) and \
                self.verify_adjacent_vertical(board, (row-1, col), up) and \
                self.verify_adjacent_vertical(board, (row+1, col), down)

        elif right is None:
            return self.verify_adjacent_horizontal(board, (row, col-1), left) and \
                self.verify_adjacent_vertical(board, pos, value) and \
                self.verify_adjacent_vertical(board, (row-1, col), up) and \
                self.verify_adjacent_vertical(board, (row+1, col), down)

        else:
            return self.verify_adjacent_horizontal(board, (row, col-1), left) and \
                self.verify_adjacent_horizontal(board, (row, col+1), right) and \
                self.verify_adjacent_horizontal(board, pos, value) and \
                self.verify_adjacent_vertical(board, (row-1, col), up) and \
                self.verify_adjacent_vertical(board, (row+1, col), down) and \
                self.verify_adjacent_vertical(board, pos, value)

    def verify_col_row(self, board: Board, pos, value):
        """ Retorna True se e só se o número de 0s e/ou 1s em cada \
        linha e coluna não excede o limite."""

        n = board.size()
        if n % 2 == 0:
            limit = n//2
        else:
            limit = n//2 + 1

        row = board.get_row(pos[0])
        col = board.get_col(pos[1])

        value_row = np.count_nonzero(row == value)
        value_col = np.count_nonzero(col == value)

        # print("row =", value_row, "col =", value_col)

        return value_row <= limit and value_col <= limit

    def unique_row_col(self, board: Board):
        """ Retorna True se e só se todas as linhas e colunas forem
         diferentes."""
        n = board.size()
        col = []
        row = []
        for i in range(n):
            col += [board.get_col(i), ]
            row += [board.get_row(i), ]

        return self.find_duplicates(col) and self.find_duplicates(row)


    def find_duplicates(self, array):
        """ Retorna True se e só se todos os elementos da lista forem únicos."""

        for i in range(len(array)):
            for j in range(i+1, len(array)):
                if np.array_equal(array[i], array[j]):
                    return False

        return True


    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        n = state.board.size()

        row, col, value = action

        new_board = Board(n)
        new_board.board_repr = state.board.board_repr.copy()
        new_board.change_number(row, col, value)

        new_state = TakuzuState(new_board, state.free - 1)

        return new_state

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        # TODO

        return state.free == 0 and self.unique_row_col(state.board)

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.

    board = Board.parse_instance_from_stdin()
    problem = Takuzu(board)
    goal_node = depth_first_tree_search(problem)
    print(goal_node.state.board.to_string())
    