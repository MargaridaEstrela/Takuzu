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

        # print("col =", col)
        # print("tabuleiro")
        # print(self.board_repr)
        for i in range(n):
            column.append(self.get_number(i, col))

        column = np.array(column)
        # print(column)
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

    def get_all_free(self) -> int:
        """Devolve o numero de posicoes livres do tabuleiro, da direita para a
         esquerda, de cima para baixo."""
        n = self.size()
        free = 0
        for row in range(n):
            for col in range(n):
                if self.get_number(row, col) == 2:
                    free += 1
        return free

    def adjacent_vertical_numbers(self, row: int, col: int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""

        n = self.size()

        if row == 0:
            up = None
        else:
            up = self.get_number(row - 1, col)

        if row == n-1:
            down = None
        else:
            down = self.get_number(row + 1, col)

        return (up, down)

    def adjacent_horizontal_numbers(self, row: int, col: int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""

        n = self.size()

        if col == 0:
            left = None
        else:
            left = self.get_number(row, col - 1)

        if col == n-1:
            right = None
        else:
            right = self.get_number(row, col+1)

        return (left, right)

    def unique_row_col(self):
        """ Retorna True se e só se todas as linhas e colunas forem
         diferentes."""
        n = self.size()
        for i in range(n):
            col = self.get_col(i).copy()
            for j in range(i+1, n):
                if np.array_equal(col, self.get_col(j)):
                    return False

        for i in range(n):
            row = self.get_row(i).copy()
            for j in range(i+1, n):
                if np.array_equal(row, self.get_row(j)):
                    return False

        return True

    def verify_row_col(self):
        """ Retorna True se e só se há o valor certo de 0s e/ou 1s em cada \
            linha e coluna."""
        n = self.size()

        if n % 2 == 0:
            limit = n//2
        else:
            limit = n//2 + 1

        for i in range(n):
            row = self.get_row(i)
            col = self.get_col(i)
            # verificar se a linha tem mais 1s ou 0s que o suposto
            if np.count_nonzero(row) > limit or \
               np.count_nonzero(row) - limit > 0:
                return False

            # verificar se a coluna tem mais 1s ou 0s que o suposto
            if np.count_nonzero(col) > limit or \
               np.count_nonzero(col) - limit > 0:
                return False

        return True

    def verify_row_adjacent(self):
        """ Retorna True caso não haja mais que 2 numeros iguais adjacentes
         horizontalmente um ao outro."""
        n = self.size()

        for row in range(n):
            for col in range(n):
                pos = (self.get_number(row, col),)
                adj = pos + self.adjacent_horizontal_numbers(row, col)
                if len(unique(adj)) == 1:
                    return False
        return True

    def verify_col_adjacent(self):
        """ Retorna True caso não haja mais que 2 numeros iguais adjacentes
         verticalmente um ao outro."""
        n = self.size()

        for row in range(n):
            for col in range(n):
                pos = (self.get_number(row, col),)
                adj = pos + self.adjacent_vertical_numbers(row, col)
                if len(unique(adj)) == 1:
                    return False
        return True

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
        self.initial = TakuzuState(board, board.get_all_free())

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO

        act = []

        if state.free == 0:
            return act
        else:
            # n = state.board.size()
            # new_board = Board(n)
            # new_board.board_repr = state.board.board_repr.copy()

            position = state.board.get_first_free()
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            # print("new action")
            # print("position:", position)
            # print("-------------")
            # print(new_board.to_string())

            # row = state.board.get_row(position[0])
            # print("row:", row)

            # col = state.board.get_col(position[1])
            # print("col:", col)
            row = position[0]
            col = position[1]
            up = state.board.get_number(row-1, col)
            left = state.board.get_number(row, col-1)
            for number in range(2):
                state.board.change_number(position[0], position[1], number)
                if self.verify_adjacent(state.board, position, number) and \
                   self.verify_col_row(state.board, position, number):

                    if (up is not None and left is not None):
                        if self.verify_adjacent(state.board, (row-1, col), up) and \
                           self.verify_adjacent(state.board, (row, col-1), left):
                            act.append((position[0], position[1], number),)

                    elif (up is not None and left is None):
                        if self.verify_adjacent(state.board, (row-1, col), up):
                            act.append((position[0], position[1], number),)

                    elif (up is None and left is not None):
                        if self.verify_adjacent(state.board, (row, col-1), left):
                            act.append((position[0], position[1], number),)

                    else:
                        act.append((position[0], position[1], number),)

            # print("########")
            # print(act)
            # print("########")
            state.board.change_number(position[0], position[1], 2)
            return act

    def verify_adjacent(self, board: Board, pos, value):
        """ Retorna True caso não haja mais que 2 numeros iguais adjacentes
         horizontalmente um ao outro."""
        row, col = pos
        adj_h = (value, ) + board.adjacent_horizontal_numbers(row, col)
        adj_v = (value, ) + board.adjacent_vertical_numbers(row, col)
        if len(unique(adj_h)) == 1 or len(unique(adj_v)) == 1:
            return False
        else:
            return True

    def verify_col_row(self, board: Board, pos, value):

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

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        n = state.board.size()

        row, col, value = action
        # atualização do tabueleiro
        new_board = Board(n)
        new_board.board_repr = state.board.board_repr.copy()
        # print("antes")
        # print(new_board.board_repr)
        new_board.change_number(row, col, value)
        # print("depois")
        new_state = TakuzuState(new_board, state.free - 1)

        return new_state

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        # TODO

        return state.free == 0 and state.board.unique_row_col()

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
    # print(board.board_repr)
    goal_node = depth_first_tree_search(problem)
    print(goal_node.state.board.to_string())
