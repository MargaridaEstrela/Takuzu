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
from utils import (
    remove_all,
    unique
)
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

    def __init__(self, board, free_positions):
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

    def get_number(self, row: int, col: int) -> int:
        """Devolve o valor na respetiva posição do tabuleiro."""

        return self.board_repr[row][col]

    def get_row(self, row: int):
        """Devolve a linha correspondente do tabuleiro."""

        return self.board_repr[row]

    def get_col(self, col: int):
        """Devolve a linha correspondente do tabuleiro."""
        column = []
        n = self.size()

        for row in range(n):
            column.append(self.board_repr[row][col])

        return column

    def get_first_free(self) -> (int, int):
        """Devolve a primeira posicao livre, da direita para a esquerda, de
        cima para baixo."""
        n = self.size()

        for row in range(n):
            for col in range(n):
                if self.get_number(row, col) == 2:
                    return row, col

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

    def adjacent_vertical_numbers(self, row: int, col: int) -> (int, int):
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

        return up, down

    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
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
            right = self.get_number(row, col)

        return left, right

    def unique_row_col(self):
        """ Retorna True se e só se todas as linhas e colunas forem
         diferentes."""
        n = self.size()
        rows = []
        cols = []

        for i in range(n):
            rows.append(self.get_row(i))
            cols.append(self.get_col(i))

        total = rows + cols

        return len(np.unique(total)) == 2*n

    def verify_row_col(self):
        """ Retorna True se e só se há o valor certo de 0s e/ou 1s em cada \
            linha e coluna."""
        n = self.size()

        if n % 2 == 0:
            limit = n//2
        else:
            limit = n//2 + 1
        for i in range(n):
            # verificar se a linha tem mais 1s ou 0s que o suposto
            if self.get_row(i).count(1) > limit or self.get_row(i).count(0) > limit:
                return False

            # verificar se a coluna tem mais 1s ou 0s que o suposto
            if self.get_col(i).count(1) > limit or \
                self.get_col(i).count(0) > limit:
                return False

            return True

    def verify_row_adjacent(self):
        """ Retorna True caso não haja mais que 2 numeros iguais adjacentes
         horizontalmente um ao outro."""
        n = self.size()

        for row in range(n):
            for col in range(n):
                pos = (self.get_number(row, col))
                adj = pos + self.adjacent_horizontal_numbers(row, col)
                if len(unique(adj) == 0):
                    return False
        return True

    def verify_col_adjacent(self):
        """ Retorna True caso não haja mais que 2 numeros iguais adjacentes
         verticalmente um ao outro."""
        n = self.size()

        for row in range(n):
            for col in range(n):
                pos = (self.get_number(row, col))
                adj = pos + self.adjacent_vertical_numbers(row, col)
                if len(unique(adj) == 0):
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

        for row in range(n):
            input = sys.stdin.readline().rstrip('\n')
            num_string = remove_all('\t', input)
            for col in range(n):
                value = num_string[col]
                if value != 2:
                    board.change_number(row, col, value)

        return board

    # TODO: outros metodos da classe
    def __repr__(self):
        return self.board_repr

    def to_string(self):
        output = ""

        for row in self.board_repr:
            for col in self.board.repr:
                output += str(self.get_number(row, col))
                output += "\t"

            output += "\n"
        return output


class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        # TODO
        self.initial = TakuzuState(board, board.get_all_free)

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO

        board = state.board
        n = board.size()
        position = board.get_first_free()

        if position is None:
            return [(n, n, n)]
        else:
            return [(position[0], position[1], 0), (position[0], position[1], 1)]

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        board = state.board
        n = board.size()

        row = action[0]
        col = action[1]
        value = action[2]

        # atualização do tabueleiro
        new_board = Board(n)
        new_board.board_repr = board.board_repr
        new_board.change_number(row, col, value)

        new_state = TakuzuState(new_board, new_board.get_all_free())

        return new_state

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        # TODO

        board = state.board

        if board.get_all_free() > 0:
            return False

        else:
            return board.unique_row_col() and board.verify_row_col()\
                 and board.verify_row_adjacent() & \
                 board.verify_col_adjacent()

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
