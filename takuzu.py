# takuzu.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 28:
# 99271 Margarida Estrela
# 99323 Rui Moniz

import sys
import numpy as np
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)


class TakuzuState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de Takuzu."""
    
    def __init__(self, n):
        self.board_repr = np.zeros((n,n), dtype = int)
        
    def size(self):
        return len(self.board_repr)
        
    def change_number(self, row, col, value):
        """Altera o valor da respetiva posição do tabuleiro"""
        
        self.board_repr[row, colr] = value

    def get_number(self, row: int, col: int) -> int:
        """Devolve o valor na respetiva posição do tabuleiro."""
        
        return self.board_repr[row, col]
        
    def get_row(self, row: int):
        """Devolve a linha correspondente do tabuleiro."""
        
        return self.board_repr[row]
        
    def get_col(self, col: int):
        """Devolve a linha correspondente do tabuleiro."""
        res = ()
        n = self.size()
        
        for row in range(n):
            res += (self.board_repr[row][col], )
        
        return res

    def adjacent_vertical_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        
        return (self.get_number(row-1, col), self.get_number(row+1, col)
        
    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
      
        return (self.get_number(row, col-1), self.get_number(row, col+1)

    @staticmethod
    def parse_instance_from_stdin():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 takuzu.py < input_T01

            > from sys import stdin
            > stdin.readline()
        """
        
        n = int(sys.stdin.readline())
        board = Board(n)
        
        for line, row in zip(sys.stdin.readlines(), n):
            board.board_repr[row] = list(map(int, line.rstrip("\n").split("\t"))
                for col in range(n):
                    value = get_number(board, row, col)
                    if value != 0:
                        board.change_number(board, row, col, value)
                        
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
        self.initial = TakuzuState(board)

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO
        
        actions = []
        board = state.board
        n = board.size()
        
        for row in range(n):
            if board.board_repr.count(0) == n%2 and board.board_repr.count(1) == n%2 or \
                        board.board_repr.count(0) == n%2 + 1 and board.board_repr.count(1) == n%2 + 1:
                continue
            elif board.board_repr.count(1) == 1:
                if n%2 == 0 and board.board_repr.count(0) == n%2:
                    
                    
        
        
        pass

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        
        board = state.board
        row, col, value = action
        
        #atualização do tabueleiro
        new_board = Board(board.size())
        new_board.board_repr = board.board_repr
        new_board.change_number(row, col, value)
        
        new_state = TakuzuState(new_board, 0)
        
        return new_board.

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        # TODO
        
        board = state.board
        n = board.size()
            
        return unique_row_col(board, n) and verify_row_col(board, n) and verify_row_adjacent(board, n) and \
                verify_row_adjacent(board, n)

        
    def unique_row_col(self, board: Board, n: int):
        """ Retorna True se e só se todas as linhas e colunas forem diferentes."""
        
        return len(unique((board.get_row(0), board.get_row(1), board.get_col(2)))) == n and \
                len(unique((board.get_col(0), board.get_col(1), board.get_col(2)))) == n
                
                
    def verify_row_col(self, board: Board, n: int):
        """ Retorna True se e só se há o valor certo de 0s e/ou 1s em cada linha e coluna."""
        
        for i in range(n):
        
            #para valores de n par
            if n%2 == 0:
                #verificar se a linha tem mais 1s ou 0s que o suposto
                if board.get_row(i).count(1) > n/2 or board.get_row(i).count(0) > n/2:
                    return False
                    
                #verificar se a coluna tem mais 1s ou 0s que o suposto
                if board.get_col(i).count(1) > n/2 or board.get_col(i).count(0) > n/2:
                    return False
                    
            #para valores de n impar
            elif n%2 != 0:
                #verificar se a linha tem mais 1s ou 0s que o suposto
                if board.get_row(i).count(1) > n/2 + 1 or board.get_row(i).count(0) > n/2 + 1:
                    return False
                    
                #verificar se a coluna tem mais 1s ou 0s que o suposto
                if board.get_col(i).count(1) > n/2 + 1 or board.get_col(i).count(0) > n/2 + 1:
                    return False
                    
            return True
            
    def verify_row_adjacent(self, board: Board, n):
        """ Retorna True caso não haja mais que 2 numeros iguais adjacentes horizontalmente
        um ao outro."""
        
        for row in range(n):
            value = board.get_number(row,0)
            count = 1
            for col in range(n):
                if row == 0 and col == 0:
                    continue
                elif board.get_number(row, col) == value:
                    count += 1
                elif cont > 2:
                    return False
                else:
                    value = board.get_number(row, col)
        
        return True
        
    def verify_col_adjacent(self, board: Board, n):
        """ Retorna True caso não haja mais que 2 numeros iguais adjacentes verticalmente
        um ao outro."""
        
        for col in range(n):
            value = board.get_number(row,0)
            count = 1
            for row in range(n):
                if col == 0 and row == 0:
                    continue
                elif board.get_number(row, col) == value:
                    count += 1
                elif cont > 2:
                    return False
                else:
                    value = board.get_number(row, col)
                    count = 0
        
        return True
            
            
        
        
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
    
    board = Board.parse_instance_from_stdin(_)
    problem = Takuzu(board)
    goal_node = depth_first_tree_search(problem)
    print(goal_node.state.board.to_string())
