############################################################
# CMPSC 442: Informed Search
############################################################

student_name = "Michael Liam Sullivan."

############################################################
# Imports
import math
import random
import copy
from queue import PriorityQueue
from queue import deque
############################################################

# Include your imports here, if any are used.



############################################################
# Section 1: Tile Puzzle
############################################################

def create_tile_puzzle(rows, cols):
    board = [[(i * cols + j + 1) % (rows * cols) for j in range(cols)] for i in range(rows)]
    empty = (rows-1, cols-1)
    return TilePuzzle(board, rows, cols, empty)


class TilePuzzle(object):
    
    # Required
    def __init__(self, board, rows=None, cols=None, empty=None):
        self.board = board
        if rows is None or cols is None or empty is None:  # This check allows flexibility in initialization
            self.rows = len(self.board)
            self.cols = len(self.board[0])
            for i in range(self.rows):
                for j in range(self.cols):
                    if board[i][j] == 0:
                        self.empty = (i, j)
        else:
            self.empty = empty
            self.rows = rows
            self.cols = cols

    def get_board(self):
        return self.board

    def perform_move(self, direction):
        if direction == "up":
            if self.empty[0] == 0:
                return None
            swap = self.board[self.empty[0]-1][self.empty[1]]
            self.board[self.empty[0]-1][self.empty[1]] = self.board[self.empty[0]][self.empty[1]]
            self.board[self.empty[0]][self.empty[1]] = swap
            self.empty = (self.empty[0]-1, self.empty[1])
            return True
        elif direction == "down":
            if self.empty[0] == self.rows-1:
                return None
            swap = self.board[self.empty[0]+1][self.empty[1]]
            self.board[self.empty[0]+1][self.empty[1]] = self.board[self.empty[0]][self.empty[1]]
            self.board[self.empty[0]][self.empty[1]] = swap
            self.empty = (self.empty[0]+1, self.empty[1])
            return True
        elif direction == "left":
            if self.empty[1] == 0:
                return False
            swap = self.board[self.empty[0]][self.empty[1]-1]
            self.board[self.empty[0]][self.empty[1]-1] = self.board[self.empty[0]][self.empty[1]]
            self.board[self.empty[0]][self.empty[1]] = swap
            self.empty = (self.empty[0], self.empty[1]-1)
            return True
        elif direction == "right":
            if self.empty[1] == self.cols - 1:
                return False
            swap = self.board[self.empty[0]][self.empty[1]+1]
            self.board[self.empty[0]][self.empty[1]+1] = self.board[self.empty[0]][self.empty[1]]
            self.board[self.empty[0]][self.empty[1]] = swap
            self.empty = (self.empty[0], self.empty[1]+1)
            return True
        else:
            return False


    def scramble(self, num_moves):
        moves = ['up', 'down', 'left', 'right']
        for i in range(num_moves):
            move = moves[random.randint(0, 3)]
            self.perform_move(move)
            
    def is_solved(self):
        #should be a sufficient case for determining this
        count = 1
        for row in range(len(self.get_board())):
            for col in range(len(self.get_board()[0])):
                if row == self.rows - 1 and col == self.cols - 1 and self.get_board()[row][col] == 0:
                    return True
                if self.get_board()[row][col] != count:
                    return False
                count += 1
        return False
                

    def copy(self):
        temp = self.get_board()        
        return TilePuzzle(copy.deepcopy(temp))

    def successors(self):
        moves = ['up', 'down', 'left', 'right']
        successors = []
        for move in moves:
            succ = self.copy()
            if succ.perform_move(move):
                successors.append((move, succ))
        return successors


    # Required
    def find_solutions_iddfs(self):
        limit = 1
        found = False
        while not found:
            for solution in self.iddfs_helper([], limit):
                yield solution
                found = True
            limit += 1
        

    # 
    def iddfs_helper(self, moves=[], limit=0):
        #potentially problematic
        if self.is_solved():
            yield moves
            return
        if limit <= 0:
            return 
        for successor in self.successors():
            yield from successor[1].iddfs_helper(moves+[successor[0]], limit-1)

    def a_star_heuristic(self):
        config = self.get_board()
        current_map  = {}
        init_map = {}
        md_sum = 0
        init =[[(i * self.cols + j + 1) % (self.rows * self.cols) for j in range(self.cols)] for i in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                current_map[config[i][j]] = (i,j)
                init_map[init[i][j]] = (i,j)
        for value in current_map:
            #print(current_map[i], init_map[i])
            current = current_map[value]
            init = init_map[value]
            a = abs(current[0] - init[0]) + abs(current[1] - init[1])
            md_sum += a
        return md_sum


    def find_solution_a_star(self):
        open_list = PriorityQueue()
        closed_list = set()
        open_list.put((self.a_star_heuristic(), 0, 0, self.copy(), []))
        timestamp = 0
        while not open_list.empty():
            current = open_list.get()
            if current and current[3].is_solved():
                return current[4]
            
            config = tuple(map(tuple, current[3].get_board()))
            if config in closed_list:
                continue
            closed_list.add(config) #these are visited nodes
            for successor in current[3].successors():
                if tuple(map(tuple, successor[1].get_board())) in closed_list:
                    continue
                new_path = current[4] + [successor[0]]
                current_heuristic = successor[1].a_star_heuristic()
                current_dist = current[2] + 1
                choice = (current_heuristic+current_dist, timestamp, current_dist, successor[1], new_path)
                open_list.put(choice)
                timestamp += 1



############################################################
# Section 2: Grid Navigation
############################################################
def euclidian_distance(start, goal):
    return math.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)

def successors(pos, scene):
    successors = []
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for move in moves:
        x = pos[0] + move[0]
        y = pos[1] + move[1]
        if 0 <= x < len(scene) and 0 <= y < len(scene[0]) and scene[x][y] == False:
            successors.append(((x,y), move))
    return successors



def find_path(start, goal, scene):
    open_list = PriorityQueue()
    closed_list = set()
    #open_list.put((euclidian_distance(start, goal), 0, start, [start]))
    if scene[start[0]][start[1]]:
            return None
    
    open_list.put((euclidian_distance(start, goal), 0, start, [start]))
    while not open_list.empty():
        #euc, path_dist, pos, points = open_list.get()
        euc, depth, pos, points = open_list.get()
        if pos == goal:
            return points
        if pos in closed_list:
            continue
        closed_list.add(pos)
        for successor, move in successors(pos, scene):
            if successor in closed_list:
                continue
            if move[0] != 0 and move[1] != 0:
                current_depth = depth + math.sqrt(2)
            else:
                current_depth = depth + 1
            new_points = points + [successor]
            open_list.put((euclidian_distance(successor, goal) + current_depth, current_depth, successor, new_points))
            #open_list.put((euclidian_distance(successor, goal) + current_dist, current_dist, successor, new_points))
    return None


############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################


def initialize_distict_disks(length, n):
    if n > length:
        return False
    puzzle_row = [0 for _ in range(length)]
    for i in range(n):
        puzzle_row[i] = i + 1
    return puzzle_row

def perform_move(row, start, finish):
    temp = row[start]
    row[start] = row[finish]
    row[finish] = temp
    return row


def is_solved_distinct(row, length, n):
    for i in range(n):
        if row[length-1-i] != i + 1:
            return False
    for i in range(length - n):
        if row[i] != 0:
            return False

    return True


def successor_rows_distinct(row):
    length = len(row)
    successors = []
    for index in range(length):
        #move index to index + 1 if index + 1 if index+ 1 is zero or is 
        moves = ((index, index+1), (index, index+2), (index, index-1), (index, index-2))
        for i in moves:
            if i[0] >= length or i[1] >= length or i[0] < 0 or i[1] < 0:
                continue
            if row[i[0]] == 0 or row[i[1]] != 0:
                continue
            if abs(i[0] - i[1]) == 2 and row[min(i[0], i[1])+1] == 0:
                continue
            new_row = perform_move(row.copy(), i[0], i[1])
            #successors.append(((i[0], i[1]), new_row))
            yield ((i[0], i[1]), new_row)
    #return successors
    return None

def distinct_heuristic(row, length, n):
    """
    nums = row[:n]
    zeroes = [0 for i in range(length-n)]
    nums.reverse()
    solved = zeroes + nums
    solved_map = {}
    current_map = {}
    for i in range(len(row)):
        if solved[i] != 0:
            solved_map[solved[i]] = i
        if row[i] != 0:
            current_map[row[i]] = i
    ret = 0
    for i in solved_map:
        ret += abs(solved_map[i] - current_map[i])
    """
    ret = 0
    for i in range(length):
        if row[i] == 0:
            continue
        expected_pos = (length - row[i])
        ret += abs(expected_pos - i)
        print(row, row[i], expected_pos)
    return ret

def solve_distinct_disks(length, n):
    # Assuming correct initialization function that returns the initial row configuration
    row = initialize_distict_disks(length, n)
    open_list = PriorityQueue()
    closed_list = set()
    init_heuristic = distinct_heuristic(row, length, n)
    open_list.put((init_heuristic, 0, tuple(row), []))
    while not open_list.empty():
        curr_heuristic, curr_depth, curr_config, curr_path = open_list.get()
        if is_solved_distinct(curr_config, length, n):
            return curr_path
        
        if curr_config in closed_list:
            continue
        closed_list.add(curr_config)

        for move, new_config in successor_rows_distinct(list(curr_config)):
            if tuple(new_config) in closed_list:
                continue
            heuristic = distinct_heuristic(new_config, length, n)
            open_list.put((heuristic + curr_depth + 1, curr_depth + 1, tuple(new_config), curr_path + [move]))
    return None

############################################################
# Section 4: Dominoes Game
############################################################

def create_dominoes_game(rows, cols):
    board = [[False for x in range(cols)] for y in range(rows)]
    return DominoesGame(board)

class DominoesGame(object):

    # Required
    def __init__(self, board):
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])
        self.root = self  

    def get_board(self):
        return self.board

    def reset(self):
        self.board = [[False for x in range(self.cols)] for y in range(self.rows)]

    def is_legal_move(self, row, col, vertical):
        valid_space = lambda r, c: r >=0 and c>= 0 and r < self.rows and c < self.cols and not self.board[r][c]
        if valid_space(row, col):
            if vertical:
                if valid_space(row+1, col):
                    return True
            else:
                if valid_space(row, col+1):
                    return True
        return False
            

    def legal_moves(self, vertical):
        moves = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.is_legal_move(i, j, vertical):
                    moves.append((i,j))
                    #yield (i,j)
        return moves

    def perform_move(self, row, col, vertical):
        if self.is_legal_move(row, col, vertical):
            self.board[row][col] = True
            if vertical:
                self.board[row+1][col] = True
            else:
                self.board[row][col+1] = True

    def game_over(self, vertical):
        if not self.legal_moves(vertical):
            return True
        return False

    def copy(self):
        new_game = DominoesGame(copy.deepcopy(self.board))
        new_game.root = self.root
        return new_game

    def successors(self, vertical):
        successors = []
        for move in self.legal_moves(vertical):
            successor_config = self.copy()
            successor_config.perform_move(move[0], move[1], vertical)
            successors.append((move, successor_config))
        return successors

    def get_random_move(self, vertical):
        pass

    def ab_heuristic(self, vertical):
        return len(self.legal_moves(vertical)) - len(self.legal_moves(not vertical))

    # Required
    def get_best_move(self, vertical, limit):
        self.leaves = 0
        self.best_move = ()
        self.limit = limit
        self.orientation = vertical
        result = self.max_turn(vertical, 1, float("-inf"), float("inf"))
        return (self.best_move, result, self.leaves)
    
    
    def max_turn(self, vertical, depth, alpha, beta):
        if depth > self.root.limit or self.game_over(vertical):
            self.root.leaves += 1
            return self.ab_heuristic(vertical)
        
        max_evaluation = -float('inf')
        for move, successor in self.successors(vertical):
            current_evaluation = successor.min_turn(not vertical, depth+1, alpha, beta)
            if current_evaluation > max_evaluation:
                max_evaluation = current_evaluation
                self.root.best_move = move
            if max_evaluation >= beta:
                break
            alpha = max(alpha, max_evaluation)
        return max_evaluation
    



    def min_turn(self, vertical, depth, alpha, beta):
        if depth > self.root.limit or self.game_over(vertical):
            self.root.leaves += 1
            return self.ab_heuristic(self.root.orientation)
        min_evaluation = float('inf')
        for move, successor in self.successors(vertical):
            current_evaluation = successor.max_turn(not vertical, depth+1, alpha, beta)
            min_evaluation = min(current_evaluation, min_evaluation)
            if min_evaluation <= alpha:
                break
            beta = min(beta, min_evaluation)
        return min_evaluation
                




if __name__ == "__main__":

    #scene = [[False, False, False], [False, True, False], [False, False, False]]
    #print(find_path((0, 0), (2, 1), scene))
    #print(solve_distinct_disks(4, 2))
    #g = create_dominoes_game(3, 3) 
    #print(list(g.legal_moves,False)))
    b = [[False] * 3 for i in range(3)] 
    g = DominoesGame(b) 
    g.perform_move(0, 1, True)
    print(g.get_best_move(False, 1))
    """
    b = [[True, False], [True, False]] 
    g = DominoesGame(b) 
    for m, new_g in g.successors(True): 
        print(m, new_g.get_board())
    """
   
   
    

    