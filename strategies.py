from __future__ import annotations
import chess
import chess.engine
import chess.polyglot
from ast import Tuple
from chess.engine import PlayResult
from engine_wrapper import MinimalEngine
from typing import Any, Union
import logging
import time
import math

MOVE = Union[chess.engine.PlayResult, list[chess.Move]]

logger = logging.getLogger(__name__)

# Initial piece value
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 305,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 2000
}

# Checkmate value that is sure to be the highes value 
CHECKMATE_VALUE = 10 * sum(PIECE_VALUES.values())

# How many pieces are in the board a the start position
INITIAL_PIECE_COUNT = {
    chess.PAWN: 16,
    chess.KNIGHT: 4,
    chess.BISHOP: 4,
    chess.ROOK: 4,
    chess.QUEEN: 2,
    chess.KING: 0  # not counting the kings
}

# How much each piece can "move" on average
PIECE_MOBILITY_WEIGHT = {
    chess.PAWN: 1,
    chess.KNIGHT: 8,
    chess.BISHOP: 13,
    chess.ROOK: 14,
    chess.QUEEN: 20
}

# Squares of the board orderd from center to outside (in a spiral)
CENTER_SORTED_SQUARES_DICT = {
    chess.D4: 0, chess.E4: 1, chess.D5: 2, chess.E5: 3,
    chess.C4: 4, chess.F4: 5, chess.C5: 6, chess.F5: 7,
    chess.D3: 8, chess.E3: 9, chess.D6: 10, chess.E6: 11,
    chess.C3: 12, chess.F3: 13, chess.C6: 14, chess.F6: 15,
    chess.B3: 16, chess.G3: 17, chess.B6: 18, chess.G6: 19,
    chess.B4: 20, chess.G4: 21, chess.B5: 22, chess.G5: 23,
    chess.D2: 24, chess.E2: 25, chess.D7: 26, chess.E7: 27,
    chess.C2: 28, chess.F2: 29, chess.C7: 30, chess.F7: 31,
    chess.B2: 32, chess.G2: 33, chess.B7: 34, chess.G7: 35,
    chess.A2: 36, chess.H2: 37, chess.A7: 38, chess.H7: 39,
    chess.A3: 40, chess.H3: 41, chess.A6: 42, chess.H6: 43,
    chess.A4: 44, chess.H4: 45, chess.A5: 46, chess.H5: 47,
    chess.D1: 48, chess.E1: 49, chess.D8: 50, chess.E8: 51,
    chess.C1: 52, chess.F1: 53, chess.C8: 54, chess.F8: 55,
    chess.B1: 56, chess.G1: 57, chess.B8: 58, chess.G8: 59,
    chess.A1: 60, chess.H1: 61, chess.A8: 62, chess.H8: 63
}

# Iteration 2: bishop and knight development
KNIGHTS_BISHOPS_STARTING_SQUARES = {
    chess.WHITE: [chess.B1, chess.G1, chess.C1, chess.F1],
    chess.BLACK: [chess.B8, chess.G8, chess.C8, chess.F8]
}

# Iteration 3: development penalty
DEVELOPMENT_PENALTY = 25

# Iteration 4: pins
PIN_PENALTY = 50

# Iteration 5: double pawn
DOUBLE_PAWN_PENALTY = 25

# Iteration 6: moving a piece twice
SAME_PIECE_MOVE_PENALTY = 30

# Iteration 7: open files for rooks and queens
OPEN_FILE_BONUS = 25

# Iteration 8: castle
CASTLING_RIGHT_BONUS = 10
CASTLING_BONUS = 50

# Iteration 9: Bonus for rook on 7th rank
ROOK_7TH_RANK_BONUS = 25

# Iteration 10: let the queen stay home!
QUEEN_STARTING_SQUARES = {chess.WHITE: chess.D1, chess.BLACK: chess.D8}
EARLY_QUEEN_MOVE_PENALTY = 50

# Necessary
class ExampleEngine(MinimalEngine):
    pass

# The class entry for the transposition tabe
class HashEntry:
    def __init__(self):
        self.key = 0
        self.depth = -1
        self.value = None
        self.flag = None
        self.best_move = None

# The transposition table class
class TranspositionTable:
    # Flag values
    EXACT = 0
    ALPHA = 1
    BETA = 2

    def __init__(self, size=30000, strategy="depth"):
        self.size = size
        self.table = [HashEntry() for _ in range(self.size)]
        self.strategy = strategy

    def __str__(self, board: chess.Board):
        index = self.get_index(board)  
        entry = self.table[index] 
        if entry:
            return f"Key: {entry.key}, Depth: {entry.depth}, Flag: {entry.flag}, Value: {entry.value}, Best Move: {entry.best_move}\n{board}"
        else:
            return f"No entry found in the tt for the board:\n{board}"

    # Returns the index where the entry should be put(from a board)
    def get_index(self, board):
        key = chess.polyglot.zobrist_hash(board)
        return key % self.size

    # Saves the entry at the tt
    def record(self, key, depth, value, flag, best_move):
        index = key % self.size
        entry = self.table[index]

        # Depending on the tt strategy, it replaces always or by depth,
        # if the depth is smaller or equal than the old entry
        replace = False
        if self.strategy == "always":
            replace = True
        elif self.strategy == "depth":
            replace = entry.depth <= depth

        # If the new entry has a best move and it should replace, replace
        if replace and best_move:
            entry.key = key
            entry.depth = depth
            entry.value = value
            entry.flag = flag
            entry.best_move = best_move

    # Returns the entry for the board, or None
    def get_entry(self, board: chess.Board):
        key = chess.polyglot.zobrist_hash(board)
        index = key % self.size
        entry = self.table[index]
        if entry and entry.key == key:
            return entry
        return None
    
    #  Will send the move of the entry from the board sent, or None
    def get_move(self, board: chess.Board):
        entry = self.get_entry(board)
        if entry:
            return entry.best_move
        return None

# This class takes care of organising all related to the different tts
class TTManager:
    def __init__(self, size=30000):
        # Two tts which are the main tts
        self.primary_tt = TranspositionTable(size, strategy="depth")
        self.secondary_tt = TranspositionTable(size, strategy="always")

        # The working tt will be used while the search is ongoing, in case a timeout, we don't
        # want to pollute the main tts with bad data
        self.working_tt = TranspositionTable(size, strategy="depth")

    # Given a board, return the entry from that board
    # it can be chosen to skip the working tt
    def get_entry(self, board: chess.Board, skip_working_tt=False):
        # Retrieve the entry frome each tt
        working_entry = None if skip_working_tt else self.working_tt.get_entry(board)
        primary_entry = self.primary_tt.get_entry(board)
        secondary_entry = self.secondary_tt.get_entry(board)

        # Filter out None entries
        valid_entries = [e for e in [working_entry, primary_entry, secondary_entry] if e]

        # If no valid entries are found, return None
        if not valid_entries:
            return None

        # Sort entries based on depth and also entry type, prioritising exact flag
        valid_entries.sort(key=lambda e: (e.depth, e.flag == TranspositionTable.EXACT), reverse=True)

        # Return the best entry based on the sorting
        return valid_entries[0]
    
    # Returns the value of the position if exists. based on the best entry
    def get_value(self, board, depth, alpha, beta):
        best_entry = self.get_entry(board)
        
        # Check if the best entry exists and its depth is greater than or equal to the current depth
        if not best_entry or best_entry.depth < depth:
            return None

        # Return the value based on the best entry's flag, and alpha beta bounds
        if best_entry.flag == TranspositionTable.EXACT:
            return best_entry.value
        elif best_entry.flag == TranspositionTable.ALPHA and best_entry.value <= alpha:
            return alpha
        elif best_entry.flag == TranspositionTable.BETA and best_entry.value >= beta:
            return beta
        return None
    
    # Will return the move saved with the board's entry
    def get_best_move(self, board: chess.Board):
        entry = self.get_entry(board, skip_working_tt=True)
        if entry and entry.best_move:
            return entry.best_move
        return None
    
    # Records the entry into the working tt
    def record(self, board, depth, value, flag, best_move):
        key = chess.polyglot.zobrist_hash(board)
        # Otherwise, record to the working TT
        self.working_tt.record(key, depth, value, flag, best_move)

    # This is called if search is completed, to tranfer from working tt to main tts
    def transfer_to_main(self):
        # Transfers all entries from working_tt to primary and secondary TTs based on their strategies
        for entry in self.working_tt.table:
            if entry.key:
                self.primary_tt.record(entry.key, entry.depth, entry.value, entry.flag, entry.best_move)
                self.secondary_tt.record(entry.key, entry.depth, entry.value, entry.flag, entry.best_move)

# This class takes care of the Principal Variation, creating it and formating it to be printed
class PV:
    def __init__(self, tt):
        self.tt = tt

    # This just formats the pv to be printed
    def format_pv(self, board: chess.Board, game_phase):
        pv = self.get_pv(board)

        temp_board = board.copy()
        evaluations = []

        for move in pv:
            temp_board.push(move)
            score = evaluate(temp_board, game_phase)
            evaluations.append((move, score))

        return " -> ".join([f"{move.uci()} ({score:.2f})" for move, score in evaluations])

    # This takes care of retrieving the principal variation from the boards entries and best moves from each entry
    # it has a 
    def get_pv(self, board: chess.Board):
        pv = []
        current_board = board.copy()
        depth = 0
        MAX_DEPTH = 10
        while depth < MAX_DEPTH:
            entry = self.tt.get_entry(current_board)
            if entry and entry.best_move:
                pv.append(entry.best_move)
                current_board.push(entry.best_move)
            else:
                break
            depth += 1
        return pv

class ThesisEngine(ExampleEngine):

    # Depths by default if set_depth is not called
    depth = 5
    quies_depth = 5

    # Game phase variables
    game_phase = "early"
    early_game_moves = 10
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tt_manager = TTManager()
        self.history_table = [[0.0 for _ in range(64)] for _ in range(64)]
        self.killer_moves = [[None, None] for _ in range(20)] # 20 Represents a ply depth we won't be searching past

        # For depth and time control
        self.search_interrupted = False
        self.last_successful_depth = 0
        
        self.depths = {
                "early": 3,
                "mid": 3,
                "mid-end": 4,
                "end": 5
            }
        
        self.quies_depths = {
                "early": 3,
                "mid": 3.0,
                "mid-end": 4.0,
                "end": 5.0
            }
        
        self.min_depths = {
                "mid": 3,
                "mid-end": 4,
                "end": 5
            }
        
        self.min_quies_depths = {
            "mid": 2.0,
            "mid-end": 3.0,
            "end": 4.0
        }
        
        self.max_depths = {
            "mid": 4,
            "mid-end": 6,
            "end": 10
            }
        
        self.max_quies_depths = {
            "mid": 5,
            "mid-end": 6,
            "end": 7
            }
        
    # Main search function being called 
    def search(self, board: chess.Board, limit: chess.engine.Limit, *args: Any) -> chess.engine.PlayResult:
        # Preparing some stuff
        self.preparation(board, limit)

        #-------------------------#
        #   Iterative deepening   #
        #-------------------------#

        self.search_interrupted = False  # Flag to check if search was interrupted due to timeout

        for depth in range(1, self.depth + 1):
            if self.search_interrupted:
                break

            logger.info(f"Searching at depth {depth}")

            self.reset_before_search()
            
            try:
                
                alpha = -50000 # Values bigger than checkmate value
                beta = 50000

                # The magic happening here
                value = self.alpha_beta(board, depth, alpha, beta)
                    
            except TimeoutError as e:
                self.log_timeout(depth, e)
                self.search_interrupted = True

            # Depth was finished, so we can transfer working_tt to normal_tt
            if not self.search_interrupted:
                self.tt_manager.transfer_to_main()
                self.log_after_each_iteration(depth, value)

                # Set fully searched depth variable
                self.last_successful_depth = depth
            
        #------------#
        #   Ending   #
        #------------#
        self.log_final_results()
        move = self.get_move(self.o_board)

        return PlayResult(move, None)
    
    def alpha_beta(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        # Early termination for time outs
        if time.time() - self.start_time > self.time_to_move:
            raise TimeoutError("Timout inside alpha_beta")
        
        # Checking if the position is in the tt
        tt_value = self.tt_manager.get_value(board, depth, alpha, beta)
        if tt_value is not None:
           return tt_value

        moves = self.get_ordered_moves(board, False, depth)

        if len(moves) == 0:
            if board.is_checkmate():
                return -CHECKMATE_VALUE + board.ply()
            else:
                return 0.0 # Stalemate

        # When depth reached the limit, call quies
        if depth == 0:
            return self.quies(board, self.quies_depth, alpha, beta)

        # Setting up
        hash_flag = TranspositionTable.ALPHA
        best_move = None
        for move in moves:
            # PUUUSH!
            board.push(move)
            
            # Going a depth deeper, using the negamax approach
            val = -self.alpha_beta(board, depth -1, -beta, -alpha)

            # POOOP
            board.pop()

            if val >= beta:
                if not board.is_capture(move) and move not in self.killer_moves[depth]:
                    # Shift the existing killer move and store the new one
                    self.killer_moves[depth][1] = self.killer_moves[depth][0]
                    self.killer_moves[depth][0] = move
                self.history_table[move.from_square][move.to_square] += depth * depth
                self.tt_manager.record(board, depth, beta, TranspositionTable.BETA, move)
                return beta
            
            if val > alpha:
                hash_flag = TranspositionTable.EXACT
                alpha = val
                best_move = move

        self.tt_manager.record(board, depth, alpha, hash_flag, best_move)
        return alpha
    
    def quies(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:        
        # Early termination for time outs
        if time.time() - self.start_time > self.time_to_move:
            raise TimeoutError("Timout inside quies")
        
        val = evaluate(board, self.game_phase)

        if board.is_check():
            moves = self.get_escaping_moves(board)
        else:
            moves = self.get_ordered_moves(board, True, depth)

        if depth == 0 or len(moves) == 0:
            return val
        
        if not board.is_check():
            if val >= beta:
                return beta
        
            if val > alpha:
                alpha = val
                    
        for move in moves:
            # PUUUSH!
            board.push(move)

            # Going one depth deeper
            val = -self.quies(board, depth-1, -beta, -alpha)

            # POOOP
            board.pop()

            if val >= beta:
                return beta
            
            if val > alpha:
                alpha = val
        
        return alpha
    
    #-------------------#
    #   Move ordering   #
    #-------------------#
    
    # Used by quies to print moves after the player has been checked (moves to escape the check)
    def get_escaping_moves(self, board: chess.Board) -> list:
        moves = list(board.legal_moves)
        moves.sort(key=self.center_score, reverse=True) # Since its checks probably going away from the middle is better
        return moves

    # Can be called with boolean quiescence, to get only checks and captures moves
    def get_ordered_moves(self, board: chess.Board, quiescence: bool, depth: int) -> list:
        if quiescence:
            moves = [move for move in board.legal_moves if board.is_capture(move) or board.gives_check(move)]
        else:
            moves = list(board.legal_moves)

        pv_move = self.tt_manager.get_best_move(board)

        # 1. Sort by center score
        moves.sort(key=self.center_score, reverse=False)

        # 2. Sort by checks
        moves.sort(key=lambda move: not board.gives_check(move))
        
        # 3. Sort by MVV-LVA
        moves.sort(key=lambda move: self.mvv_lva_score(board, move), reverse=True)

        # 4. Sort by history heuristic
        moves.sort(key=lambda move: self.history_table[move.from_square][move.to_square], reverse=True)

        # 5. Add killer moves to the front
        for killer in reversed(self.killer_moves[depth]):
            if killer in moves:
                moves.remove(killer)
                moves.insert(0, killer)

        # 6. Add pv_moves
        if pv_move in moves:
            moves.remove(pv_move)
            moves.insert(0, pv_move)

        return moves

    def center_score(self, move: chess.Move) -> int:
        return CENTER_SORTED_SQUARES_DICT[move.to_square]
    
    def mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int:
        # Non-capture moves have no MVV-LVA value
        if not board.is_capture(move):
            return 0
        
        # Get the attacker's value
        attacker_value = PIECE_VALUES[board.piece_at(move.from_square).piece_type]
        
        # Handle en passant capture
        if board.is_en_passant(move):
            return PIECE_VALUES[chess.PAWN] - attacker_value

        # Handle regular captures and promotions
        victim = board.piece_at(move.to_square)
        if victim is not None:
            victim_value = PIECE_VALUES[victim.piece_type]
            return victim_value - attacker_value

        # Handle pawn promotions where there isn't a direct victim
        if move.promotion:
            promotion_value = PIECE_VALUES[move.promotion]
            return promotion_value - attacker_value

        # If we reach here, something unexpected has happened...
        logging.info("Unexpected scenario in MVV_LVA...")
        return 0

    #---------------#
    #   "Setters"   #
    #---------------#

    def set_depth(self):
        # For simplicity
        phase = self.game_phase

        # Early game, no adjustment
        if phase == "early":
            self.depth = self.depths[phase]
            self.quies_depth = self.quies_depths[phase]
            return
            
        depth_difference = self.depth - self.last_successful_depth
        # If the last search was interrupted
        if self.search_interrupted:

            # If completed depth is above (or equal) to minimum depth, use it as next depth ie depth was 4, searched to 3, min is 3: use 3, and quies -= difference
            # 3 >= 3 (for mid), so yes: stay at 3, quies -diff
            if self.last_successful_depth >= self.min_depths[phase]:
                self.depths[phase] = self.last_successful_depth
                self.quies_depths[phase] -= depth_difference
                self.quies_depths[phase] = math.floor(self.quies_depths[phase]) # So that 4.75 - 1 won't be 3.75, but 3.0
            else: # If last_depth is smaller than minimum (for example: min = 3 and last succ. was 2)
                # If quies is min quies already, then reduce min depth by 1
                if math.floor(self.quies_depths[phase]) > self.min_quies_depths[phase]:
                    self.quies_depths[phase] = self.min_depths[phase]
                else:
                    self.min_depths[phase] -= 1  # Decrease the minimum depth for the game phase
                    self.depths[phase] = self.min_depths[phase]
        else:
            # If quies is still smaller than max quies
            if self.quies_depths[phase] < self.max_quies_depths[phase]:
                self.quies_depths[phase] += 0.25
            else: # Quies reached maximum quies
                if self.depths[phase] < self.max_depths[phase]:
                    # We have not reached max depth, so depth  increases by 1 and quies = min quies
                    self.depths[phase] += 1
                    self.quies_depths[phase] = self.min_quies_depths[phase]

        # Set the depth, with a minimum of 2
        self.depth = max(self.depths[phase], 2)

        # Quies should not go under min quies
        self.quies_depths[phase] = max(self.min_quies_depths[phase], self.quies_depths[phase])
        # Set quies depth, rounding down so that quies +1 takes 4 steps
        self.quies_depth = math.floor(self.quies_depths[phase])

    def set_time_to_move(self):
        proposed_time = 1

        # Calculate the proposed time for the move based on the game phase
        if self.game_phase == "early":
            proposed_time = self.time_remaining * 0.5 # Let early game all time it needs
        elif self.game_phase == "mid":
            proposed_time = self.time_remaining * 0.3 # Use 30% of the total time left
        elif self.game_phase == "mid-end":
            proposed_time = self.time_remaining * 0.3  # Use 30% of the total time left
        elif self.game_phase == "end":
            proposed_time = self.time_remaining * 0.35  # Use 35% of the total time left

        # Ensure there's a minimum buffer to avoid running out of time completely
        self.time_to_move = min(proposed_time, self.time_remaining - 0.5)

        # If we're in mid-end or end game phase and not using our time efficiently, be more generous in the next moves
        if self.game_phase in ["mid-end", "end"] and self.time_remaining - self.time_to_move > 20:
            self.time_to_move += 5  # Allocate 5 more seconds for the next move

    def set_game_phase(self, board: chess.Board):
        # Early game based on moves
        if board.fullmove_number <= self.early_game_moves:
            self.game_phase = "early"
            return

        # Calculate remaining material complexity percentage, excluding kings
        total_complexity = sum([PIECE_VALUES[piece.piece_type] * PIECE_MOBILITY_WEIGHT[piece.piece_type] 
                                for piece in board.piece_map().values() if piece.piece_type != chess.KING])

        # Calculate maximum possible material complexity, without the kings
        max_complexity_without_kings = sum(PIECE_VALUES[piece_type] * INITIAL_PIECE_COUNT[piece_type] * PIECE_MOBILITY_WEIGHT[piece_type] 
                                        for piece_type in PIECE_VALUES if piece_type != chess.KING)
        material_complexity_percentage = total_complexity / max_complexity_without_kings

        logger.debug(f"The material complexity percentage is: {material_complexity_percentage*100:.2f}%")

        # Determine game phase based on material complexity
        if material_complexity_percentage > 0.25:
            self.game_phase = "mid"
        elif material_complexity_percentage > 0.10:
            self.game_phase = "mid-end"
        else:
            self.game_phase = "end"

    def set_time_remaining(self, limit: chess.engine.Limit):
        if self.color == chess.WHITE:
            self.time_remaining = limit.white_clock if limit.white_clock is not None else 60
        else:
            self.time_remaining = limit.black_clock if limit.black_clock is not None else 60

    def reset_before_search(self):
        self.search_time = time.time()

    def preparation(self, board, limit):
        #-----------------#
        #   Preparation   #
        #-----------------#

        # Board copy, original board
        self.o_board = board.copy()

        # Set engine's playing color
        self.color = self.o_board.turn

        # Game phase
        self.set_game_phase(self.o_board)

        # Time variables
        self.start_time = time.time()
        self.set_time_remaining(limit)
        self.set_time_to_move()

        # Set depth
        self.set_depth()

        # Decaying history heuristic table, 
        for i in range(64):
            for j in range(64):
                self.history_table[i][j] *= 0.9

        # Initializing the Principal Variation object
        self.pv = PV(self.tt_manager)

        #--------------------------#
        #   Preparation: logging   #
        #--------------------------#

        logger.info(f"\nTime to move: {self.time_to_move:0.3f} seconds")
        logger.info(f"Searching depth: {self.depth}, quies: {self.quies_depth}")
        logger.info(f"Game phase: {self.game_phase} game")

    def log_timeout(self, depth: int, e):
        logger.info(f"\n************")
        logger.info(f"*** TIME ***, last full depth searched: {depth - 1}")
        logger.info(f"************\n")
        logger.debug(e)
        
    def log_after_each_iteration(self, depth: int, value: float):
        search_time = time.time() - self.search_time
        logger.debug(f"Depth {depth} Duration: {search_time:.3f} seconds")
        logger.debug(f"Evaluation Value: {value:.1f}")

    def log_final_results(self):
        total_search_time = time.time() - self.start_time
        logger.info(f"TOTAL Duration: {total_search_time:.3f} seconds")
        logger.info(f"\nPV: {self.pv.format_pv(self.o_board, self.game_phase)}\n")

    def get_move(self, board: chess.Board):
        move = self.tt_manager.get_best_move(board)
        if not move or not board.is_legal(move):
            logger.error(f"tt_manager.get_best_move returned None, or the move was illegal")
            logger.error(f"Here is the entry when searching with the board: \n{board}")
            logger.error(f"Primary tt: \n{self.tt_manager.primary_tt.__str__(board)}")
            logger.error(f"Secondary tt: \n{self.tt_manager.secondary_tt.__str__(board)}")
            logger.error(f"Playing a random move...")
            
            moves = self.get_ordered_moves(board, False, 0)

            if moves:
                move = moves[0]
            else:
                logger.error(f"No legal moves were found for board:\n{board}\n I am very confused...")
                move = None

        logger.info(f"\tPlaying the move: {move}")
        return move

#----------------#
#   Evaluating   #
#----------------#
    
# Attention always evaluate max for white, and when returning the evaluation
# function, if board.turn == chess.BLACK, flip the evaluation sign
def evaluate(board: chess.Board, game_phase) -> float:                    
    if board.is_checkmate():
        # The + ply is so it takes as little time as possible to checkmate
        return -CHECKMATE_VALUE + board.ply()

    evaluation = 0
    
    # Iteration:
    # Iteration 1: Count how much material each player has
    # Iteration 3: add a bonus or penalty, the closer the pawn is to promotion
    # Iteration 4: adding a bonus or penalty if a piece is pinned to the king
    # Iteration 5: Double pawns
    # Iteration 7: open files
    # All are on the same method because then it needs to loop over all the squares of the board only once
    evaluation += square_looper(board, game_phase)

    # Iteration 6:
    # same piece move twice penalty
    evaluation += move_repetition_penalty(board)

    # Iteration 8: let it castle!
    evaluation += castling_bonus(board)

    # iteration

    if board.turn == chess.BLACK:
        return -evaluation
    return evaluation

# Iteration 1: Count how much material each player has
# Iteration 2: Develop knights and bishops in the early game
# Iteration 3: add a bonus or penalty, the closer the pawn is to promotion
# Iteration 4: adding a bonus or penalty if a piece is pinned to the king
# Iteration 5: Double pawns
# Iteration 7: open files
# Iteration 9: Rook in the 7th rank
# Iteration 10: no queen in the early game!
def square_looper(board, game_phase):
    # Initialize evaluation scores for white and black to zero
    white_evaluation, black_evaluation = 0, 0

    # Initialize dictionaries for other iterations
    white_pawn_files = {}
    black_pawn_files = {}
    
    # Loop over all squares on the board
    for square, piece in board.piece_map().items():
        # Iteration 1: General material counting
        # donea t the end after all the other iterations edit the value
        value = PIECE_VALUES[piece.piece_type]

        # Iteration 2: develop knights and bishop in the early game
        # Iteration 10: no queen out in the early game
        if game_phase == "early":

            # Iteration 2: Develop knights and bishops in the early game
            if square in KNIGHTS_BISHOPS_STARTING_SQUARES.get(piece.color, []):
                if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    value -= DEVELOPMENT_PENALTY

            # Iteration 10: Penalize moving the queen in the early game
            if piece.piece_type == chess.QUEEN:
                start_square = QUEEN_STARTING_SQUARES[piece.color]
                if square != start_square:
                    value -= EARLY_QUEEN_MOVE_PENALTY

        # Iteration 3: Pawn proximity to promotion
        if piece.piece_type == chess.PAWN:
            if game_phase == "early":
                continue
            elif game_phase == "mid":
                # if mid game, value of pawn += rank (1 through 7)
                rank = chess.square_rank(square)
                value += (rank if piece.color == chess.WHITE else 7 - rank)
            else:
                # If mid-end or end, same as before, but cubed
                rank = chess.square_rank(square)
                value += (rank if piece.color == chess.WHITE else 7 - rank) ** 3

        # Iteration 4: Evaluate pins
        if board.is_pinned(piece.color, square):
            value -= PIN_PENALTY

        # Iteration 5: Doubled pawns
        if piece.piece_type == chess.PAWN:
            file_index = chess.square_file(square)
            if piece.color == chess.WHITE:
                white_pawn_files[file_index] = white_pawn_files.get(file_index, 0) + 1
            else:
                black_pawn_files[file_index] = black_pawn_files.get(file_index, 0) + 1

        # Iteration 7: open files
        if piece.piece_type in [chess.ROOK, chess.QUEEN]:
            file_index = chess.square_file(square)
            
            # Check if the file is open (contains no pawns)
            is_open_file = all(
                board.piece_at(chess.square(file_index, rank)).piece_type != chess.PAWN
                for rank in range(8)
                if board.piece_at(chess.square(file_index, rank))
            )

            if is_open_file:
                value += OPEN_FILE_BONUS  # Add a bonus if on an open file

        # Iteration 9: Rook on 7th rank
        if piece.piece_type == chess.ROOK:
            rank = chess.square_rank(square)
            if (piece.color == chess.WHITE and rank == 6) or (piece.color == chess.BLACK and rank == 1):
                value += ROOK_7TH_RANK_BONUS

        # Add value to the appropriate side's evaluation
        if piece.color == chess.WHITE:
            white_evaluation += value
        else:
            black_evaluation += value

    # Apply double pawn penalty
    for count in white_pawn_files.values():
        if count > 1:
            white_evaluation -= DOUBLE_PAWN_PENALTY * (count - 1)
    for count in black_pawn_files.values():
        if count > 1:
            black_evaluation -= DOUBLE_PAWN_PENALTY * (count - 1)

    return white_evaluation - black_evaluation

# Iteration 6: same piece twice penalty
def move_repetition_penalty(board):
    penalty = 0
    
    # Retrieve the last three moves from the move stack
    if len(board.move_stack) < 3:
        return 0
    
    last_three_moves = list(board.move_stack)[-3:]
    
    # Penalize for moving the same piece twice (self_color's last move)
    if last_three_moves[0].to_square == last_three_moves[2].from_square:
        penalty -= SAME_PIECE_MOVE_PENALTY

    return penalty


# Iteration 8: let it castle!
def castling_bonus(board: chess.Board) -> float:
    bonus = 0.0

    # Check if white can still castle kingside or queenside
    if board.has_kingside_castling_rights(chess.WHITE):
        bonus += CASTLING_RIGHT_BONUS
    if board.has_queenside_castling_rights(chess.WHITE):
        bonus += CASTLING_RIGHT_BONUS

    # Check if black can still castle kingside or queenside
    if board.has_kingside_castling_rights(chess.BLACK):
        bonus -= CASTLING_RIGHT_BONUS
    if board.has_queenside_castling_rights(chess.BLACK):
        bonus -= CASTLING_RIGHT_BONUS

    # Check the move stack for any previous castling moves
    for move in board.move_stack:
        if move in [chess.Move.from_uci('e1g1'), chess.Move.from_uci('e1c1')]:
            bonus += CASTLING_BONUS
        elif move in [chess.Move.from_uci('e8g8'), chess.Move.from_uci('e8c8')]:
            bonus -= CASTLING_BONUS

    return bonus
