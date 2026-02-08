import pygame
import random
import numpy as np
import traceback
import copy
import math

# --- WRAPPER ---
def main_wrapper():
    try:
        main()
    except Exception as e:
        print("\nCRASH DETECTED!"); traceback.print_exc(); input("Enter to close...")

# --- CONFIG ---
SCREEN_WIDTH = 750; SCREEN_HEIGHT = 700
BLOCK_SIZE = 25; COLS = 10; ROWS = 20

# SPEEDS
FPS_TURBO = 999
FPS_NORMAL = 3      

# SEARCH
BEAM_WIDTH = 8
SEARCH_DEPTH = 2

# EVOLUTION SETTINGS
EVOLUTION_START_SCORE = 100000 
REQUIRED_CANDIDATES = 3
VALIDATION_RUNS = 3    # Games to prove consistency
DIVERSITY_THRESHOLD = 5.0 

# AGGRESSION
BASE_NOISE = 1.5       
KICK_NOISE = 20.0      
STAGNATION_LIMIT = 10  

# RISING FLOOR
FLOOR_START_INTERVAL = 200 
FLOOR_DECREASE = 10         

# COLORS
BLACK = (20,20,20); GRAY = (50,50,50); WHITE = (240,240,240)
GREEN = (50,200,50); RED = (200,50,50); CYAN = (50,200,200)
YELLOW = (200, 200, 50); BLUE = (50, 50, 255); ORANGE = (255, 165, 0); PURPLE = (128, 0, 128)
BEDROCK_COLOR = (40, 40, 40)
BTN_COLOR = (0, 100, 200); BTN_HOVER = (0, 150, 255)

SHAPES = [
    [[1, 1, 1, 1]], [[1, 1], [1, 1]], [[0, 1, 0], [1, 1, 1]], 
    [[1, 0, 0], [1, 1, 1]], [[0, 0, 1], [1, 1, 1]], 
    [[0, 1, 1], [1, 1, 0]], [[1, 1, 0], [0, 1, 1]]
]
SHAPE_COLORS = [CYAN, YELLOW, PURPLE, ORANGE, BLUE, GREEN, RED]

# --- ENVIRONMENT ---
class TetrisEnv:
    def __init__(self): self.reset()

    def reset(self):
        self.board = [[(0,0,0)] * COLS for _ in range(ROWS)]
        self.score = 0
        self.lines_cleared = 0
        self.level = 0
        self.floor_timer = FLOOR_START_INTERVAL
        self.game_over = False
        self.current_piece = self._get_random_piece()
        self.next_piece = self._get_random_piece()
        return self.board

    def _get_random_piece(self):
        idx = random.randint(0, len(SHAPES) - 1)
        return {"shape": SHAPES[idx], "idx": idx, "color": SHAPE_COLORS[idx]}

    def raise_floor(self):
        if any(c != (0,0,0) for c in self.board[0]):
            self.game_over = True
            return
        self.board.pop(0) 
        self.board.append([BEDROCK_COLOR] * COLS)

    def step(self, action):
        x, rot_idx = action
        shape = self.current_piece["shape"]
        color = self.current_piece["color"]
        
        for _ in range(rot_idx): shape = [list(row) for row in zip(*shape[::-1])]
            
        dropped_board, lines = self._hard_drop_simulate(self.board, shape, x, color)
        
        if dropped_board is None:
            self.game_over = True
            return self.board, -50.0, True, 0
        
        self.board = dropped_board
        
        self.floor_timer -= 1
        if self.floor_timer <= 0:
            self.raise_floor()
            new_int = max(50, FLOOR_START_INTERVAL - (self.level * FLOOR_DECREASE))
            self.floor_timer = new_int
            self.level += 1
        
        multiplier = self.level + 1
        pts = 0
        if lines == 1: pts = 40
        elif lines == 2: pts = 100
        elif lines == 3: pts = 300
        elif lines == 4: pts = 1200
        
        self.score += pts * multiplier
        tetris_bonus_flag = 1.0 if lines >= 4 else 0.0
        
        self.current_piece = self.next_piece
        self.next_piece = self._get_random_piece()
        
        if self._check_collision(self.board, self.current_piece["shape"], 3, 0):
            self.game_over = True
            
        return self.board, pts * multiplier, self.game_over, tetris_bonus_flag

    def _check_collision(self, board, shape, off_x, off_y):
        for cy, row in enumerate(shape):
            for cx, cell in enumerate(row):
                if cell:
                    bx, by = off_x + cx, off_y + cy
                    if bx < 0 or bx >= COLS: return True
                    if by >= ROWS: return True
                    if by >= 0 and board[by][bx] != (0,0,0): return True
        return False

    def _hard_drop_simulate(self, board, shape, x, color=(255,255,255)):
        piece_width = len(shape[0])
        if x < 0 or x + piece_width > COLS: return None, 0

        y = 0
        while not self._check_collision(board, shape, x, y + 1): y += 1
        
        for cy, row in enumerate(shape):
            for cx, cell in enumerate(row):
                if cell:
                    if y + cy < 0: return None, 0

        new_board = [r[:] for r in board]
        for cy, row in enumerate(shape):
            for cx, cell in enumerate(row):
                if cell: 
                    if 0 <= y + cy < ROWS:
                        new_board[y + cy][x + cx] = color
        
        lines = 0; final_board = []
        for r in new_board:
            if (0,0,0) not in r:
                if r[0] == BEDROCK_COLOR: final_board.append(r)
                else: lines += 1
            else: final_board.append(r)
                
        while len(final_board) < ROWS: final_board.insert(0, [(0,0,0)]*COLS)
        return final_board, lines

    def get_legal_moves(self, board, piece_dict):
        moves = []
        shape = piece_dict["shape"]
        dummy_color = (1,1,1) 
        for rot in range(4):
            width = len(shape[0])
            for x in range(COLS - width + 1): 
                res_board, lines = self._hard_drop_simulate(board, shape, x, dummy_color)
                if res_board is not None:
                    is_tetris = 1.0 if lines >= 4 else 0.0
                    moves.append({ "board": res_board, "lines": lines, "x": x, "rot": rot, "tetris": is_tetris })
            shape = [list(row) for row in zip(*shape[::-1])]
        return moves

# --- AGENT ---
class Agent:
    def __init__(self):
        self.weights = np.zeros(5, dtype=np.float64)
        
        # Format: [{'weights': w, 'high_score': s, 'validation_scores': []}, ...]
        self.candidates = [] 
        
        self.phase = "PRIMORDIAL"
        self.validating_idx = 0
        self.runs_left = 0
        
        self.king_weights = None
        self.best_score_ever = 0
        self.stagnation_counter = 0

    def generate_random_genome(self):
        w = np.zeros(5)
        w[0] = random.uniform(-10.0, 10.0)  # Lines
        w[1] = random.uniform(-20.0, -5.0)  # Holes
        w[2] = random.uniform(-10.0, 10.0)   # Bump
        w[3] = random.uniform(-10.0, 10.0)   # Height
        w[4] = random.uniform(5.0, 10.0) # Tetris
        return w

    def prepare_episode_weights(self):
        if self.phase == "PRIMORDIAL":
            self.test_weights = self.generate_random_genome()
            
        elif self.phase == "VALIDATION":
            # Exact weights of candidate, no noise
            self.test_weights = self.candidates[self.validating_idx]['weights']
            
        elif self.phase == "CIVILIZATION":
            # Aggressive Evolution from the King
            if self.stagnation_counter >= STAGNATION_LIMIT:
                noise_scale = KICK_NOISE
                print(f">>> STAGNATION KICK! (Noise: {KICK_NOISE})")
                self.stagnation_counter = 0
            else:
                noise_scale = BASE_NOISE

            # Mutate King's weights directly
            noise = np.random.normal(scale=noise_scale, size=5)
            self.test_weights = self.king_weights + noise

    def get_features(self, board, lines_cleared, is_tetris):
        heights = [0] * COLS; holes = 0
        for c in range(COLS):
            found = False
            for r in range(ROWS):
                val = board[r][c]
                if val != (0,0,0): 
                    if not found: heights[c] = ROWS - r; found = True
                elif found and val == (0,0,0): holes += 1
        
        agg_height = sum(heights)
        min_h = min(heights); well_col = heights.index(min_h)
        bumpiness = 0
        for c in range(COLS - 1):
            h1 = heights[c]; h2 = heights[c+1]
            diff = abs(h1 - h2)
            if c == well_col or (c+1) == well_col:
                if diff <= 4: diff = 0 
                else: diff -= 4
            bumpiness += diff
        return np.array([lines_cleared, holes, bumpiness, agg_height, is_tetris], dtype=np.float64)

    def evaluate(self, board, lines, is_tetris):
        feats = self.get_features(board, lines, is_tetris)
        return np.dot(self.test_weights, feats)

    def beam_search(self, env):
        moves_1 = env.get_legal_moves(env.board, env.current_piece)
        if not moves_1: return None

        scored_moves_1 = []
        for m in moves_1:
            score = self.evaluate(m["board"], m["lines"], m["tetris"])
            scored_moves_1.append((score, m))
        
        scored_moves_1.sort(key=lambda x: x[0], reverse=True)
        beam = scored_moves_1[:BEAM_WIDTH]
        
        if SEARCH_DEPTH == 1: return beam[0][1]

        best_total = -float('inf'); best_move = None
        for score_1, m1 in beam:
            moves_2 = env.get_legal_moves(m1["board"], env.next_piece)
            if not moves_2: future = -99999
            else:
                future = -float('inf')
                for m2 in moves_2:
                    s = self.evaluate(m2["board"], m2["lines"], m2["tetris"])
                    if s > future: future = s
            
            total = score_1 + future
            if total > best_total:
                best_total = total
                best_move = m1
        
        return best_move

    def check_diversity(self, new_weights):
        for cand in self.candidates:
            dist = np.linalg.norm(new_weights - cand['weights'])
            if dist < DIVERSITY_THRESHOLD: return False 
        return True

    def update_policy(self, score):
        
        # --- PHASE 1: PRIMORDIAL ---
        if self.phase == "PRIMORDIAL":
            if score >= EVOLUTION_START_SCORE:
                # Store RAW weights
                if self.check_diversity(self.test_weights):
                    print(f">>> CANDIDATE FOUND! ({score})")
                    self.candidates.append({
                        'weights': self.test_weights.copy(),
                        'high_score': score,
                        'validation_scores': []
                    })
                    
                    if len(self.candidates) >= REQUIRED_CANDIDATES:
                        self.phase = "VALIDATION"
                        self.validating_idx = 0
                        self.runs_left = VALIDATION_RUNS
                        print(">>> STARTING GAUNTLET (Validation)")
                else:
                    print(f"Duplicate Strategy ({score}). Rejected.")

        # --- PHASE 2: VALIDATION ---
        elif self.phase == "VALIDATION":
            current_cand = self.candidates[self.validating_idx]
            current_cand['validation_scores'].append(score)
            print(f"Candidate #{self.validating_idx+1} Run: {score}")
            
            self.runs_left -= 1
            if self.runs_left <= 0:
                self.validating_idx += 1
                self.runs_left = VALIDATION_RUNS
                
                if self.validating_idx >= len(self.candidates):
                    print(">>> GAUNTLET COMPLETE. CROWNING THE KING.")
                    self.crown_the_king()

        # --- PHASE 3: CIVILIZATION (Aggro Evolution) ---
        elif self.phase == "CIVILIZATION":
            if score > self.best_score_ever:
                print(f">>> NEW KING RECORD: {score}")
                # The mutant becomes the new King
                self.king_weights = self.test_weights.copy()
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

        if score > self.best_score_ever: self.best_score_ever = score
        self.prepare_episode_weights()

    def crown_the_king(self):
        best_avg = -1
        king_idx = -1
        
        print("\n=== GAUNTLET RESULTS ===")
        for i, cand in enumerate(self.candidates):
            avg = sum(cand['validation_scores']) / len(cand['validation_scores'])
            print(f"Cand #{i+1} Avg: {avg:.0f} (Max: {cand['high_score']})")
            
            if avg > best_avg:
                best_avg = avg
                king_idx = i
                
        print(f"=== THE KING IS CANDIDATE #{king_idx+1} ({best_avg:.0f}) ===")
        
        # Keep raw weights
        self.king_weights = self.candidates[king_idx]['weights'].copy()
        self.best_score_ever = self.candidates[king_idx]['high_score']
        
        self.phase = "CIVILIZATION"

# --- UI & MAIN ---
def draw_ui(screen, env, agent, episode, btn_rect, current_fps):
    dx = 380 
    
    if agent.phase == "PRIMORDIAL":
        color_phase = RED
        txt_phase = "PRIMORDIAL (Head-Hunting)"
        sub_txt = f"Candidates: {len(agent.candidates)}/{REQUIRED_CANDIDATES}"
    
    elif agent.phase == "VALIDATION":
        color_phase = YELLOW
        txt_phase = "THE GAUNTLET"
        sub_txt = f"Testing Cand #{agent.validating_idx+1} ({agent.runs_left} left)"
    
    else:
        color_phase = CYAN
        txt_phase = "CIVILIZATION (Aggro Evolve)"
        sub_txt = f"Stagnation: {agent.stagnation_counter}/{STAGNATION_LIMIT}"

    draw_text(screen, "TETRIS BOT", dx, 50, 30, YELLOW)
    draw_text(screen, txt_phase, dx, 90, 20, color_phase)
    draw_text(screen, sub_txt, dx, 115, 20, WHITE)
    
    draw_text(screen, f"Episode: {episode}", dx, 150, 24)
    draw_text(screen, f"Best Ever: {agent.best_score_ever}", dx, 180, 24)
    draw_text(screen, f"Current: {env.score}", dx, 210, 24, WHITE)
    draw_text(screen, f"Level: {env.level}", dx, 240, 24, ORANGE)
    
    draw_text(screen, "NEXT:", dx, 280, 24, CYAN)
    draw_next_piece(screen, env.next_piece, dx, 310)
    
    draw_text(screen, "CURRENT Weights:", dx, 440, 26, CYAN)
    lbls = ["Lines", "Holes", "Bump", "Height", "Tetris"]
    disp_w = agent.test_weights
    for i, (l, w) in enumerate(zip(lbls, disp_w)):
        c = GREEN if w > 0 else RED
        draw_text(screen, f"{l}: {w:.1f}", dx, 470 + i*25, 22, c)
        
    mouse_pos = pygame.mouse.get_pos()
    color = BTN_HOVER if btn_rect.collidepoint(mouse_pos) else BTN_COLOR
    pygame.draw.rect(screen, color, btn_rect)
    btn_txt = "SPEED: TURBO" if current_fps == FPS_TURBO else "SPEED: NORMAL"
    draw_text(screen, btn_txt, btn_rect.x + 15, btn_rect.y + 10, 20, WHITE)

def draw_next_piece(screen, piece, x, y):
    pygame.draw.rect(screen, GRAY, (x, y, 100, 100), 2)
    shape = piece["shape"]
    color = piece["color"]
    start_x = x + 10; start_y = y + 10
    for r, row in enumerate(shape):
        for c, cell in enumerate(row):
            if cell: pygame.draw.rect(screen, color, (start_x + c*20, start_y + r*20, 18, 18))

def draw_text(screen, text, x, y, size=18, color=WHITE):
    font = pygame.font.Font(None, size)
    img = font.render(text, True, color)
    screen.blit(img, (x, y))

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Tetris Bot")
    clock = pygame.time.Clock()

    env = TetrisEnv()
    agent = Agent()
    agent.prepare_episode_weights()
    
    episode = 0
    render_enabled = True
    current_fps = FPS_TURBO
    btn_rect = pygame.Rect(400, 620, 160, 40)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB: render_enabled = not render_enabled
                if event.key == pygame.K_r: 
                    agent = Agent(); agent.prepare_episode_weights()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if btn_rect.collidepoint(event.pos):
                    if current_fps == FPS_TURBO: current_fps = FPS_NORMAL
                    else: current_fps = FPS_TURBO

        best_move = agent.beam_search(env)
        
        if best_move:
            _, _, done, _ = env.step((best_move["x"], best_move["rot"]))
        else:
            done = True

        if done:
            agent.update_policy(env.score)
            episode += 1
            env.reset()

        if render_enabled:
            screen.fill(BLACK)
            off_x, off_y = 50, 50
            pygame.draw.rect(screen, GRAY, (off_x-5, off_y-5, COLS*BLOCK_SIZE+10, ROWS*BLOCK_SIZE+10), 2)
            for r in range(ROWS):
                for c in range(COLS):
                    color_val = env.board[r][c]
                    if color_val != (0,0,0):
                        pygame.draw.rect(screen, color_val, (off_x + c*BLOCK_SIZE, off_y + r*BLOCK_SIZE, BLOCK_SIZE-1, BLOCK_SIZE-1))
            draw_ui(screen, env, agent, episode, btn_rect, current_fps)
            pygame.display.flip()
            clock.tick(current_fps)
        else:
            pygame.event.pump()
    pygame.quit()

if __name__ == "__main__": main_wrapper()