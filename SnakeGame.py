import pygame
import numpy as np
import random
import time
from abc import abstractclassmethod


class GameStatusCallback:
    status_quit = -2
    status_invalid_move = -1
    status_killed = 0
    status_resurrected = 1
    status_eat = 2
    status_change_direction = 3
    status_near_food = 4
    
    @abstractclassmethod
    def StatusUpdate(self, status_update: int, **kwargs) -> None:
        raise NotImplementedError("Please implement this method")
    
class SnakeFOV:
    VIEW_DISTANCE = 0
    OFFSET = 0
    VISION_SHAPE = ()
    _SIZE = 0
    def __init__(self, view_distance: int, offset: int) -> None:
        self.VIEW_DISTANCE = view_distance
        self._SIZE = (view_distance) * 2 + 1
        self.OFFSET = offset
        self._template_coords = self.__generate_coordinates()
        self.VISION_SHAPE = self._template_coords.shape[0:2]
        
    def __generate_coordinates(self) -> np.ndarray:
        """get fix vision depends on size"""
        v_dist = self.VIEW_DISTANCE
                
        x_range = np.arange(-v_dist, v_dist + 1)
        y_range = np.arange(-v_dist, v_dist + 1)

        # Create a 2D grid of x and y coordinates
        x_coords, y_coords = np.meshgrid(x_range, y_range)

        # Stack the x and y coordinates to get a 2D array of coordinates (shape is grid)
        coordinates = np.vstack((x_coords.flatten(), y_coords.flatten())).T.reshape((self._SIZE,
                                                                                     self._SIZE,
                                                                                     2))
        return coordinates
    
    def get_FOV_coord(self, head_coords: np.ndarray, direction: int):
        #head_coords = (x, y)
        #obs_vision = shape((SIZE*SIZE, 2))
        
        offset = self.OFFSET
        
        if head_coords.shape != (2, ):
            raise ValueError(f"Value Error: head_coords = {head_coords}")
        
        coords = self._template_coords.copy() + head_coords
        
        match (direction):
            case 0:
                coords += np.array([0, -(offset)])
                coords = np.rot90(coords, 2)
            case 1:
                coords += np.array([0, offset])
            case 2:
                coords += np.array([-(offset), 0])
                coords = np.rot90(coords, 1)
            case 3:
                coords += np.array([offset, 0])
                coords = np.rot90(coords, 3)
        
        return coords
        

class SnakeGame:
    GAME_RUNNING = True
    
    WIDTH = 0
    HEIGHT = 0
    GRID_SIZE = 0
    FPS = 0
    
    COLUMN = 0
    ROW = 0
    
    SPAWN = np.array([0, 0])
    SNAKE_COORD = list(np.array([0, 0])) # [0] - head coords
    DEATH = 0
    HIGHEST_LENGTH = 0
    
    __DIRECTION = 3
    DIRECTION_MAP = np.array([
        [0, -1],  # up
        [0, 1],   # down
        [-1, 0],  # left
        [1, 0]    # right
    ])
    
    VISION_HIGHLIGHT = (50, 50, 50)
    BLOCKS = [
        (0, 0, 0),        # empty
        (0, 255, 255),   # snake body
        (255, 0, 0),     # food
        (255, 255, 255)  # obstacle
    ]
    
    BLOCK_MAP = {
        "empty" : 0,
        "snake body" : 1,
        "food" : 2,
        "obstacle" : 3
    }
    
    game_status_callback = None
    UPDATE_COUNT = 0
    
    # block datas
    # 0 - empty space
    # 1 - snakebody
    # 2 - food
    # 3 - obstacle
    # -1 - out of bound
    def __init__(self, col=40, row=30, gridpx_size=25, fps=10, snake_view_distance=2, snake_view_offset=2):
        # init variables
        self.WIDTH = col * gridpx_size
        self.HEIGHT = row * gridpx_size
        self.GRID_SIZE = gridpx_size
        self.FPS = fps
        self.COLUMN = col
        self.ROW = row
        self.SPAWN = np.array([
            col // 2,
            row // 2
        ])
        
        self.__environment = self._generate_enviroment(self.WIDTH, self.HEIGHT, gridpx_size)
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.FONT = pygame.font.Font(None, 24)
        
        # snake vision initialize
        self.SNAKE_FOV = SnakeFOV(snake_view_distance, snake_view_offset)
        
        # spawn snake
        self._resurrect()
        for i in range(5):
            self._generate_food()
        
        self.update()
    
    def __str__(self) -> str:
        description = "Snake Game Pygame\n"
        description += f"Environment description:\n"
        description += f"width = {self.WIDTH}\n"
        description += f"height = {self.HEIGHT}\n"
        description += f"grid size = {self.GRID_SIZE}\n"
        description += f"column = {self.COLUMN}\n"
        description += f"row = {self.ROW}\n"
        description += f"\nSnake Description:\n"
        description += f"Head coords : {self.SNAKE_COORD[0]}\n"
        description += f"Length: {len(self.SNAKE_COORD)}\n"
        description += f"\nFull numpy:\n"
        description += f"{self.__environment}\n"
        return description
    
    def set_callback(self, callback: GameStatusCallback):
        self.game_status_callback = callback
        
    def _generate_enviroment(self, width: int, height: int, grid_size: int) -> np.ndarray:
        """Generate 2D environment base on screen width and height"""
        return np.zeros((self.ROW, self.COLUMN), int)
    
    def _pop_block(self, coords: np.ndarray):
        """pop block at given coordinate
        coords - 2 element array"""
        self._add_block(coords, "empty")
    
    def _add_block(self, coords: np.ndarray, data_block: str):
        """Adds data block to the environment
        use for adding snake body
        coords - 2 element array"""
        if data_block in self.BLOCK_MAP:
            data = self.BLOCK_MAP[data_block]
            self.__environment[coords[1]][coords[0]] = data
        else:
            raise ValueError("Invalid block int {data}")
    
    def _check_block(self, coords: np.ndarray) -> int:
        np_environ = self.__environment
        if coords[0] >= 0 and coords[0] < self.COLUMN and coords[1] >= 0 and coords[1] < self.ROW:
            return np_environ[coords[1]][coords[0]]
        return -1
    
    def __game_callback(self, status_update: int, **kwargs):
        """Calls the Game Callback"""
        if self.game_status_callback is not None:
            self.game_status_callback.StatusUpdate(status_update, **kwargs)
    
    def _generate_food(self, is_eaten = False):
        """generate food"""
        col = len(self.__environment[0])
        row = len(self.__environment)
        np_environ = self.__environment
        
        while True:
            food = (random.randint(1, col - 2), random.randint(1, row - 2))
            if np_environ[food[1]][food[0]] == 0:
                self._add_block(food, "food")
                break
            
        # update the AI
        self.__game_callback(GameStatusCallback.status_eat)
    
    def _kill(self):
        """kill snake"""
        self.HIGHEST_LENGTH = max(self.HIGHEST_LENGTH, len(self.SNAKE_COORD) - 1)
        self.__environment[0:][self.__environment == 1] = 0
        self.SNAKE_COORD = []
        self.DEATH += 1
        
        # update the AI
        self.__game_callback(GameStatusCallback.status_killed)
        
    def _resurrect(self):
        """resurrect the snake / resets the game"""
        self.__DIRECTION = 3
        self.SNAKE_COORD = []
        
        for n in range(4):
            self.SNAKE_COORD.append(self.SPAWN - np.array([n, 0]))
        
        for coord in self.SNAKE_COORD:
            self._add_block(coord, "snake body")
        
        # update the AI
        self.__game_callback(GameStatusCallback.status_resurrected)
        
    def change_direction(self, direction: int):
        d = self.__DIRECTION
        if (direction // 2 == 0 and d // 2 == 1) or (direction // 2 == 1 and d // 2 == 0):
            self.__DIRECTION = direction
            
            # update the AI
            self.__game_callback(GameStatusCallback.status_change_direction, direction=direction)
    
    def update(self):
        if not self.GAME_RUNNING:
            return
        
        np_environ = self.__environment
        snake = self.SNAKE_COORD
        
        # update snake position (move snake)
        if len(snake) > 0:
            new_head = snake[0] + self.DIRECTION_MAP[self.__DIRECTION]
            snake.insert(0, new_head) # update snake coord
            tail = snake[-1]

            match (self._check_block(new_head)): # check what head hit
                case 1 | -1 | 3:
                    self._kill()
                    time.sleep(0.2)
                    self._resurrect()
                case 2:
                    self._generate_food(is_eaten=True)
                    self._add_block(new_head, "snake body")
                case _:
                    snake.pop()
                    self._pop_block(tail)
                    self._add_block(new_head, "snake body")
                    
        # Check if snake vision is near food
        if self.__check_if_near_food():
            self.__game_callback(GameStatusCallback.status_near_food)

        self.__check_click_exit()
        self._render()
        self.UPDATE_COUNT += 1
    
    def _render(self):
        np_environ = self.__environment
        grid_size = self.GRID_SIZE

        self.screen.fill((0, 0, 0))
        
        self.__render_vision_on_box()
        
        for row, row_block in enumerate(np_environ):
            for col, block in enumerate(row_block):
                if block > 0:
                    pygame.draw.rect(self.screen, self.BLOCKS[block], (col * grid_size, row * grid_size, grid_size, grid_size))
        # Text render
        length_text = self.FONT.render(f"Length: {len(self.SNAKE_COORD)}  [{self.HIGHEST_LENGTH}]     Deaths: {self.DEATH}", True, (155, 155, 155))
        self.screen.blit(length_text, (0, 0))
        
        pygame.display.flip()
        self.clock.tick(self.FPS)
        
    def __render_vision_on_box(self):
        obs = np.rot90(self.get_environment_observation(), 2)
        grid_size = int(self.GRID_SIZE * 0.75)
        shape = obs.shape
        
        corner = (self.WIDTH - (grid_size * shape[1]), 0)
        vis_text = self.FONT.render("What it sees (AI FoV): ", True, (155, 155, 155))
        for row, row_block in enumerate(obs):
            for col, block in enumerate(row_block):
                color = self.BLOCKS[block]
                if block == 0:
                    color = self.VISION_HIGHLIGHT
                if block >= 0:
                    pygame.draw.rect(self.screen, color, (corner[0] + (col * grid_size), row * grid_size, grid_size, grid_size))
        self.screen.blit(vis_text, (corner[0] - 150, corner[1]))
        
    def __render_vision_area(self):
        """renders the FOV of the snake"""
        head_coord = self.SNAKE_COORD[0]
        direction = self.__DIRECTION
        grid_size = self.GRID_SIZE
        
        fov_coords = self.SNAKE_FOV.get_FOV_coord(head_coord, direction)
        x, y = fov_coords[0][0]
        x2, y2 = fov_coords[-1][-1]
        
        start_x, start_y = min(x, x2), min(y, y2)
        end_x, end_y = max(x, x2) + 1, max(y, y2) + 1
        dis_x, dis_y = end_x - start_x, end_y - start_y
        
        pygame.draw.rect(self.screen, self.VISION_HIGHLIGHT, (start_x * grid_size, start_y * grid_size, dis_x * grid_size, dis_y * grid_size))
    
    near_food_bool = False
    def __check_if_near_food(self) -> bool:
        obs = self.get_flatten_observation()
        if self.BLOCK_MAP["food"] in obs:
            try:
                return not self.near_food_bool
            finally:
                self.near_food_bool = True
        self.near_food_bool = False
        return self.near_food_bool
    
    def __check_click_exit(self):
        """checks for exit button click then closes the window
        *must be called only in self.update function"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.exit()
                print("exit clicked...")
                break
        
    def exit(self):
        self.GAME_RUNNING = False
        pygame.display.quit()
        pygame.quit()
    
    last_obs = None
    last_upd_count = 0
    def get_environment_observation(self) -> np.ndarray:
        if self.last_upd_count == self.UPDATE_COUNT and self.last_obs is not None:
            return self.last_obs

        head_coord = self.SNAKE_COORD[0]
        direction = self.__DIRECTION
        
        fov_coords = self.SNAKE_FOV.get_FOV_coord(head_coord, direction) # matrix form 2d
        row, col = self.SNAKE_FOV.VISION_SHAPE
        empty_obs = np.zeros((row, col), int)
        
        for y in range(row):
            for x in range(col):
                empty_obs[y][x] = self._check_block(fov_coords[y][x])
        
        self.last_obs = empty_obs
        self.last_upd_count = self.UPDATE_COUNT
        return empty_obs
    
    def get_flatten_observation(self):
        obs = self.get_environment_observation().flatten()
        return obs
        
    # snake properties
    @property
    def direction(self):
        return self.__DIRECTION
        
if __name__ == '__main__':
    snake_game = SnakeGame()
    running = True
    snake_game.get_environment_observation()
    while snake_game.GAME_RUNNING:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                snake_game.exit()
                print("exit clicked...")
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    snake_game.change_direction(0)
                elif event.key == pygame.K_DOWN:
                    snake_game.change_direction(1)
                elif event.key == pygame.K_LEFT:
                    snake_game.change_direction(2)
                elif event.key == pygame.K_RIGHT:
                    snake_game.change_direction(3)
        if snake_game.GAME_RUNNING:
            snake_game.update()
    