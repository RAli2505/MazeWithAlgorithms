"""
Эта программа на Pygame визуализирует поиск пути в лабиринте.

Интерфейс: окно показывает лабиринт; стены и поля отмечены синим, поиск алгоритма – розовым, кратчайший путь – жёлтым/фиолетовым.

Параметры: размеры лабиринта можно менять с помощью изменения (WIDTH, HEIGHT) и ячеек (CELL_SIZE).

Алгоритмы: реализованы BFS, DFS, A* с Манхэттенской и Евклидовой эвристикой.

Управление:

1–4 — выбор алгоритма,

SPACE — новый лабиринт,

S — запуск поиска,

ESC — выход.
"""
import pygame 
import random
import time
import math
from collections import deque
import heapq

# Инициализация Pygame
pygame.init()

# Настройки лабиринта
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 40
ROWS = HEIGHT // CELL_SIZE
COLS = WIDTH // CELL_SIZE

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
WALL_COLOR = (50, 50, 150)
PATH_COLOR = (200, 200, 255)
VISITED_COLOR = (150, 150, 200)
EXPLORE_COLOR = (255, 200, 200)

# Создание окна
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Поиск пути в лабиринте с алгоритмами BSF,DFS,A*")
clock = pygame.time.Clock()

class Cell:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.visited = False
        self.walls = [True, True, True, True]  # top, right, bottom, left
        self.in_path = False
        self.explored = False
        self.g_score = float('inf')  # стоимость пути от старта
        self.f_score = float('inf')  # общая оценка стоимости (g + h)
        self.came_from = None  # предыдущая клетка в пути
    
    def draw(self):
        x = self.col * CELL_SIZE
        y = self.row * CELL_SIZE
        
        # Рисуем клетку
        if self.in_path:
            pygame.draw.rect(screen, YELLOW, (x, y, CELL_SIZE, CELL_SIZE))
        elif self.explored:
            pygame.draw.rect(screen, EXPLORE_COLOR, (x, y, CELL_SIZE, CELL_SIZE))
        elif self.visited:
            pygame.draw.rect(screen, VISITED_COLOR, (x, y, CELL_SIZE, CELL_SIZE))
        else:
            pygame.draw.rect(screen, PATH_COLOR, (x, y, CELL_SIZE, CELL_SIZE))
        
        # Рисуем стены
        if self.walls[0]:  # top
            pygame.draw.line(screen, WALL_COLOR, (x, y), (x + CELL_SIZE, y), 2)
        if self.walls[1]:  # right
            pygame.draw.line(screen, WALL_COLOR, (x + CELL_SIZE, y), (x + CELL_SIZE, y + CELL_SIZE), 2)
        if self.walls[2]:  # bottom
            pygame.draw.line(screen, WALL_COLOR, (x, y + CELL_SIZE), (x + CELL_SIZE, y + CELL_SIZE), 2)
        if self.walls[3]:  # left
            pygame.draw.line(screen, WALL_COLOR, (x, y), (x, y + CELL_SIZE), 2)
    
    def get_neighbors(self, grid):
        neighbors = []
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
        
        for dr, dc in directions:
            new_row, new_col = self.row + dr, self.col + dc
            if (0 <= new_row < ROWS and 0 <= new_col < COLS):
                # Проверяем, нет ли стены между клетками
                if dr == -1 and not self.walls[0]:  # up
                    neighbors.append(grid[new_row][new_col])
                elif dr == 1 and not self.walls[2]:  # down
                    neighbors.append(grid[new_row][new_col])
                elif dc == -1 and not self.walls[3]:  # left
                    neighbors.append(grid[new_row][new_col])
                elif dc == 1 and not self.walls[1]:  # right
                    neighbors.append(grid[new_row][new_col])
        
        return neighbors

    def reset_scores(self):
        """Сброс оценок для A*"""
        self.g_score = float('inf')
        self.f_score = float('inf')
        self.came_from = None

def remove_walls(current, next_cell, grid):
    # Определяем направление между клетками
    dr = next_cell[0] - current.row
    dc = next_cell[1] - current.col
    
    if dr == -1:  # up
        current.walls[0] = False
        grid[next_cell[0]][next_cell[1]].walls[2] = False
    elif dr == 1:  # down
        current.walls[2] = False
        grid[next_cell[0]][next_cell[1]].walls[0] = False
    elif dc == -1:  # left
        current.walls[3] = False
        grid[next_cell[0]][next_cell[1]].walls[1] = False
    elif dc == 1:  # right
        current.walls[1] = False
        grid[next_cell[0]][next_cell[1]].walls[3] = False

def generate_maze():
    # Создаем сетку клеток
    grid = [[Cell(row, col) for col in range(COLS)] for row in range(ROWS)]
    stack = []
    
    # Начинаем с начальной клетки
    current = grid[0][0]
    current.visited = True
    stack.append(current)
    
    # Генерация лабиринта с использованием алгоритма Depth-First Search
    while stack:
        current = stack[-1]
        neighbors = []
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        for dr, dc in directions:
            new_row, new_col = current.row + dr, current.col + dc
            if (0 <= new_row < ROWS and 0 <= new_col < COLS and 
                not grid[new_row][new_col].visited):
                neighbors.append((new_row, new_col))
        
        if not neighbors:
            stack.pop()
            continue
        
        # Выбираем случайного соседа
        next_row, next_col = random.choice(neighbors)
        next_cell = grid[next_row][next_col]
        
        # Убираем стену между текущей и следующей клеткой
        remove_walls(current, (next_row, next_col), grid)
        
        # Переходим к следующей клетке
        next_cell.visited = True
        stack.append(next_cell)
    
    # Сбрасываем флаги для поиска пути
    for row in grid:
        for cell in row:
            cell.visited = False
            cell.explored = False
            cell.in_path = False
            cell.reset_scores()
    
    return grid

def heuristic(cell1, cell2):
    """Эвристическая функция (манхэттенское расстояние)"""
    return abs(cell1.row - cell2.row) + abs(cell1.col - cell2.col)

def euclidean_distance(cell1, cell2):
    """Евклидово расстояние"""
    return math.sqrt((cell1.row - cell2.row)**2 + (cell1.col - cell2.col)**2)

def find_path_bfs(grid, start, end):
    """Поиск пути с использованием BFS"""
    queue = deque([start])
    visited = {start: None}  # Для хранения пути
    
    while queue:
        current = queue.popleft()
        current.explored = True
        
        if current == end:
            # Восстанавливаем путь
            path = []
            while current:
                path.append(current)
                current = visited[current]
            return path[::-1]  # Разворачиваем путь
        
        for neighbor in current.get_neighbors(grid):
            if neighbor not in visited:
                visited[neighbor] = current
                queue.append(neighbor)
                neighbor.visited = True
    
    return None

def find_path_dfs_iterative(grid, start, end):
    """Итеративная версия DFS"""
    stack = [start]
    visited = {start: None}  # Для хранения пути
    start.visited = True
    
    while stack:
        current = stack.pop()
        current.explored = True
        
        if current == end:
            # Восстанавливаем путь
            path = []
            while current:
                path.append(current)
                current = visited[current]
            return path[::-1]  # Разворачиваем путь
        
        # Получаем соседей в случайном порядке для разнообразия
        neighbors = current.get_neighbors(grid)
        random.shuffle(neighbors)
        
        for neighbor in neighbors:
            if not neighbor.visited:
                neighbor.visited = True
                visited[neighbor] = current
                stack.append(neighbor)
    
    return None

def find_path_astar(grid, start, end, heuristic_func=heuristic):
    """Поиск пути с использованием алгоритма A*"""
    # Сбрасываем оценки
    for row in grid:
        for cell in row:
            cell.reset_scores()
            cell.visited = False
            cell.explored = False
    
    # Инициализация начальной клетки
    start.g_score = 0
    start.f_score = heuristic_func(start, end)
    
    # Открытый список (приоритетная очередь)
    open_set = []
    heapq.heappush(open_set, (start.f_score, id(start), start))
    open_set_dict = {start: True}
    
    while open_set:
        # Берем клетку с наименьшей f_score
        current = heapq.heappop(open_set)[2]
        open_set_dict.pop(current, None)
        current.explored = True
        
        if current == end:
            # Восстанавливаем путь
            path = []
            while current:
                path.append(current)
                current = current.came_from
            return path[::-1]
        
        current.visited = True
        
        for neighbor in current.get_neighbors(grid):
            if neighbor.visited:
                continue
                
            # Предполагаем, что стоимость перехода между соседями = 1
            tentative_g_score = current.g_score + 1
            
            if tentative_g_score < neighbor.g_score:
                # Этот путь лучше предыдущего
                neighbor.came_from = current
                neighbor.g_score = tentative_g_score
                neighbor.f_score = tentative_g_score + heuristic_func(neighbor, end)
                
                if neighbor not in open_set_dict:
                    open_set_dict[neighbor] = True
                    heapq.heappush(open_set, (neighbor.f_score, id(neighbor), neighbor))
    
    return None  # Путь не найден

def bfs_generator(grid, start, end):
    """Генератор для пошагового отображения BFS"""
    queue = deque([start])
    visited = {start: None}
    start.visited = True
    
    while queue:
        current = queue.popleft()
        current.explored = True
        yield current
        
        if current == end:
            break
        
        for neighbor in current.get_neighbors(grid):
            if neighbor not in visited:
                visited[neighbor] = current
                queue.append(neighbor)
                neighbor.visited = True

def dfs_iterative_generator(grid, start, end):
    """Генератор для пошагового отображения итеративного DFS"""
    stack = [start]
    visited = {start: None}
    start.visited = True
    
    while stack:
        current = stack.pop()
        current.explored = True
        yield current
        
        if current == end:
            break
        
        neighbors = current.get_neighbors(grid)
        random.shuffle(neighbors)
        
        for neighbor in neighbors:
            if not neighbor.visited:
                neighbor.visited = True
                visited[neighbor] = current
                stack.append(neighbor)

def astar_generator(grid, start, end, heuristic_func=heuristic):
    """Генератор для пошагового отображения A*"""
    # Сбрасываем оценки
    for row in grid:
        for cell in row:
            cell.reset_scores()
            cell.visited = False
            cell.explored = False
    
    # Инициализация начальной клетки
    start.g_score = 0
    start.f_score = heuristic_func(start, end)
    
    open_set = []
    heapq.heappush(open_set, (start.f_score, id(start), start))
    open_set_dict = {start: True}
    
    while open_set:
        current = heapq.heappop(open_set)[2]
        open_set_dict.pop(current, None)
        current.explored = True
        
        yield current  # Возвращаем текущую клетку для анимации
        
        if current == end:
            break
        
        current.visited = True
        
        for neighbor in current.get_neighbors(grid):
            if neighbor.visited:
                continue
                
            tentative_g_score = current.g_score + 1
            
            if tentative_g_score < neighbor.g_score:
                neighbor.came_from = current
                neighbor.g_score = tentative_g_score
                neighbor.f_score = tentative_g_score + heuristic_func(neighbor, end)
                
                if neighbor not in open_set_dict:
                    open_set_dict[neighbor] = True
                    heapq.heappush(open_set, (neighbor.f_score, id(neighbor), neighbor))

def get_algorithm_generator(grid, start, end, algorithm_type):
    """Возвращает генератор для выбранного алгоритма"""
    # Сбрасываем флаги
    for row in grid:
        for cell in row:
            cell.visited = False
            cell.explored = False
            cell.in_path = False
            cell.reset_scores()
    
    if algorithm_type == "bfs":
        return bfs_generator(grid, start, end)
    elif algorithm_type == "dfs_iterative":
        return dfs_iterative_generator(grid, start, end)
    elif algorithm_type == "astar_manhattan":
        return astar_generator(grid, start, end, heuristic)
    elif algorithm_type == "astar_euclidean":
        return astar_generator(grid, start, end, euclidean_distance)
    else:
        return astar_generator(grid, start, end, heuristic)

def find_path_with_algorithm(grid, start, end, algorithm_type):
    """Находит путь с использованием выбранного алгоритма"""
    # Сбрасываем флаги
    for row in grid:
        for cell in row:
            cell.visited = False
            cell.explored = False
            cell.in_path = False
            cell.reset_scores()
    
    if algorithm_type == "bfs":
        return find_path_bfs(grid, start, end)
    elif algorithm_type == "dfs_iterative":
        return find_path_dfs_iterative(grid, start, end)
    elif algorithm_type == "astar_manhattan":
        return find_path_astar(grid, start, end, heuristic)
    elif algorithm_type == "astar_euclidean":
        return find_path_astar(grid, start, end, euclidean_distance)
    else:
        return find_path_astar(grid, start, end, heuristic)

def animate_exploration(grid, path_generator):
    """Анимированное отображение процесса исследования"""
    try:
        for current_cell in path_generator:
            draw_maze(grid, None, current_cell, True)
            pygame.display.flip()
            time.sleep(0.05)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return False
        return True
    except:
        return False

def animate_solution(grid, path):
    """Анимированное отображение решения"""
    for i, cell in enumerate(path):
        cell.in_path = True
        draw_maze(grid, path[:i+1])
        pygame.display.flip()
        time.sleep(0.1)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
    
    return True

def draw_maze(grid, path=None, current_cell=None, exploring=False, algorithm_name="", gen_time=None, explore_time=None, solve_time=None):
    screen.fill(WHITE)
    
    # Рисуем все клетки
    for row in grid:
        for cell in row:
            cell.draw()
    
    # Подсвечиваем текущую клетку при исследовании
    if current_cell and exploring:
        x = current_cell.col * CELL_SIZE + 5
        y = current_cell.row * CELL_SIZE + 5
        
        # Для A* показываем оценки
        if hasattr(current_cell, 'f_score') and current_cell.f_score < float('inf'):
            # Рисуем клетку с градиентом в зависимости от оценки
            score_ratio = min(current_cell.f_score / 50, 1.0)  # Нормализуем оценку
            color = (255, int(255 * (1 - score_ratio)), 0)
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE - 10, CELL_SIZE - 10))
            
            # Отображаем оценку
            font = pygame.font.SysFont(None, 12)
            text = font.render(f"{current_cell.f_score:.1f}", True, BLACK)
            screen.blit(text, (x + 2, y + 2))
        else:
            pygame.draw.rect(screen, RED, (x, y, CELL_SIZE - 10, CELL_SIZE - 10))
    
    # Рисуем путь
    if path:
        for i, cell in enumerate(path):
            if cell != path[0] and cell != path[-1]:  # Не рисуем старт и финиш
                x = cell.col * CELL_SIZE + 8
                y = cell.row * CELL_SIZE + 8
                pygame.draw.rect(screen, PURPLE, (x, y, CELL_SIZE - 16, CELL_SIZE - 16))
    
    # Рисуем старт и финиш
    start_x, start_y = 5, 5
    end_x, end_y = (COLS-1) * CELL_SIZE + 5, (ROWS-1) * CELL_SIZE + 5
    pygame.draw.rect(screen, GREEN, (start_x, start_y, CELL_SIZE - 10, CELL_SIZE - 10))
    pygame.draw.rect(screen, BLUE, (end_x, end_y, CELL_SIZE - 10, CELL_SIZE - 10))
    
    # Отображаем информацию об алгоритме
    font = pygame.font.SysFont(None, 20)
    text = font.render(f"Выбор алгоритма: {algorithm_name}", True, BLACK)
    screen.blit(text, (10, HEIGHT - 110))
    
    # Инструкции
    instructions = [
        "1 - BFS, 2 - DFS итеративный",
        "3 - A* (Манхэттен), 4 - A* (Евклидов)",
        "SPACE - новый лабиринт, S - решение, ESC - выход"
    ]
    
    for i, instruction in enumerate(instructions):
        t = font.render(instruction, True, BLACK)
        screen.blit(t, (10, HEIGHT - 80 + i * 22))
    
    # Отображаем замеры времени, если есть
    small_font = pygame.font.SysFont(None, 18)
    y0 = HEIGHT - 180
    if gen_time is not None:
        t = small_font.render(f"Генерация лабиринта: {gen_time*1000:.2f} ms", True, BLACK)
        screen.blit(t, (WIDTH - 320, y0))
        y0 += 20
    if explore_time is not None:
        t = small_font.render(f"Поиск пути алгоритма в лабиринте: {explore_time*1000:.2f} ms", True, BLACK)
        screen.blit(t, (WIDTH - 320, y0))
        y0 += 20
    if solve_time is not None:
        t = small_font.render(f"Кратчайший путь в лабиринте: {solve_time*1000:.2f} ms", True, BLACK)
        screen.blit(t, (WIDTH - 320, y0))
        y0 += 20
    
    pygame.display.flip()

def main():
    # первые замеры пустые
    gen_time = None
    explore_time = None
    solve_time = None

    # Генерация лабиринта (с замером времени)
    t0 = time.perf_counter()
    grid = generate_maze()
    t1 = time.perf_counter()
    gen_time = t1 - t0
    print(f"Генерация лабиринта: {gen_time*1000:.2f} ms")
    
    start = grid[0][0]
    end = grid[ROWS-1][COLS-1]
    
    path = None
    running = True
    show_solution = False
    current_algorithm = "astar_manhattan"  # алгоритм по умолчанию
    algorithm_names = {
        "bfs": "BFS (поиск в ширину)",
        "dfs_iterative": "DFS итеративный",
        "dfs_recursive": "DFS рекурсивный",
        "astar_manhattan": "A* (Манхэттенское расстояние)",
        "astar_euclidean": "A* (Евклидово расстояние)"
    }
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Генерируем новый лабиринт (с замером)
                    t0 = time.perf_counter()
                    grid = generate_maze()
                    t1 = time.perf_counter()
                    gen_time = t1 - t0
                    print(f"Генерация лабиринта: {gen_time*1000:.2f} ms")
                    
                    start = grid[0][0]
                    end = grid[ROWS-1][COLS-1]
                    path = None
                    show_solution = False
                
                elif event.key == pygame.K_s:
                    # Показываем процесс поиска пути алгоритма и кратчайший путь в лабиринте
                    if not show_solution:
                        # Процесс поиска пути алгоритма в лабиринте (с замером времени)
                        explorer = get_algorithm_generator(grid, start, end, current_algorithm)
                        t0 = time.perf_counter()
                        ok = animate_exploration(grid, explorer)
                        t1 = time.perf_counter()
                        explore_time = t1 - t0
                        print(f"Поиск пути алгоритма в лабиринте: {explore_time*1000:.2f} ms")
                        
                        if ok:
                            # Находим и показываем кратчайший путь в лабиринте(замер только на восстановление/поиск полного пути)
                            t0 = time.perf_counter()
                            path = find_path_with_algorithm(grid, start, end, current_algorithm)
                            t1 = time.perf_counter()
                            solve_time = t1 - t0
                            print(f"Кратчайший путь в лабиринте : {solve_time*1000:.2f} ms")
                            
                            if path:
                                show_solution = animate_solution(grid, path)
                
                # Выбор алгоритма
                elif event.key == pygame.K_1:
                    current_algorithm = "bfs"
                    show_solution = False
                
                elif event.key == pygame.K_2:
                    current_algorithm = "dfs_iterative"
                    show_solution = False
                
                elif event.key == pygame.K_3:
                    current_algorithm = "astar_manhattan"
                    show_solution = False
                
                elif event.key == pygame.K_4:
                    current_algorithm = "astar_euclidean"
                    show_solution = False
                
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        draw_maze(grid, path if show_solution else None, 
                  algorithm_name=algorithm_names[current_algorithm],
                  gen_time=gen_time, explore_time=explore_time, solve_time=solve_time)
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()
