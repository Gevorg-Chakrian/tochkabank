import sys
import heapq
import time


class GevorgChakrian:
    def __init__(self, room_depth=2, verbose=True):
        # Глубина комнат
        self.room_depth = room_depth
        # Позиции в коридоре, где находятся входы в комнаты
        self.room_entrances = [2, 4, 6, 8]
        # Соответствие типа объекта целевой комнате: A→0, B→1, C→2, D→3
        self.target_rooms = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        # Стоимость перемещения для каждого типа объекта
        self.move_cost = {'A': 1, 'B': 10, 'C': 100, 'D': 1000}
        # Флаг для отображения промежуточных шагов
        self.verbose = verbose
        # Счетчики для отслеживания процесса
        self.step_count = 0
        self.states_evaluated = 0
        self.max_memory_usage = 0

    def get_memory_usage(self):
        """Приблизительная оценка использования памяти в МБ без внешних библиотек"""
        # Оцениваем размер основных структур данных на основе количества обработанных состояний
        memory_estimate = 0

        # Каждое состояние в очереди занимает примерно 500 байт
        memory_estimate += self.states_evaluated * 500

        # Учитываем память для словаря состояний
        memory_estimate += len(str(self)) * 10

        # Переводим байты в мегабайты
        return memory_estimate / 1024 / 1024

    def update_max_memory(self):
        """Обновляем максимальное использование памяти для статистики"""
        current_memory = self.get_memory_usage()
        if current_memory > self.max_memory_usage:
            self.max_memory_usage = current_memory

    def heuristic(self, state):
        """
        Эвристическая функция для алгоритма A*
        Оценивает минимальную стоимость достижения цели из текущего состояния
        """
        hallway, rooms = state
        cost = 0

        # Оцениваем стоимость перемещения объектов из коридора в их целевые комнаты
        for pos, object_ in enumerate(hallway):
            if object_ != '.':
                target_room = self.target_rooms[object_]
                room_entrance = self.room_entrances[target_room]
                # Стоимость = расстояние до входа + 1 шаг внутрь комнаты
                cost += (abs(pos - room_entrance) + 1) * self.move_cost[object_]

        # Оцениваем стоимость перемещения объектов из неправильных комнат
        for room_idx, room in enumerate(rooms):
            for depth, object_ in enumerate(room):
                if object_ != '.':
                    target_room = self.target_rooms[object_]
                    if room_idx != target_room:
                        # Объект не в своей комнате - нужно его переместить
                        # Стоимость = выход из текущей комнаты + перемещение по коридору + вход в целевую
                        cost += (((depth + 1) + abs(
                            self.room_entrances[room_idx] - self.room_entrances[target_room]) + 1)
                                 * self.move_cost[object_])

        # Делим на 10 для более агрессивной эвристики (ускоряет поиск)
        return cost // 10

    def print_state(self, state, total_cost, step_cost, description=""):
        """Красиво отображает текущее состояние лабиринта с цветовой подсветкой"""
        if not self.verbose:
            return

        hallway, rooms = state

        # Шапка с информацией о шаге
        print(f"\n{'=' * 60}")
        if description:
            print(f"ШАГ {self.step_count}: {description}")
        if step_cost > 0:
            print(f"Энергия: {total_cost - step_cost} + {step_cost} = {total_cost}")
        else:
            print(f"Энергия: {total_cost}")
        print(f"{'=' * 60}")

        # Верхняя граница лабиринта
        print("#############")

        # Отображаем коридор с подсветкой позиций над комнатами
        hallway_str = "#"
        for i, cell in enumerate(hallway):
            if i in self.room_entrances:
                # Позиции над входами в комнаты - серым цветом
                hallway_str += f"\033[90m{cell}\033[0m"
            else:
                hallway_str += cell
        hallway_str += "#"
        print(hallway_str)

        # Отображаем комнаты
        for depth in range(self.room_depth):
            if depth == 0:
                line = "###"  # Верхний уровень комнат
            else:
                line = "  #"  # Нижние уровни

            for room_idx in range(4):
                cell = rooms[room_idx][depth]
                target_type = ['A', 'B', 'C', 'D'][room_idx]

                # Цветовая подсветка: зеленый - на своем месте, красный - не на своем
                if cell == target_type:
                    line += f"\033[92m{cell}\033[0m"  # Зеленый - правильно
                elif cell != '.':
                    line += f"\033[91m{cell}\033[0m"  # Красный - неправильно
                else:
                    line += cell  # Пустая клетка

                # Разделители между комнатами
                if room_idx < 3:
                    line += "#"
                else:
                    if depth == 0:
                        line += "###"  # Завершение верхней линии
                    else:
                        line += "#"  # Завершение нижних линий
            print(line)

        # Нижняя граница лабиринта (зависит от глубины комнат)
        if self.room_depth == 2:
            print("  #########")
        elif self.room_depth == 3:
            print("  #########")
            print("  #########")
        elif self.room_depth == 4:
            print("  #########")
            print("  #########")

        print(f"{'=' * 60}")

    def parse_input(self, lines):
        """Парсит входные данные и создает начальное состояние"""
        # Инициализируем коридор (11 позиций, все пустые)
        hallway = ['.'] * 11
        # Инициализируем 4 комнаты
        rooms = [[] for _ in range(4)]

        # Обрабатываем каждую строку с объектами
        for i in range(self.room_depth):
            line = lines[2 + i].strip()
            # Очищаем строку от символов стен
            if line.startswith('###'):
                line = line[3:-3]  # Убираем начальные и конечные ###
            elif line.startswith('  #'):
                line = line[3:-1]  # Убираем начальные пробелы и #

            # Извлекаем только символы объектов и пустых клеток
            room_chars = []
            for char in line:
                if char in 'ABCD.':
                    room_chars.append(char)

            # Распределяем объекты по комнатам
            for j in range(4):
                if j < len(room_chars):
                    rooms[j].append(room_chars[j] if room_chars[j] != '.' else '.')
                else:
                    rooms[j].append('.')

        return (tuple(hallway), tuple(tuple(room) for room in rooms))

    def is_goal_state(self, state):
        """Проверяет, является ли состояние целевым"""
        hallway, rooms = state

        # В целевом состоянии коридор должен быть пустым
        if any(cell != '.' for cell in hallway):
            return False

        # Каждая комната должна содержать только свои объекты
        for room_idx, room in enumerate(rooms):
            target_type = ['A', 'B', 'C', 'D'][room_idx]
            if any(amp != target_type for amp in room if amp != '.'):
                return False
        return True

    def can_move_to_room(self, object_, room_idx, rooms):
        """Проверяет, может ли объект переместиться в указанную комнату"""
        target_room = self.target_rooms[object_]
        # Объект может перемещаться только в свою целевую комнату
        if room_idx != target_room:
            return None

        room = rooms[room_idx]
        # В комнате не должно быть чужих объектов
        for amp in room:
            if amp != '.' and self.target_rooms[amp] != room_idx:
                return None

        # Ищем самую глубокую свободную позицию в комнате
        for i in range(self.room_depth - 1, -1, -1):
            if room[i] == '.':
                return i
        return None

    def is_hallway_clear(self, hallway, start, end):
        """Проверяет, свободен ли путь в коридоре между двумя позициями"""
        if start == end:
            return True

        # Определяем направление движения
        step = 1 if end > start else -1
        # Проверяем все промежуточные позиции
        for pos in range(start + step, end + step, step):
            if hallway[pos] != '.':
                return False
        return True

    def generate_moves(self, state):
        """Генерирует все возможные ходы из текущего состояния"""
        hallway, rooms = state
        room_moves = []  # Ходы прямо в комнаты (высший приоритет)
        hallway_moves = []  # Ходы в коридор (низший приоритет)

        # ПРИОРИТЕТ 1: Ходы из коридора прямо в целевые комнаты
        for hall_pos in range(11):
            if hallway[hall_pos] != '.':
                object_ = hallway[hall_pos]
                target_room = self.target_rooms[object_]
                room_entrance = self.room_entrances[target_room]
                depth_pos = self.can_move_to_room(object_, target_room, rooms)
                # Проверяем возможность перемещения
                if depth_pos is not None and self.is_hallway_clear(hallway, hall_pos, room_entrance):
                    steps = abs(hall_pos - room_entrance) + (depth_pos + 1)
                    cost = steps * self.move_cost[object_]
                    # Создаем новое состояние
                    new_hallway = list(hallway)
                    new_hallway[hall_pos] = '.'
                    new_rooms = [list(room) for room in rooms]
                    new_rooms[target_room][depth_pos] = object_

                    room_names = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
                    move_desc = (f"{object_} из коридора позиция {hall_pos} → "
                                 f"комната {room_names[target_room]} ({steps} шагов × "
                                 f"{self.move_cost[object_]} = {cost})")
                    room_moves.append(
                        (cost, (tuple(new_hallway), tuple(tuple(room) for room in new_rooms)), move_desc, 1))

        # Если есть ходы прямо в комнаты - возвращаем только их
        if room_moves:
            return room_moves

        # ПРИОРИТЕТ 2: Ходы между комнатами (прямое перемещение)
        for room_idx in range(4):
            room = rooms[room_idx]
            for depth in range(self.room_depth):
                if room[depth] != '.':
                    object_ = room[depth]
                    target_room = self.target_rooms[object_]
                    # Объект не в своей целевой комнате
                    if room_idx != target_room:
                        room_entrance = self.room_entrances[room_idx]
                        target_entrance = self.room_entrances[target_room]
                        depth_pos = self.can_move_to_room(object_, target_room, rooms)
                        # Проверяем возможность прямого перемещения между комнатами
                        if (depth_pos is not None and
                                self.is_hallway_clear(hallway, room_entrance, target_entrance) and
                                all(rooms[room_idx][d] == '.' for d in range(depth))):  # Ничего не блокирует сверху
                            steps = abs(room_entrance - target_entrance) + (depth + 1) + (depth_pos + 1)
                            cost = steps * self.move_cost[object_]
                            new_hallway = list(hallway)
                            new_rooms = [list(room) for room in rooms]
                            new_rooms[room_idx][depth] = '.'
                            new_rooms[target_room][depth_pos] = object_

                            room_names = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
                            move_desc = (f"{object_} из комнаты {room_names[room_idx]} → "
                                         f"комната {room_names[target_room]} ({steps} шагов × "
                                         f"{self.move_cost[object_]} = {cost})")
                            room_moves.append(
                                (cost, (tuple(new_hallway), tuple(tuple(room) for room in new_rooms)), move_desc, 1))

        if room_moves:
            return room_moves

        # ПРИОРИТЕТ 3: Ходы из комнат в коридор (когда прямые ходы невозможны)
        for room_idx in range(4):
            room = rooms[room_idx]
            move_depth = None
            # Находим верхний объект, который может/должен двигаться
            for depth in range(self.room_depth):
                if room[depth] != '.':
                    object_ = room[depth]
                    should_move = False
                    # Объект должен двигаться если он не в своей комнате
                    if self.target_rooms[object_] != room_idx:
                        should_move = True
                    else:
                        # Или если он блокирует чужой объект под собой
                        for d in range(depth + 1, self.room_depth):
                            if room[d] != '.' and self.target_rooms[room[d]] != room_idx:
                                should_move = True
                                break
                    if should_move:
                        move_depth = depth
                        break

            if move_depth is not None:
                object_ = room[move_depth]
                room_entrance = self.room_entrances[room_idx]
                target_room = self.target_rooms[object_]
                target_entrance = self.room_entrances[target_room]

                # Выбираем стратегические позиции в коридоре
                strategic_positions = []
                for hall_pos in range(11):
                    # Пропускаем позиции над входами в комнаты
                    if hall_pos in self.room_entrances:
                        continue
                    if self.is_hallway_clear(hallway, room_entrance, hall_pos):
                        # Приоритетные позиции - между текущей и целевой комнатой
                        if (room_entrance < target_entrance and room_entrance < hall_pos < target_entrance) or \
                                (room_entrance > target_entrance and target_entrance < hall_pos < room_entrance):
                            strategic_positions.append((hall_pos, 1))  # Высокий приоритет
                        else:
                            strategic_positions.append((hall_pos, 3))  # Низкий приоритет

                # Сортируем по приоритету и близости к целевой комнате
                strategic_positions.sort(key=lambda x: (x[1], abs(x[0] - target_entrance)))

                for hall_pos, priority in strategic_positions:
                    steps = abs(hall_pos - room_entrance) + (move_depth + 1)
                    cost = steps * self.move_cost[object_]
                    new_hallway = list(hallway)
                    new_hallway[hall_pos] = object_
                    new_rooms = [list(room) for room in rooms]
                    new_rooms[room_idx][move_depth] = '.'

                    room_names = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
                    depth_names = {0: "глубина 1", 1: "глубина 2", 2: "глубина 3", 3: "глубина 4"}
                    move_desc = (f"{object_} из комнаты {room_names[room_idx]} {depth_names[move_depth]} → "
                                 f"коридор позиция {hall_pos} ({steps} шагов × {self.move_cost[object_]} = {cost})")
                    hallway_moves.append(
                        (cost, (tuple(new_hallway), tuple(tuple(room) for room in new_rooms)), move_desc, priority))

        # Возвращаем ходы отсортированные по приоритету
        return sorted(hallway_moves, key=lambda x: x[3])

    def solve(self, initial_state):
        start_time = time.time()
        # Очередь с приоритетом для A*: (оценка_стоимости, реальная_стоимость, состояние, путь)
        queue = []
        initial_heuristic = self.heuristic(initial_state)
        heapq.heappush(queue, (initial_heuristic, 0, initial_state, []))

        # Словарь для хранения минимальной стоимости достижения каждого состояния
        min_cost = {initial_state: (0, [])}
        self.states_evaluated = 0
        self.max_memory_usage = self.get_memory_usage()

        if self.verbose:
            print("НАЧАЛЬНОЕ СОСТОЯНИЕ:")
            self.step_count = 0
            self.print_state(initial_state, 0, 0, "Начальное состояние")

        while queue:
            self.states_evaluated += 1
            self.update_max_memory()

            # Извлекаем состояние с наименьшей оценкой стоимости
            estimated_cost, current_cost, current_state, path = heapq.heappop(queue)

            # Пропускаем если нашли более дешевый путь к этому состоянию
            if current_cost > min_cost.get(current_state, (float('inf'), []))[0]:
                continue

            # Проверяем достигли ли целевого состояния
            if self.is_goal_state(current_state):
                end_time = time.time()
                execution_time = end_time - start_time

                if self.verbose:
                    print(f"Итоговая энергия: {current_cost}")
                    print(f"Всего шагов: {len(path)}")

                    print("\nПОСЛЕДОВАТЕЛЬНОСТЬ ХОДОВ:")
                    total_cost = 0
                    # Выводим всю последовательность ходов
                    for i, (step_desc, step_state, step_cost) in enumerate(path):
                        total_cost += step_cost
                        self.step_count = i + 1
                        self.print_state(step_state, total_cost, step_cost, step_desc)

                # Вывод статистики производительности
                print(f"\nВремя выполнения: {execution_time:.2f} секунд")
                print(f"Потребление памяти: {self.max_memory_usage:.1f} МБ")

                return current_cost

            # Генерируем и обрабатываем все возможные ходы
            for move_cost, next_state, move_desc, priority in self.generate_moves(current_state):
                new_cost = current_cost + move_cost
                new_path = path + [(move_desc, next_state, move_cost)]
                # A* оценка: реальная стоимость + эвристическая оценка
                new_heuristic = new_cost + self.heuristic(next_state)

                # Если нашли более дешевый путь к состоянию
                if new_cost < min_cost.get(next_state, (float('inf'), []))[0]:
                    min_cost[next_state] = (new_cost, new_path)
                    priority_bonus = 0 if priority == 1 else 1000
                    heapq.heappush(queue, (new_heuristic + priority_bonus, new_cost, next_state, new_path))

        return -1  # Решение не найдено


def solve(lines: list[str]) -> int:
    """Основная функция решения задачи"""
    # Определяем глубину комнат по количеству строк входных данных
    room_depth = len(lines) - 3
    verbose = True  # Показать шаги решения
    solver = GevorgChakrian(room_depth, verbose=verbose)
    # Парсим входные данные в начальное состояние
    initial_state = solver.parse_input(lines)
    # Запускаем алгоритм решения
    return solver.solve(initial_state)


def main():
    """Чтение входных данных и запуск решения"""
    lines = []
    for line in sys.stdin:
        lines.append(line.rstrip('\n'))

    result = solve(lines)
    print(f"Результат: {result}")


if __name__ == "__main__":
    main()