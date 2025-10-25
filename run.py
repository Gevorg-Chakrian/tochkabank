import sys
import heapq
import time


class GevorgChakrian:
    def __init__(self, room_depth=2, verbose=True):
        self.room_depth = room_depth
        self.room_entrances = [2, 4, 6, 8]
        self.target_rooms = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        self.move_cost = {'A': 1, 'B': 10, 'C': 100, 'D': 1000}
        self.verbose = verbose
        self.step_count = 0
        self.states_evaluated = 0
        self.max_memory_usage = 0

    def get_memory_usage(self):
        """Приблизительная оценка использования памяти в МБ"""
        # Оцениваем размер основных структур данных
        memory_estimate = 0

        # Очередь состояний (примерная оценка)
        memory_estimate += self.states_evaluated * 500  # байт на состояние

        # Словарь min_cost
        memory_estimate += len(str(self)) * 10  # грубая оценка

        # Переводим в МБ
        return memory_estimate / 1024 / 1024

    def update_max_memory(self):
        """Обновить максимальное использование памяти"""
        current_memory = self.get_memory_usage()
        if current_memory > self.max_memory_usage:
            self.max_memory_usage = current_memory

    def heuristic(self, state):
        """считаем минимальную стоимость до цели"""
        hallway, rooms = state
        cost = 0

        for pos, object_ in enumerate(hallway):
            if object_ != '.':
                target_room = self.target_rooms[object_]
                room_entrance = self.room_entrances[target_room]
                cost += (abs(pos - room_entrance) + 1) * self.move_cost[object_]

        for room_idx, room in enumerate(rooms):
            for depth, object_ in enumerate(room):
                if object_ != '.':
                    target_room = self.target_rooms[object_]
                    if room_idx != target_room:
                        cost += (((depth + 1) + abs(
                            self.room_entrances[room_idx] - self.room_entrances[target_room]) + 1)
                                 * self.move_cost[object_])

        return cost // 10

    def print_state(self, state, total_cost, step_cost, description=""):
        if not self.verbose:
            return

        hallway, rooms = state

        print(f"\n{'=' * 60}")
        if description:
            print(f"ШАГ {self.step_count}: {description}")
        if step_cost > 0:
            print(f"Энергия: {total_cost - step_cost} + {step_cost} = {total_cost}")
        else:
            print(f"Энергия: {total_cost}")
        print(f"{'=' * 60}")

        print("#############")

        hallway_str = "#"
        for i, cell in enumerate(hallway):
            if i in self.room_entrances:
                hallway_str += f"\033[90m{cell}\033[0m"
            else:
                hallway_str += cell
        hallway_str += "#"
        print(hallway_str)

        for depth in range(self.room_depth):
            if depth == 0:
                line = "###"
            else:
                line = "  #"

            for room_idx in range(4):
                cell = rooms[room_idx][depth]
                target_type = ['A', 'B', 'C', 'D'][room_idx]

                if cell == target_type:
                    line += f"\033[92m{cell}\033[0m"
                elif cell != '.':
                    line += f"\033[91m{cell}\033[0m"
                else:
                    line += cell

                if room_idx < 3:
                    line += "#"
                else:
                    if depth == 0:
                        line += "###"
                    else:
                        line += "#"
            print(line)

        # Для глубины 5 и более нужно больше строк для нижней границы
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
        hallway = ['.'] * 11
        rooms = [[] for _ in range(4)]

        for i in range(self.room_depth):
            line = lines[2 + i].strip()
            if line.startswith('###'):
                line = line[3:-3]  # Убираем начальные и конечные ###
            elif line.startswith('  #'):
                line = line[3:-1]  # Убираем начальные пробелы и #

            room_chars = []
            for char in line:
                if char in 'ABCD.':
                    room_chars.append(char)

            for j in range(4):
                if j < len(room_chars):
                    rooms[j].append(room_chars[j] if room_chars[j] != '.' else '.')
                else:
                    rooms[j].append('.')

        return (tuple(hallway), tuple(tuple(room) for room in rooms))

    def is_goal_state(self, state):
        hallway, rooms = state
        if any(cell != '.' for cell in hallway):
            return False
        for room_idx, room in enumerate(rooms):
            target_type = ['A', 'B', 'C', 'D'][room_idx]
            if any(amp != target_type for amp in room if amp != '.'):
                return False
        return True

    def can_move_to_room(self, amphipod, room_idx, rooms):
        target_room = self.target_rooms[amphipod]
        if room_idx != target_room:
            return None
        room = rooms[room_idx]
        for amp in room:
            if amp != '.' and self.target_rooms[amp] != room_idx:
                return None
        for i in range(self.room_depth - 1, -1, -1):
            if room[i] == '.':
                return i
        return None

    def is_hallway_clear(self, hallway, start, end):
        if start == end:
            return True
        step = 1 if end > start else -1
        for pos in range(start + step, end + step, step):
            if hallway[pos] != '.':
                return False
        return True

    def generate_moves(self, state):
        hallway, rooms = state
        room_moves = []
        hallway_moves = []

        # Ходы из коридора прямо в свои комнаты
        for hall_pos in range(11):
            if hallway[hall_pos] != '.':
                object_ = hallway[hall_pos]
                target_room = self.target_rooms[object_]
                room_entrance = self.room_entrances[target_room]
                depth_pos = self.can_move_to_room(object_, target_room, rooms)
                if depth_pos is not None and self.is_hallway_clear(hallway, hall_pos, room_entrance):
                    steps = abs(hall_pos - room_entrance) + (depth_pos + 1)
                    cost = steps * self.move_cost[object_]
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

        if room_moves:
            return room_moves

        # Ходы из комнат прямо в свои комнаты
        for room_idx in range(4):
            room = rooms[room_idx]
            for depth in range(self.room_depth):
                if room[depth] != '.':
                    object_ = room[depth]
                    target_room = self.target_rooms[object_]
                    if room_idx != target_room:
                        room_entrance = self.room_entrances[room_idx]
                        target_entrance = self.room_entrances[target_room]
                        depth_pos = self.can_move_to_room(object_, target_room, rooms)
                        if (depth_pos is not None and
                                self.is_hallway_clear(hallway, room_entrance, target_entrance) and
                                all(rooms[room_idx][d] == '.' for d in range(depth))):
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

        # Ходы из комнат в коридор
        for room_idx in range(4):
            room = rooms[room_idx]
            move_depth = None
            for depth in range(self.room_depth):
                if room[depth] != '.':
                    object_ = room[depth]
                    should_move = False
                    if self.target_rooms[object_] != room_idx:
                        should_move = True
                    else:
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

                strategic_positions = []
                for hall_pos in range(11):
                    if hall_pos in self.room_entrances:
                        continue
                    if self.is_hallway_clear(hallway, room_entrance, hall_pos):
                        if (room_entrance < target_entrance and room_entrance < hall_pos < target_entrance) or \
                                (room_entrance > target_entrance and target_entrance < hall_pos < room_entrance):
                            strategic_positions.append((hall_pos, 1))
                        else:
                            strategic_positions.append((hall_pos, 3))

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

        return sorted(hallway_moves, key=lambda x: x[3])

    def solve(self, initial_state):
        start_time = time.time()
        queue = []
        initial_heuristic = self.heuristic(initial_state)
        heapq.heappush(queue, (initial_heuristic, 0, initial_state, []))

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

            estimated_cost, current_cost, current_state, path = heapq.heappop(queue)

            if current_cost > min_cost.get(current_state, (float('inf'), []))[0]:
                continue

            if self.is_goal_state(current_state):
                end_time = time.time()
                execution_time = end_time - start_time

                if self.verbose:
                    print(f"Итоговая энергия: {current_cost}")
                    print(f"Всего шагов: {len(path)}")

                    print("\nПОСЛЕДОВАТЕЛЬНОСТЬ ХОДОВ:")
                    total_cost = 0
                    for i, (step_desc, step_state, step_cost) in enumerate(path):
                        total_cost += step_cost
                        self.step_count = i + 1
                        self.print_state(step_state, total_cost, step_cost, step_desc)

                # Вывод времени выполнения и потребления памяти
                print(f"\nВремя выполнения: {execution_time:.2f} секунд")
                print(f"Потребление памяти: {self.max_memory_usage:.1f} МБ")

                return current_cost

            for move_cost, next_state, move_desc, priority in self.generate_moves(current_state):
                new_cost = current_cost + move_cost
                new_path = path + [(move_desc, next_state, move_cost)]
                new_heuristic = new_cost + self.heuristic(next_state)

                if new_cost < min_cost.get(next_state, (float('inf'), []))[0]:
                    min_cost[next_state] = (new_cost, new_path)
                    priority_bonus = 0 if priority == 1 else 1000
                    heapq.heappush(queue, (new_heuristic + priority_bonus, new_cost, next_state, new_path))

        return -1


def solve(lines: list[str]) -> int:
    room_depth = len(lines) - 3
    verbose = True  # Показать шаги
    solver = GevorgChakrian(room_depth, verbose=verbose)
    initial_state = solver.parse_input(lines)
    return solver.solve(initial_state)


def main():
    lines = []
    for line in sys.stdin:
        lines.append(line.rstrip('\n'))

    result = solve(lines)
    print(f"Результат: {result}")


if __name__ == "__main__":
    main()