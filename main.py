import time
import heapq
from typing import List, Tuple, Set, Dict, Optional

class Tile:
    START = "S"
    GOAL = "G"
    EMPTY = "."
    WALL = "W"
    MONSTER = "M"
    POTION = "P"
    KEY = "K"
    DOOR = "D"


class Board:
    def __init__(self, grid: List[List[str]]):
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0
        self.start = self._find(Tile.START)
        self.goal = self._find(Tile.GOAL)
        self.keys = self._find_all(Tile.KEY)
        self.doors = self._find_all(Tile.DOOR)
        if self.start is None or self.goal is None:
            raise ValueError("Board must have exactly one Start (S) and one Goal (G).")
    def _find(self, symbol: str) -> Optional[Tuple[int, int]]:
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c] == symbol:
                    return (r, c)
        return None
    def _find_all(self, symbol: str) -> List[Tuple[int, int]]:
        positions = []
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c] == symbol:
                    positions.append((r, c))
        return positions
    def in_bounds(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width
    def tile_at(self, pos: Tuple[int, int]) -> str:
        r, c = pos
        return self.grid[r][c]
    def is_passable(self, pos: Tuple[int, int], has_key: bool) -> bool:
        t = self.tile_at(pos)
        if t == Tile.WALL:
            return False
        if t == Tile.DOOR and not has_key:
            return False
        return True
    def neighbors4(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        r, c = pos
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [p for p in candidates if self.in_bounds(p)]

class State:
     __slots__ = ("pos", "energy", "has_key")
    def __init__(self, pos: Tuple[int, int], energy: int, has_key: bool):
        self.pos = pos
        self.energy = energy
        self.has_key = has_key
    def __eq__(self, other):
        return isinstance(other, State) and (self.pos == other.pos) and (self.energy == other.energy) and (self.has_key == other.has_key)
    def __hash__(self):
        return hash((self.pos, self.energy, self.has_key))
    def __repr__(self):
        return f"State(pos={self.pos}, energy={self.energy}, has_key={self.has_key})"

class SearchProblem:
    def __init__(
        self,
        board: Board,
        initial_energy: int = 10,
        cost_move: int = 1,
        cost_monster: int = 3,
        cost_unlock: int = 2,
        potion_energy_gain: int = 5,
    ):
        self.board = board
        self.initial_energy = initial_energy
        self.cost_move = cost_move
        self.cost_monster = cost_monster
        self.cost_unlock = cost_unlock
        self.potion_energy_gain = potion_energy_gain
        self.initial_state = State(pos=board.start, energy=initial_energy, has_key=False)
    def is_goal(self, state: State) -> bool:
        return (state.pos == self.board.goal) and (state.energy >= 0)
    def successors(self, state: State) -> List[Tuple[State, int, str]]:
        succs = []
        for nbr in self.board.neighbors4(state.pos):
            tile = self.board.tile_at(nbr)
            if not self.board.is_passable(nbr, has_key=state.has_key):
                continue
            step_cost = self.cost_move
            energy_change = -self.cost_move
            if tile == Tile.MONSTER:
                step_cost += self.cost_monster
                energy_change -= self.cost_monster
            if tile == Tile.DOOR:
                step_cost += self.cost_unlock
                energy_change -= self.cost_unlock
            next_has_key = state.has_key or (tile == Tile.KEY)
            next_energy = state.energy + energy_change
            if tile == Tile.POTION:
                next_energy += self.potion_energy_gain  
            if next_energy < 0:
                continue
            action_label = f"Move to {nbr} ({tile})"
            succs.append((State(nbr, next_energy, next_has_key), step_cost, action_label))
        return succs

class SearchResult:
    def __init__(
        self,
        found: bool,
        path: List[Tuple[State, str]],
        total_cost: int,
        nodes_expanded: int,
        max_frontier_size: int,
        runtime_sec: float,
    ):
        self.found = found
        self.path = path
        self.total_cost = total_cost
        self.nodes_expanded = nodes_expanded
        self.max_frontier_size = max_frontier_size
        self.runtime_sec = runtime_sec
    def __repr__(self):
        return (
            f"SearchResult(found={self.found}, total_cost={self.total_cost}, "
            f"nodes_expanded={self.nodes_expanded}, max_frontier_size={self.max_frontier_size}, "
            f"runtime_sec={self.runtime_sec:.6f}, path_len={len(self.path)})"
        )
        
def reconstruct_path(came_from: Dict[State, Tuple[Optional[State], int, str]], goal_state: State) -> Tuple[List[Tuple[State, str]], int]:

    path: List[Tuple[State, str]] = []
    total_cost = 0
    cur = goal_state
    while cur in came_from and came_from[cur][0] is not None:
        parent, step_cost, action_label = came_from[cur]
        path.append((cur, action_label))
        total_cost += step_cost
        cur = parent
    path.reverse()
    return path, total_cost

def uniform_cost_search(problem: SearchProblem) -> SearchResult:

    start_time = time.perf_counter()

    start = problem.initial_state
    frontier = []
    tie = 0
    heapq.heappush(frontier, (0, tie, start))

    came_from: Dict[State, Tuple[Optional[State], int, str]] = {start: (None, 0, "Start")}
    g_cost: Dict[State, int] = {start: 0}

    explored: Set[State] = set()
    nodes_expanded = 0
    max_frontier = 1

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        cur_g, _, cur = heapq.heappop(frontier)
        if cur in explored:
            continue
        explored.add(cur)
        nodes_expanded += 1
        if problem.is_goal(cur):
            path, total_cost = reconstruct_path(came_from, cur)
            end_time = time.perf_counter()
            return SearchResult(
                found=True,
                path=path,
                total_cost=total_cost,
                nodes_expanded=nodes_expanded,
                max_frontier_size=max_frontier,
                runtime_sec=end_time - start_time,
            )
        for nxt, step_cost, action_label in problem.successors(cur):
            new_g = cur_g + step_cost
            if (nxt not in g_cost) or (new_g < g_cost[nxt]):
                g_cost[nxt] = new_g
                tie += 1
                heapq.heappush(frontier, (new_g, tie, nxt))
                came_from[nxt] = (cur, step_cost, action_label)
    end_time = time.perf_counter()
    return SearchResult(
        found=False,
        path=[],
        total_cost=float("inf"),
        nodes_expanded=nodes_expanded,
        max_frontier_size=max_frontier,
        runtime_sec=end_time - start_time,
    )


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
def admissible_heuristic(state: State, problem: SearchProblem) -> int:
    return manhattan(state.pos, problem.board.goal)

def a_star_search(problem: SearchProblem, heuristic_fn=admissible_heuristic) -> SearchResult:

    start_time = time.perf_counter()
    start = problem.initial_state
    frontier = []
    tie = 0
    start_h = heuristic_fn(start, problem)
    heapq.heappush(frontier, (start_h, tie, start))
    came_from: Dict[State, Tuple[Optional[State], int, str]] = {start: (None, 0, "Start")}
    g_cost: Dict[State, int] = {start: 0}
    explored: Set[State] = set()
    nodes_expanded = 0
    max_frontier = 1
    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        _, _, cur = heapq.heappop(frontier)
        if cur in explored:
            continue
        explored.add(cur)
        nodes_expanded += 1
        if problem.is_goal(cur):
            path, total_cost = reconstruct_path(came_from, cur)
            end_time = time.perf_counter()
            return SearchResult(
                found=True,
                path=path,
                total_cost=total_cost,
                nodes_expanded=nodes_expanded,
                max_frontier_size=max_frontier,
                runtime_sec=end_time - start_time,
            )
        for nxt, step_cost, action_label in problem.successors(cur):
            tentative_g = g_cost[cur] + step_cost
            if (nxt not in g_cost) or (tentative_g < g_cost[nxt]):
                g_cost[nxt] = tentative_g
                f_cost = tentative_g + heuristic_fn(nxt, problem)
                tie += 1
                heapq.heappush(frontier, (f_cost, tie, nxt))
                came_from[nxt] = (cur, step_cost, action_label)
    end_time = time.perf_counter()
    return SearchResult(
        found=False,
        path=[],
        total_cost=float("inf"),
        nodes_expanded=nodes_expanded,
        max_frontier_size=max_frontier,
        runtime_sec=end_time - start_time,
    )


def pretty_path(board: Board, result: SearchResult) -> str:
    if not result.found:
        return "No solution found."
    lines = []
    lines.append(f"Total cost: {result.total_cost}")
    lines.append(f"Path length (states): {len(result.path)}")
    lines.append("Sequence of actions:")
    for i, (st, action) in enumerate(result.path, start=1):
        lines.append(f"  {i:02d}. {action} -> pos={st.pos}, energy={st.energy}, has_key={st.has_key}")
    return "\n".join(lines)


def compare_algorithms(problem: SearchProblem):
    print("Running Uniform Cost Search (UCS)...")
    ucs_res = uniform_cost_search(problem)
    print(ucs_res)
    print(pretty_path(problem.board, ucs_res))
    print()
    print("Running A*...")
    astar_res = a_star_search(problem, heuristic_fn=admissible_heuristic)
    print(astar_res)
    print(pretty_path(problem.board, astar_res))
    print()
    print("Comparison Summary:")
    print(f"- UCS:   found={ucs_res.found}, total_cost={ucs_res.total_cost}, nodes_expanded={ucs_res.nodes_expanded}, "
          f"max_frontier={ucs_res.max_frontier_size}, runtime={ucs_res.runtime_sec:.6f}s")
    print(f"- A*:    found={astar_res.found}, total_cost={astar_res.total_cost}, nodes_expanded={astar_res.nodes_expanded}, "
          f"max_frontier={astar_res.max_frontier_size}, runtime={astar_res.runtime_sec:.6f}s")


def sample_board() -> Board:
    grid = [
        [Tile.START, Tile.EMPTY,  Tile.MONSTER, Tile.EMPTY,  Tile.POTION],
        [Tile.EMPTY, Tile.WALL,   Tile.EMPTY,   Tile.KEY,    Tile.EMPTY],
        [Tile.EMPTY, Tile.EMPTY,  Tile.MONSTER, Tile.EMPTY,  Tile.EMPTY],
        [Tile.POTION,Tile.EMPTY,  Tile.WALL,    Tile.EMPTY,  Tile.EMPTY],
        [Tile.EMPTY, Tile.EMPTY,  Tile.EMPTY,   Tile.DOOR,   Tile.GOAL],
    ]
    return Board(grid)

def main():
    board = sample_board()
    problem = SearchProblem(
        board=board,
        initial_energy=10,
        cost_move=1,
        cost_monster=3,
        cost_unlock=2,
        potion_energy_gain=5,
    )
    print("Board size:", board.height, "x", board.width)
    print("Start:", board.start, "Goal:", board.goal)
    print("Keys:", board.keys, "Doors:", board.doors)
    print("Initial energy:", problem.initial_energy)
    print()
    compare_algorithms(problem)
if __name__ == "__main__":
    main()

