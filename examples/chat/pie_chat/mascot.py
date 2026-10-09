BODY = {2: "▐▟▙▙▙▛▌", 3: "▜██▄██▛"}
LOOP = 8
SPAWNS = [(0, 8), (3, 8), (6, 8), (2, 9), (5, 9)]
BODY_STYLE = "class:accent"
BUBBLE_STYLE = "fg:#3a2a26"
WIDTH = 10
ROWS = 4


def bubble_at(age: int) -> tuple[int, str] | None:
    if not 1 <= age <= 7:
        return None
    return 3 - age // 2, ("▖" if age % 2 == 0 else "▘")


def bubbles_for(frame: int) -> list[tuple[int, int, str]]:
    bubbles = []
    for spawn_frame, column in SPAWNS:
        placed = bubble_at((frame - spawn_frame) % LOOP)
        if placed is not None:
            row, char = placed
            bubbles.append((row, column, char))
    return bubbles


def frame_grid(frame: int) -> list[list[tuple[str, str]]]:
    grid = [[("", " ")] * WIDTH for _ in range(ROWS)]
    for row, text in BODY.items():
        for col, ch in enumerate(text):
            grid[row][col] = (BODY_STYLE, ch)
    for row, col, ch in bubbles_for(frame):
        grid[row][col] = (BUBBLE_STYLE, ch)
    return grid


FRAMES = [frame_grid(frame) for frame in range(LOOP)]


def mascot_rows(frame: int) -> list[list[tuple[str, str]]]:
    return [list(row) for row in FRAMES[frame % LOOP]]
