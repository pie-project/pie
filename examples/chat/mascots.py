"""Small pixel animals for the chat banner: 10 columns by 8 rows each, no outlines.

Each entry is (palette, rows). '.' is empty; every other letter is a palette colour.
"""

ANIMALS = {
    "bear": ({"b": (139, 90, 43), "s": (100, 64, 30), "t": (214, 172, 120), "d": (30, 22, 18)}, [
        "bb......bb",
        "bbb....bbb",
        "bbbbbbbbbb",
        "bbdbbbbdbb",
        "bbbtttttbb",
        "bbbtddttbb",
        "bbbbttbbbb",
        "..ssssss..",
    ]),
    "frog": ({"g": (120, 192, 60), "l": (185, 228, 115), "s": (84, 150, 44), "w": (255, 255, 255), "d": (28, 88, 32)}, [
        "gwwg..gwwg",
        "gwdg..gdwg",
        "gggggggggg",
        "gggdddddgg",
        "gggggggggg",
        "llllllllll",
        "llllllllll",
        ".ssssssss.",
    ]),
    "cat": ({"o": (245, 140, 40), "s": (205, 100, 22), "p": (250, 182, 170), "w": (250, 236, 210), "d": (60, 36, 20), "n": (176, 90, 80)}, [
        "oo......oo",
        "opo....opo",
        "oooooooooo",
        "oodoooodoo",
        "ooooonoooo",
        "owwwwwwwwo",
        ".owwwwwwo.",
        "..ssssss..",
    ]),
    "fox": ({"o": (240, 120, 30), "s": (190, 86, 22), "c": (255, 236, 205), "d": (40, 26, 20), "n": (40, 26, 20)}, [
        "oo......oo",
        "ooo....ooo",
        "oooooooooo",
        "oodoooodoo",
        "occcnnccco",
        "occcccccco",
        ".occccccco",
        "..ssssss..",
    ]),
    "chick": ({"y": (250, 215, 50), "s": (222, 176, 28), "r": (220, 40, 40), "o": (245, 110, 30), "d": (40, 30, 26)}, [
        "....rr....",
        "..yyyyyy..",
        ".yyyyyyyy.",
        ".yydyydyy.",
        "yyyyooyyyy",
        ".yyyyyyyy.",
        "..ssssss..",
        "..o....o..",
    ]),
    "sheep": ({"w": (250, 250, 250), "s": (210, 210, 216), "g": (96, 100, 104), "d": (30, 30, 34)}, [
        ".wwwwwwww.",
        "wwwwwwwwww",
        "wwwwwwwwww",
        "wwggggggww",
        "wwgdggdgww",
        "wwggggggww",
        ".wwwwwwww.",
        "..ss..ss..",
    ]),
}
