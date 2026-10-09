"""Small pixel animals for the chat banner: 10 columns by 8 rows each, drawn at that size.

Each entry is (palette, rows). '.' is empty; every other letter is a palette colour.
"""

K = (24, 24, 28)  # outline

ANIMALS = {
    "bear": ({"K": K, "b": (139, 90, 43), "t": (214, 172, 120), "d": (24, 24, 28)}, [
        "KKK....KKK",
        "KbbK..KbbK",
        "KbbbbbbbbK",
        "KbdbbbbdbK",
        "KbbtttttbK",
        "KbbtKKtbbK",
        ".KbbbbbbK.",
        "..KKKKKK..",
    ]),
    "frog": ({"K": K, "g": (120, 192, 60), "l": (185, 228, 115), "w": (255, 255, 255), "d": (28, 88, 32)}, [
        "KwwK..KwwK",
        "KwdK..KdwK",
        "KggggggggK",
        "KgggKKgggK",
        "KggggggggK",
        "KllllllllK",
        "KllllllllK",
        ".KKKKKKKK.",
    ]),
    "cat": ({"K": K, "o": (245, 140, 40), "p": (250, 182, 170), "d": (205, 100, 22), "w": (250, 236, 210), "n": (176, 90, 80)}, [
        "KKK....KKK",
        "KpoK..KopK",
        "KooooooooK",
        "KodoooodoK",
        "KoooonoooK",
        "KowwwwwwoK",
        ".KoooooooK",
        "..KKKKKKK.",
    ]),
    "fox": ({"K": K, "o": (240, 120, 30), "c": (255, 236, 205), "d": (24, 24, 28), "n": (24, 24, 28)}, [
        "KKK....KKK",
        "KoK....KoK",
        "KooooooooK",
        "KodoooodoK",
        "KcccnnccoK",
        "KccccccccK",
        ".KccccccK.",
        "..KKKKKK..",
    ]),
    "chick": ({"K": K, "y": (250, 215, 50), "r": (220, 40, 40), "o": (245, 110, 30), "d": (40, 30, 26)}, [
        "...rr.....",
        "..KKKK....",
        ".KyyyyK...",
        "KyydyyyK..",
        "KyyyoyyyK.",
        ".KyyyyyyK.",
        "..KyyyyK..",
        "..K.oo.K..",
    ]),
    "sheep": ({"K": K, "w": (250, 250, 250), "g": (96, 100, 104), "d": (30, 30, 34)}, [
        ".KKKKKKKK.",
        "KwwwwwwwwK",
        "KwwwwwwwwK",
        "KwgggggggK",
        "KwgdggdgwK",
        "KwggggggwK",
        ".KwwwwwwK.",
        "..KKKKKK..",
    ]),
}
