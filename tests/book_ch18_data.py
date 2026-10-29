"""Данные главы 18.

- DEMETHYLATION — Table 18.1 (стр. 481): (доза, нормальные растения, всего в лотке), 24 лотка.
- LADYBIRD — Table 18.5 (стр. 491): (ряд, чашка, пол, жертва, съедено из 6), 60 чашек.
- APHIDS — Table 18.10 (стр. 504): (поле, проба, число тлей), 45 проб.
- CONIDIA — Table 18.12 (стр. 509): (источник, прогон, время мин, конидии), без времени 0.
"""

_DOSES = (0, 0.01, 0.1, 0.5, 1.0, 1.5)
_NORMAL = ((59, 58, 54, 4, 3, 3), (58, 59, 53, 11, 2, 3),
           (99, 98, 88, 14, 2, 1), (98, 99, 87, 15, 1, 3))
_TOTAL = (60, 60, 100, 100)
DEMETHYLATION = [(d, _NORMAL[r][j], _TOTAL[r]) for r in range(4) for j, d in enumerate(_DOSES)]

_LADYBIRD = """1 1 F A 5|1 2 M A 2|1 3 F L 3|1 4 M L 0|2 1 F A 5|2 2 M A 2|2 3 F L 1|2 4 M L 1|
3 1 F A 3|3 2 F L 0|3 3 M A 0|3 4 M L 0|4 1 M L 1|4 2 M A 1|4 3 F A 4|4 4 F L 2|
5 1 M A 2|5 2 F A 1|5 3 M L 0|5 4 F L 4|6 1 M L 1|6 2 F L 4|6 3 F A 0|6 4 M A 0|
7 1 M A 0|7 2 M L 2|7 3 F L 2|7 4 F A 4|8 1 M A 3|8 2 M L 2|8 3 F L 5|8 4 F A 3|
9 1 F L 1|9 2 M L 0|9 3 F A 0|9 4 M A 1|10 1 F A 4|10 2 M A 0|10 3 F L 0|10 4 M L 0|
11 1 M L 0|11 2 M A 0|11 3 F L 2|11 4 F A 2|12 1 M A 2|12 2 F L 0|12 3 M L 1|12 4 F A 4|
13 1 M A 0|13 2 M L 0|13 3 F A 2|13 4 F L 0|14 1 M L 2|14 2 F L 3|14 3 F A 2|14 4 M A 1|
15 1 F A 2|15 2 F L 1|15 3 M A 0|15 4 M L 0"""
LADYBIRD = [(r, d, s, p, int(e)) for r, d, s, p, e in
            (row.split() for row in _LADYBIRD.replace("\n", "").split("|"))]

_APHIDS = ((0, 3, 5, 15, 7, 5, 2, 5, 4, 6, 1, 5, 1, 9, 1),
           (24, 10, 21, 28, 43, 11, 14, 22, 8, 7, 1, 20, 6, 10, 6),
           (3, 2, 10, 14, 4, 11, 6, 2, 3, 5, 3, 13, 5, 4, 15))
APHIDS = [(f + 1, k + 1, c) for f, counts in enumerate(_APHIDS) for k, c in enumerate(counts)]

_TIMES = (5, 10, 15, 20, 25, 40, 80)
_CONIDIA = {("Standard", 1): (6, 71, 157, 568, 883, 1436, 3543),
            ("Standard", 2): (71, 223, 426, 1391, 1098, 993, 4295),
            ("Clone", 1): (8, 173, 165, 584, 1296, 400, 4981),
            ("Clone", 2): (44, 209, 383, 1188, 627, 1628, 4302)}
CONIDIA = [(s, r, t, c) for (s, r), counts in _CONIDIA.items() for t, c in zip(_TIMES, counts)]
