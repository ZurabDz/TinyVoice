import itertools

CHARSET = ' აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ'


def to_text(x):
    x = [k for k, g in itertools.groupby(x)]
    return ''.join([CHARSET[c-1] for c in x if c != 0])


def from_text(x):
    return [CHARSET.index(c)+1 for c in x.lower() if c in CHARSET]
