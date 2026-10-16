from normalize_text import normalize_text


test_cases = [
    "1d10t",
    "i.d.i.o.t",
    "i d i o t",
    "stuuupid",
    "you are an 1d10t",
    "you are s.t.u.p.i.d",
    "you are s t u p i d",
    "you are stuuupid",
]


for text in test_cases:

    normalized = normalize_text(text)

    print(
        f"{text:30} -> {normalized}"
    )