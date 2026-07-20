import csv

words = [
    {"written": "שלך", "m": "shelcha", "f": "shelach", "meaning": "yours"},
    {"written": "לך", "m": "lecha", "f": "lach", "meaning": "to you"},
    {"written": "אותך", "m": "otcha", "f": "otach", "meaning": "you (object)"},
    {"written": "עליך", "m": "aleicha", "f": "alaich", "meaning": "on/about you"},
    {"written": "אליך", "m": "eleicha", "f": "elaich", "meaning": "to you"},
    {"written": "איתך", "m": "itcha", "f": "itach", "meaning": "with you"},
    {"written": "ממך", "m": "mimcha", "f": "mimech", "meaning": "from you"},
    {"written": "בך", "m": "becha", "f": "bach", "meaning": "in you"},
    {"written": "בשבילך", "m": "bishvilcha", "f": "bishvilech", "meaning": "for you"},
    {"written": "אצלך", "m": "etzlecha", "f": "etzlech", "meaning": "at your place"},
    {"written": "בעצמך", "m": "be'atzmecha", "f": "be'atzmech", "meaning": "yourself"},
    {"written": "כמוך", "m": "kamocha", "f": "kamoch", "meaning": "like you"},
    {"written": "לידיעתך", "m": "leyedi'at'cha", "f": "leyedi'atech", "meaning": "for your info"},
    {"written": "ידיעתך", "m": "yedi'at'cha", "f": "yedi'atech", "meaning": "your info"},
    {"written": "דעתך", "m": "da'at'cha", "f": "da'atech", "meaning": "your opinion"},
    {"written": "שמך", "m": "shimcha", "f": "shmech", "meaning": "your name"},
    {"written": "רצונך", "m": "retzoncha", "f": "retzonech", "meaning": "your desire"},
    {"written": "שאלתך", "m": "she'elatcha", "f": "she'elatech", "meaning": "your question"},
    {"written": "כוונתך", "m": "kavanat'cha", "f": "kavanatech", "meaning": "your intention"},
    {"written": "תשובתך", "m": "tshuvatcha", "f": "tshuvatech", "meaning": "your answer"},
    {"written": "זכותך", "m": "zchutcha", "f": "zchutech", "meaning": "your right"},
    {"written": "אמרת", "m": "amarta", "f": "amart", "meaning": "you said"},
    {"written": "עשית", "m": "asita", "f": "asit", "meaning": "you did"},
    {"written": "הלכת", "m": "halachta", "f": "halacht", "meaning": "you walked"},
    {"written": "רצית", "m": "ratzita", "f": "ratzit", "meaning": "you wanted"},
    {"written": "חשבת", "m": "chashavta", "f": "chashavt", "meaning": "you thought"},
    {"written": "הבנת", "m": "hevanta", "f": "hevant", "meaning": "you understood"},
    {"written": "ידעת", "m": "yadata", "f": "yadat", "meaning": "you knew"},
    {"written": "ראית", "m": "ra'ita", "f": "ra'it", "meaning": "you saw"},
    {"written": "שמעת", "m": "shamata", "f": "shamat", "meaning": "you heard"},
    {"written": "כתבת", "m": "katavta", "f": "katavt", "meaning": "you wrote"}
]

data = []
for w in words:
    data.append({"type": "hebrew_gender_m", "written": f"{w['written']} (Read as Masculine)", "spoken": w["m"]})
    data.append({"type": "hebrew_gender_f", "written": f"{w['written']} (Read as Feminine)", "spoken": w["f"]})

with open("hebrew_all_prompts.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["type", "written", "spoken"])
    writer.writeheader()
    writer.writerows(data)
print("Done")
