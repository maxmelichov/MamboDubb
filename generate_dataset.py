import random
import datetime
from num2words import num2words
import csv

def generate_number():
    num = random.randint(0, 1000000)
    # num2words gives e.g. "one hundred and twenty-three"
    text = num2words(num)
    return str(num), text

def generate_date():
    start_date = datetime.date(1900, 1, 1)
    end_date = datetime.date(2050, 12, 31)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + datetime.timedelta(days=random_number_of_days)
    
    # formats
    formats = [
        ("%Y-%m-%d", "{month} {day}, {year}"),
        ("%B %d, %Y", "{month} {day}, {year}"),
        ("%d/%m/%Y", "{month} {day}, {year}"),
        ("%m/%d/%Y", "{month} {day}, {year}")
    ]
    fmt, text_fmt = random.choice(formats)
    
    month_name = random_date.strftime("%B")
    day_num = random_date.day
    year_num = random_date.year
    
    day_word = num2words(day_num, to="ordinal")
    
    # Year logic
    if 2000 <= year_num <= 2009:
        year_word = "two thousand" if year_num == 2000 else f"two thousand and {num2words(year_num % 2000)}"
    else:
        century = year_num // 100
        decade = year_num % 100
        if decade == 0:
            year_word = f"{num2words(century)} hundred"
        elif decade < 10:
            year_word = f"{num2words(century)} oh {num2words(decade)}"
        else:
            year_word = f"{num2words(century)} {num2words(decade)}"
            
    written = f"{random_date.strftime(fmt)}"
    spoken = f"{month_name} {day_word}, {year_word}"
    
    return written, spoken

def generate_time():
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    
    is_am = hour < 12
    hour_12 = hour if hour <= 12 else hour - 12
    if hour_12 == 0: hour_12 = 12
    
    minute_word = "o'clock" if minute == 0 else (f"oh {num2words(minute)}" if minute < 10 else num2words(minute))
    ampm = "AM" if is_am else "PM"
    
    written = f"{hour:02d}:{minute:02d}"
    
    if minute == 0:
        spoken = f"{num2words(hour_12)} {ampm}"
    else:
        spoken = f"{num2words(hour_12)} {minute_word} {ampm}"
        
    return written, spoken

def generate_currency():
    dollars = random.randint(0, 1000)
    cents = random.randint(0, 99)
    
    written = f"${dollars}.{cents:02d}"
    
    dollar_word = "dollar" if dollars == 1 else "dollars"
    cent_word = "cent" if cents == 1 else "cents"
    
    if cents == 0:
        spoken = f"{num2words(dollars)} {dollar_word}"
    elif dollars == 0:
        spoken = f"{num2words(cents)} {cent_word}"
    else:
        spoken = f"{num2words(dollars)} {dollar_word} and {num2words(cents)} {cent_word}"
        
    return written, spoken

def generate_phone():
    area = random.randint(200, 999)
    prefix = random.randint(200, 999)
    line = random.randint(1000, 9999)
    
    written = f"{area}-{prefix}-{line}"
    
    def digit_to_word(d):
        return num2words(int(d)) if d != '0' else 'zero'
        
    spoken = " ".join([digit_to_word(d) for d in str(area)]) + ", " + \
             " ".join([digit_to_word(d) for d in str(prefix)]) + ", " + \
             " ".join([digit_to_word(d) for d in str(line)])
             
    return written, spoken

def generate_decimal():
    whole = random.randint(0, 100)
    fraction = random.randint(1, 999)
    
    written = f"{whole}.{fraction}"
    
    spoken = f"{num2words(whole)} point " + " ".join([num2words(int(d)) if d != '0' else 'zero' for d in str(fraction)])
    return written, spoken

def generate_hebrew_gender():
    # Words where the spelling is identical but pronunciation changes based on addressee's gender
    words = [
        # Prepositions
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
        
        # Nouns with possessive suffixes
        {"written": "לידיעתך", "m": "leyedi'at'cha", "f": "leyedi'atech", "meaning": "for your info"},
        {"written": "ידיעתך", "m": "yedi'at'cha", "f": "yedi'atech", "meaning": "your info"},
        {"written": "דעתך", "m": "da'at'cha", "f": "da'atech", "meaning": "your opinion"},
        {"written": "שמך", "m": "shimcha", "f": "shmech", "meaning": "your name"},
        {"written": "רצונך", "m": "retzoncha", "f": "retzonech", "meaning": "your desire"},
        {"written": "שאלתך", "m": "she'elatcha", "f": "she'elatech", "meaning": "your question"},
        {"written": "כוונתך", "m": "kavanat'cha", "f": "kavanatech", "meaning": "your intention"},
        {"written": "תשובתך", "m": "tshuvatcha", "f": "tshuvatech", "meaning": "your answer"},
        {"written": "זכותך", "m": "zchutcha", "f": "zchutech", "meaning": "your right"},

        # Past tense verbs (2nd person singular)
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
    
    item = random.choice(words)
    gender = random.choice(["M", "F"])
    
    # Simulating a prompt that gives context or explicitly states the gender to read for
    written = f"{item['written']} (Read as {'Masculine' if gender == 'M' else 'Feminine'})"
    spoken = item["m"] if gender == "M" else item["f"]
    
    return written, spoken

generators = [
    (generate_number, 0.15),
    (generate_date, 0.15),
    (generate_time, 0.15),
    (generate_currency, 0.15),
    (generate_phone, 0.1),
    (generate_decimal, 0.15),
    (generate_hebrew_gender, 0.15)
]

data = []
for _ in range(1000):
    gen = random.choices([g[0] for g in generators], weights=[g[1] for g in generators])[0]
    written, spoken = gen()
    
    # clean up spoken formatting
    spoken = spoken.replace(" and ", " ").replace("-", " ")
    
    data.append({
        "type": gen.__name__.replace("generate_", ""),
        "written": written,
        "spoken": spoken
    })

with open("text_normalization_prompts.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["type", "written", "spoken"])
    writer.writeheader()
    writer.writerows(data)

print(f"Generated {len(data)} examples.")
