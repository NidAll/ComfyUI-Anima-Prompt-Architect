"""User direction for image clones: turn free text into tag edits and a caption brief.

The image analysis says what the picture shows; the direction says what the new
picture should change. Phrases are resolved against the PixAI vocabulary so they
become real Anima tags in their proper slot, image tags that contradict them are
replaced (hair colour, time of day, expression, outfit...), and the visual model
is asked to describe the picture with the changes already applied.
"""

import re

COLOR_WORDS = {"red", "blue", "green", "black", "white", "yellow", "pink", "purple", "orange", "brown",
               "grey", "blonde", "silver", "aqua", "gold", "golden", "light brown", "dark blue",
               "light blue", "dark green", "dark", "multicolored", "two-tone", "gradient", "beige", "teal"}
COLOR_SPELLINGS = {"gray": "grey", "blond": "blonde"}
COUNT_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6"}
NUMBERED = re.compile(r"^(?:\d+|multiple)(?:girls|boys|others|girl|boy|other)$")
RATINGS = {"safe", "sensitive", "nsfw", "explicit"}

# Whole-phrase rewrites applied before the vocabulary lookup.
PHRASES = {"redhead": "red hair", "redhaired": "red hair", "smiling": "smile", "smiles": "smile",
           "night time": "night", "nighttime": "night", "at night": "night", "daytime": "day",
           "raining": "rain", "rainy": "rain", "snowy": "snowing", "sunny": "sunlight",
           "eyes closed": "closed eyes", "silver hair": "grey hair", "blond": "blonde hair", "blonde": "blonde hair", "eyes shut": "closed eyes", "frowning": "frown",
           "laughing": "laughing", "crying": "crying", "angry face": "angry", "sad face": "sad"}

# One value per slot: a user value replaces every other value the image had.
SLOTS = {
    "hair length": {"short hair", "medium hair", "long hair", "very long hair", "absurdly long hair"},
    "expression": {"smile", "grin", "frown", "angry", "sad", "crying", "expressionless", "serious",
                   "surprised", "scared", "annoyed", "pout", "smug", "laughing"},
    "time": {"day", "night", "sunset", "sunrise", "evening", "dusk", "twilight", "dawn", "morning"},
    "weather": {"rain", "snowing", "sunlight", "clear sky", "fog", "cloudy sky", "overcast"},
    "framing": {"portrait", "upper body", "cowboy shot", "full body", "close-up", "lower body"},
    "setting": {"indoors", "outdoors"},
    "eyes": {"closed eyes", "open eyes"},
    "gaze": {"looking at viewer", "looking away", "looking up", "looking down", "looking back",
             "looking to the side"},
    "colour": {"monochrome", "greyscale", "full color"},
}
# Things a value rules out beyond its own slot.
EXCLUDES = {
    "night": {"blue sky", "sunlight", "daylight", "light rays", "sunbeam"},
    "evening": {"blue sky"}, "sunset": {"blue sky"}, "dusk": {"blue sky"},
    "day": {"night sky", "starry sky", "moon", "full moon", "crescent moon"},
    "rain": {"sunlight", "blue sky", "clear sky"},
    "indoors": {"sky", "blue sky", "cloud", "cloudy sky", "starry sky"},
    "closed eyes": {"looking at viewer"},
}
# Caption words that contradict a requested value even when no image tag said so.
CAPTION_CONFLICTS = {
    "night": ("daytime", "daylight", "sunlight", "sunny", "sunlit", "midday", "bright natural light"),
    "day": ("nighttime", "at night", "moonlight", "moonlit", "starry", "starlit"),
    "evening": ("midday",), "sunset": ("midday",),
    "rain": ("sunny", "clear sky", "cloudless"), "snowing": ("sunny", "summer"),
    "indoors": ("outdoors", "open sky"), "outdoors": ("indoors",),
}
OUTFITS = {"dress", "kimono", "school uniform", "uniform", "swimsuit", "bikini", "armor", "suit", "bodysuit",
           "jumpsuit", "robe", "gown", "overalls", "serafuku", "maid", "yukata", "leotard", "pajamas"}
GARMENTS = OUTFITS | {"apron", "shirt", "raincoat", "coat", "jacket", "hoodie", "sweater", "skirt", "pants",
                      "jeans", "denim", "shorts", "blouse", "cardigan", "vest", "t-shirt", "tank top", "crop top"}
# Single words that are tags but too vague to lift out of a sentence.
VAGUE = {"solo", "holding", "light", "dark", "text", "signature", "simple background", "censored",
         "day", "looking", "standing", "facing", "from", "outdoors", "indoors", "sky", "water", "close",
         "male focus", "female focus", "english text", "one", "two", "bad", "blur", "no", "falling",
         "paper", "make", "turn", "give", "background", "front"}
STOPWORDS = {"the", "a", "an", "and", "with", "her", "his", "their", "its", "she", "he", "they", "is", "are",
             "in", "on", "at", "of", "to", "into", "from", "under", "over", "while", "as", "by", "it", "this",
             "that", "behind", "front", "some", "very", "has", "have", "wears", "wearing", "stands", "sits"}


# Sentence verbs in the tag form Danbooru uses ("holds a sword" -> "holding sword").
VERB_FORMS = {"looks": "looking", "look": "looking", "holds": "holding", "hold": "holding", "sits": "sitting",
              "stands": "standing", "lies": "lying", "runs": "running", "walks": "walking", "flies": "flying",
              "reads": "reading", "sleeps": "sleeping", "eats": "eating", "drinks": "drinking", "dances": "dancing",
              "jumps": "jumping", "kneels": "kneeling", "leans": "leaning", "sings": "singing", "cries": "crying",
              "points": "pointing", "smiles": "smile", "smiling": "smile", "frowns": "frown"}
DETERMINERS = {"a", "an", "the", "her", "his", "their", "its", "some"}


def normalize(tag):
    tag = str(tag).strip().lower()
    if not re.fullmatch(r"score_\d+", tag):
        tag = tag.replace("_", " ")
    return re.sub(r"\s+", " ", tag)


def _respell(phrase):
    words = [COLOR_SPELLINGS.get(w, w) for w in phrase.split()]
    phrase = " ".join(words)
    phrase = re.sub(r"\b(\w+)[- ]haired\b", r"\1 hair", phrase)
    phrase = re.sub(r"\b(\w+)[- ]eyed\b", r"\1 eyes", phrase)
    match = re.fullmatch(r"(one|two|three|four|five|six) (girl|boy)s?", phrase)
    if match:
        phrase = COUNT_WORDS[match[1]] + match[2] + ("" if match[1] == "one" else "s")
    return PHRASES.get(phrase, phrase)


def vocabulary(raw):
    """{tag: category} for every tag the tagger knows (its full output, not only hits)."""
    return {normalize(tag): category for category, scores in (raw or {}).items() for tag in scores}


def resolve(phrase, vocab):
    """A user phrase as an Anima tag: (tag, category, known)."""
    tag = _respell(normalize(phrase).strip(" .!?"))
    for candidate in (tag, tag + "s", tag[:-1] if tag.endswith("s") else None, tag.replace("-", " ")):
        if candidate and candidate in vocab:
            return candidate, vocab[candidate], True
    if NUMBERED.fullmatch(tag):
        return tag, "general", True
    return tag, "general", False


def sentence_tags(sentence, vocab):
    """Known tags named in a sentence (longest match first), plus 'no X' / 'without X' removals."""
    text = _respell(normalize(sentence).strip(" .!?"))
    words = [VERB_FORMS.get(w, w) for w in re.findall(r"[a-z0-9][a-z0-9'-]*", text) if w not in DETERMINERS]

    def match(index):
        for size in (4, 3, 2, 1):
            if index + size > len(words):
                continue
            phrase = _respell(" ".join(words[index:index + size]))
            if phrase not in vocab and phrase.endswith("s") and phrase[:-1] in vocab:
                phrase = phrase[:-1]
            if vocab.get(phrase) not in ("general", "meta"):
                continue
            if size == 1 and (phrase in VAGUE or phrase in STOPWORDS):
                return None
            return phrase, size
        return None

    found, removed, index = [], [], 0
    while index < len(words):
        hit = match(index)
        if not hit:
            index += 1
            continue
        phrase, size = hit
        # "holding paper lantern": a longer tag starting inside this one wins.
        inner = match(index + 1) if size > 1 else None
        if inner and index + 1 + inner[1] > index + size:
            if phrase.startswith("holding ") and "holding " + inner[0].split()[-1] in vocab:
                found.append("holding " + inner[0].split()[-1])
            index += 1
            continue
        negated = index > 0 and words[index - 1] in ("no", "without") or words[max(0, index - 2):index] == ["instead", "of"]
        (removed if negated else found).append(phrase)
        # Danbooru lists the object with the action: "holding sword" also gets "sword".
        held = phrase[8:] if phrase.startswith("holding ") else ""
        if held and not negated and vocab.get(held) == "general":
            found.append(held)
        index += size
    return list(dict.fromkeys(found)), list(dict.fromkeys(removed))


def parse(text, vocab=None):
    """Split direction into tags (with their tagger category), prose and removals."""
    vocab = vocab or {}
    tags, prose, removed = [], [], []

    def add(tag, category, known, source):
        if tag and tag not in (t["tag"] for t in tags):
            tags.append({"tag": tag, "category": category, "known": known, "source": source})

    for line in re.split(r"[\r\n]+", str(text or "")):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",") if p.strip()]
        sentence = bool(re.search(r"[.!?]$", line)) or (len(parts) == 1 and len(line.split()) > 6)
        if sentence and not all(p.startswith("-") for p in parts):
            parts = [line]
        for part in parts:
            if part.startswith("-"):
                tag = resolve(part.lstrip("- "), vocab)[0]
                if tag:
                    removed.append(tag)
            elif sentence or len(part.split()) > 6:
                prose.append(part.rstrip(".!? ") + ".")
                found, negated = sentence_tags(part, vocab)
                for tag in found:
                    add(tag, vocab.get(tag, "general"), True, "sentence")
                removed.extend(negated)
            else:
                negation = re.fullmatch(r"(?:no|without)\s+(.+)", normalize(part))
                if negation:
                    removed.append(resolve(negation[1], vocab)[0])
                    continue
                add(*resolve(part, vocab), "user")
    removed = list(dict.fromkeys(t for t in removed if t))
    return [t for t in tags if t["tag"] not in removed], prose, removed


def color_split(tag):
    for color in sorted(COLOR_WORDS, key=len, reverse=True):
        if tag.startswith(color + " "):
            return color, tag[len(color) + 1:]
    return None, tag


def _noun(tag):
    for garment in sorted(GARMENTS, key=len, reverse=True):
        if tag == garment or tag.endswith(" " + garment):
            return garment
    return None


def conflicts(user_tags, image_tags):
    """{image tag: user tag that replaces it}; removals are handled separately."""
    replaced = {}
    user = list(user_tags)
    for tag in user:
        for slot in SLOTS.values():
            if tag in slot:
                replaced.update({other: tag for other in image_tags if other in slot and other not in user})
        replaced.update({other: tag for other in image_tags if other in EXCLUDES.get(tag, ()) and other not in user})
        color, noun = color_split(tag)
        if color:
            for other in image_tags:
                other_color, other_noun = color_split(other)
                if other_color and other_noun == noun and other not in user:
                    replaced[other] = tag
        # A new full outfit replaces the clothes the image showed.
        if _noun(tag) in OUTFITS:
            replaced.update({other: tag for other in image_tags if _noun(other) and other not in user})
    counts = [t for t in user if NUMBERED.fullmatch(t)]
    if counts:
        replaced.update({t: counts[0] for t in image_tags if NUMBERED.fullmatch(t) and t not in user})
        if any(not t.startswith("1") for t in counts):
            replaced["solo"] = counts[0]
    return {old: new for old, new in replaced.items() if old in image_tags}


def removals(removed, image_tags):
    """Removing 'hat' also removes 'witch hat'; removing 'smile' leaves 'smiley face' alone."""
    out = set()
    for tag in removed:
        out.update(other for other in image_tags if other == tag or other.endswith(" " + tag)
                   or (" " not in other and len(tag) >= 3 and other.endswith(tag)))  # bag -> handbag
    return out


def contradicts(text, user_tags):
    """A caption phrase that says the opposite of a requested value."""
    return any(re.search(r"\b" + re.escape(word) + r"\b", text) for tag in user_tags for word in CAPTION_CONFLICTS.get(tag, ()))


def mentions(text, tag):
    if re.search(r"\b" + re.escape(tag) + r"s?\b", text):
        return True
    color, noun = color_split(tag)
    return bool(color) and bool(re.search(r"\b" + re.escape(color) + r"\b[^.]{0,30}\b" + re.escape(noun) + r"\b", text))


def recolor(sentence, old, new):
    """'long blonde hair' -> 'long red hair' when the user recoloured an attribute."""
    old_color, noun = color_split(old)
    new_color, new_noun = color_split(new)
    if not (old_color and new_color and noun == new_noun):
        return sentence
    spellings = [old_color] + [k for k, v in COLOR_SPELLINGS.items() if v == old_color]
    pattern = r"\b(?:" + "|".join(map(re.escape, spellings)) + r")\b(?=[^.,;]{0,30}?\b" + re.escape(noun) + r")"
    return re.sub(pattern, new_color, sentence, flags=re.IGNORECASE)


def brief(text):
    """The direction as a short change list for the visual model."""
    lines = []
    for line in re.split(r"[\r\n]+", str(text or "")):
        for part in (p.strip() for p in line.split(",")):
            if not part:
                continue
            lines.append("remove " + part.lstrip("- ") if part.startswith("-") else part.rstrip("."))
    return "; ".join(lines)[:600]


def covered(sentence, caption):
    """Whether the caption already says what a user sentence asked for."""
    words = [w for w in re.findall(r"[a-z]{4,}", sentence.lower()) if w not in STOPWORDS]
    if not words:
        return True
    caption = caption.lower()
    return sum(1 for w in words if w[:5] in caption) / len(words) >= .6


def edit_list(user, prose, replacements, removed):
    """Explicit changes for the rewrite: what replaces what, what goes, what is added."""
    lines, replacing = [], {}
    for old, new in replacements.items():
        replacing.setdefault(new, []).append(old)
    for item in user:
        tag = item["tag"]
        if tag in replacing:
            lines.append(f"replace {' and '.join(repr(o) for o in replacing[tag])} with {tag!r}")
        elif item["source"] == "user":
            lines.append(f"add {tag!r}")
    lines += [f"remove {tag!r}" for tag in removed]
    lines += [sentence.rstrip(".") for sentence in prose]
    return "; ".join(lines)[:800]


def prune(sentence, stale):
    """Drop the clauses of a sentence that stale() flags; '' when too little is left to keep."""
    clauses = re.split(r",\s*(?:and\s+)?|\s+and\s+|;\s*|\s+while\s+", sentence)
    kept = [c.strip() for c in clauses if c.strip() and not stale(c)]
    if not kept or sum(map(len, kept)) < .5 * len(sentence) or stale(kept[0]):
        return ""
    return ", ".join(kept)


REWRITE_EXAMPLE = (
    "Description: A girl with long blonde hair smiles, wearing a white dress and a straw hat, in a sunny field "
    "under a blue sky.\n"
    "Changes: replace 'blonde hair' with 'black hair'; replace 'smile' with 'crying'; "
    "replace 'blue sky' with 'rain'; remove 'straw hat'; add 'holding umbrella'\n"
    "Rewritten: A girl with long black hair is crying, wearing a white dress and holding an umbrella, in a field "
    "under heavy rain and a grey sky.")


def rewrite_request(caption, brief):
    """Ask for the caption rewritten as the changed picture, nothing else, with one worked example."""
    return ("Rewrite a picture description so it describes the picture after the listed changes. "
            "Apply every change: replace the old detail with the new one, delete removed details, and work each "
            "added detail into the sentence where it belongs. Adjust light, sky and colors when a change implies it "
            "(night is dark, rain has no sunlight). Keep every other detail and sentence as written, never mention "
            "the original or that anything changed, and return only the rewritten description.\n\n"
            "Example\n" + REWRITE_EXAMPLE + "\n\n"
            "Description: " + caption + "\nChanges: " + brief + "\nRewritten:")


def accept_rewrite(original, rewritten):
    """The rewrite, cleaned, or '' when it is unusable and the image caption should be kept."""
    text = re.sub(r"^(?:here is|here's|rewritten(?: description)?)[^:\n]*:\s*", "", str(rewritten or "").strip(), flags=re.IGNORECASE)
    text = text.strip().strip('"\u201c\u201d').strip()
    if any(marker in text for marker in ("<think>", "</think>", "<|im_start|>")):
        return ""
    if len(text) < .4 * len(original) or len(text) > 2.5 * len(original) + 200:
        return ""
    if re.search(r"\b(?:original (?:image|picture)|the request|was changed|has been changed|instead of the original)\b", text, re.IGNORECASE):
        return ""
    return text


WEATHER = {"rain": "it is raining", "snowing": "it is snowing", "fog": "the scene is foggy",
           "night": "it is night", "sunset": "it is sunset", "evening": "it is evening", "dusk": "it is dusk",
           "day": "it is daytime", "indoors": "the scene is indoors", "outdoors": "the scene is outdoors"}


def _said(tag, text):
    words = [w for w in re.findall(r"[a-z]+", tag) if w not in ("holding", "looking")]
    return all(re.search(r"\b" + re.escape(w[:max(4, len(w) - 2)]), text) for w in words) if words else True


def missing_sentence(user, prose, counts):
    """One plain sentence for requested tags the description still never mentions."""
    text = prose.lower()
    tags = [t["tag"] for t in user if t["source"] == "user" and t["category"] == "general"
            and not NUMBERED.fullmatch(t["tag"]) and not _said(t["tag"], text)]
    if not tags:
        return ""
    subject = "she" if any(c in ("1girl",) for c in counts) else "he" if "1boy" in counts else "they"
    be, have = ("are", "have") if subject == "they" else ("is", "has")
    actions, traits, scene = [], [], []
    for tag in tags:
        if tag in WEATHER:
            scene.append(WEATHER[tag])
        elif tag.startswith("holding "):
            noun = tag[8:]
            actions.append("holding " + ("an " if noun[0] in "aeiou" else "a ") + noun)
        elif tag.split()[0].endswith("ing") or tag in SLOTS["expression"]:
            actions.append(tag if tag.split()[0].endswith("ing") else "showing a " + tag)
        else:
            traits.append(tag)
    parts = []
    if actions:
        parts.append(f"{subject} {be} " + " and ".join(actions))
    if traits:
        parts.append(f"{subject} {have} " + ", ".join(traits))
    parts += scene
    sentence = ", and ".join(parts) if len(parts) < 3 else ", ".join(parts[:-1]) + ", and " + parts[-1]
    return sentence[0].upper() + sentence[1:] + "."
