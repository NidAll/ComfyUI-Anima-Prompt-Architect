"""What the PixAI tagger knows, applied to a typed idea before the language model writes.

Two kinds of knowledge are used:

* The vocabulary (30,877 tags in six categories) recognises named characters, series,
  artists, art media/styles and camera/framing words in the idea, so they reach their
  official Anima slot even when the small language model would miss or misspell them.
* The classifier head is one learned vector per tag; tags that appear together in
  pictures point the same way. Centred and corrected for hub tags (CSLS), it links a
  character to its series and to its usual look (hair, eyes, outfit). Measured on the
  shipped weights: the series came out top-1 for Miku, Fern, Frieren, Rem and Saber.
  The look is noisier, so it is only offered to the language model as a hint and a hint
  tag is kept only when the model's own answer agrees with it.

Nothing here downloads, caches across executions or runs the tagger on an image.
"""

import re

# Spoken words for camera, framing and viewpoint, mapped to their Danbooru tags.
CAMERA = {
    "from above": "from above", "bird's eye view": "from above", "birds eye view": "from above",
    "bird's-eye view": "from above", "overhead view": "from above", "overhead shot": "from above",
    "top-down view": "from above", "top down view": "from above", "high angle": "from above",
    "from below": "from below", "low angle": "from below", "worm's eye view": "from below",
    "worms eye view": "from below", "looking up at her": "from below", "looking up at him": "from below",
    "from the side": "from side", "from side": "from side", "side view": "from side",
    "in profile": "profile", "profile view": "profile",
    "from behind": "from behind", "back view": "from behind", "rear view": "from behind",
    "seen from behind": "from behind", "dutch angle": "dutch angle", "tilted angle": "dutch angle",
    "tilted camera": "dutch angle", "canted angle": "dutch angle", "pov": "pov", "first person": "pov",
    "first-person": "pov", "point of view": "pov", "fisheye": "fisheye", "fish-eye": "fisheye",
    "wide shot": "wide shot", "long shot": "wide shot", "establishing shot": "wide shot",
    "very wide shot": "very wide shot", "extreme wide shot": "very wide shot",
    "close up": "close-up", "close-up": "close-up", "closeup": "close-up",
    "headshot": "portrait", "head shot": "portrait", "portrait shot": "portrait",
    "waist up": "upper body", "waist-up": "upper body", "half body": "upper body", "half-body": "upper body",
    "bust shot": "upper body", "upper body": "upper body", "knees up": "cowboy shot", "thighs up": "cowboy shot",
    "cowboy shot": "cowboy shot", "medium shot": "cowboy shot", "full body": "full body",
    "full-body": "full body", "full length": "full body", "full-length": "full body", "head to toe": "full body",
    "whole body": "full body", "looking back": "looking back", "over the shoulder": "looking back",
    "over her shoulder": "looking back", "over his shoulder": "looking back",
    "looking at the viewer": "looking at viewer", "looking at viewer": "looking at viewer",
    "looking at the camera": "looking at viewer", "looking at camera": "looking at viewer",
    "eye contact": "looking at viewer", "symmetrical": "symmetry", "symmetry": "symmetry",
    "turnaround": "multiple views", "reference sheet": "reference sheet", "multiple views": "multiple views",
    "foreshortening": "foreshortening", "dynamic pose": "dynamic pose", "action pose": "dynamic pose",
}
# Spoken words for media and art styles; "(medium)" / "(style)" variants are also tried.
STYLES = {
    "watercolor": "watercolor (medium)", "watercolour": "watercolor (medium)", "oil painting": "oil painting (medium)",
    "oil paint": "oil painting (medium)", "acrylic": "acrylic paint (medium)", "gouache": "gouache (medium)",
    "pencil": "graphite (medium)", "graphite": "graphite (medium)", "colored pencil": "colored pencil (medium)",
    "coloured pencil": "colored pencil (medium)", "ink": "ink (medium)", "ink wash": "ink wash painting",
    "marker": "marker (medium)", "pastel": "pastel (medium)", "crayon": "crayon (medium)",
    "line art": "lineart", "lineart": "lineart", "pencil sketch": "sketch", "sketch": "sketch",
    "pixel art": "pixel art", "pixel": "pixel art", "8-bit": "pixel art", "chibi": "chibi",
    "flat colors": "flat color", "flat color": "flat color", "flat colours": "flat color",
    "cel shading": "anime coloring", "cel shaded": "anime coloring", "cel-shaded": "anime coloring",
    "anime screenshot": "anime screenshot", "screencap": "anime screenshot",
    "black and white": "monochrome", "monochrome": "monochrome", "greyscale": "greyscale", "grayscale": "greyscale",
    "retro": "retro artstyle", "90s anime": "1990s (style)", "90s": "1990s (style)", "1990s": "1990s (style)",
    "80s anime": "1980s (style)", "80s": "1980s (style)", "1980s": "1980s (style)", "70s": "1970s (style)",
    "2000s": "2000s (style)", "realistic": "realistic", "photorealistic": "photorealistic", "3d": "3d",
    "ukiyo-e": "ukiyo-e", "ukiyoe": "ukiyo-e", "art nouveau": "art nouveau", "impressionist": "impressionism",
    "impressionism": "impressionism", "oekaki": "oekaki", "official art": "official art",
    "traditional media": "traditional media", "painterly": "painting (medium)", "painting": "painting (medium)",
}
# Viewpoints and framings that rule each other out (the first one named wins).
CAMERA_GROUPS = [
    {"from above", "from below"},
    {"portrait", "close-up", "upper body", "cowboy shot", "full body", "wide shot", "very wide shot"},
]
POSITION_WORDS = re.compile(
    r"\b(?:on|to|at) the (?:left|right)\b|\b(?:left|right)(?:-hand)? side\b|\bin the (?:foreground|background|"
    r"center|centre|middle|distance|corner)\b|\bin front of\b|\bbehind\b|\bnext to\b|\bbeside\b|\bbetween\b|"
    r"\bon top of\b|\bunder(?:neath)?\b|\babove\b|\bbelow\b|\bfacing each other\b|\bback to back\b|\bside by side\b",
    re.IGNORECASE)
ARTIST_CUES = re.compile(
    r"(?:\bin the (?:art )?style of|\bstyle of|\bart by|\bdrawn by|\billustrated by|\bpainted by|\bartist)\s+"
    r"([^,.;\n]{2,48}?)(?=\s*(?:[,.;\n]|$|\bstyle\b|\bwith\b|\band\b))|"
    r"\bby\s+([^,.;\n]{2,48}?)(?=\s*(?:[,.;\n]|$))|"
    r"([^,.;\n]{2,40}?)\s+(?:art ?style|style)\b", re.IGNORECASE)
COMMON = {"the", "a", "an", "and", "of", "in", "on", "with", "girl", "boy", "woman", "man", "anime", "style",
          "art", "night", "day", "city", "sea", "sky", "rain", "light", "dark", "red", "blue", "green", "black",
          "white", "cat", "dog", "fox", "witch", "knight", "maid", "angel", "demon", "queen", "king", "princess",
          "warrior", "mage", "nurse", "teacher", "student", "idol", "ninja", "samurai", "pirate", "robot", "dragon"}
# Everyday words that are also well-known character names; typed in lowercase they stay words.
NAME_WORDS = {"link", "saber", "archer", "lancer", "rider", "caster", "assassin", "berserker", "ruler", "avenger",
              "shadow", "ash", "may", "sakura", "rose", "lily", "iris", "violet", "ruby", "amber", "jade", "pearl",
              "coco", "sugar", "honey", "star", "sun", "moon", "sky", "snow", "angel", "ivy", "holly", "marine",
              "aqua", "doll", "ghost", "joker", "alice", "eve", "chief", "master", "mother", "nurse", "red", "blue",
              "rin", "mercy", "echo", "tracer", "venus", "mars", "luna", "nova", "kitty", "bunny", "cherry", "peach"}
APPEARANCE = re.compile(
    r"(?:^|\s)(?:\S+\s)?(?:hair|eyes)$|^(?:very long|long|short|medium) hair$|twintails|ponytail|braid|bangs|ahoge|"
    r"hair bun|bob cut|drill|sidelocks|hair ornament|hairclip|hair ribbon|hair bow|hairband|ears$|tail$|horns|"
    r"halo|wings|dress|shirt|skirt|necktie|cape|capelet|coat|jacket|armor|thighhighs|gloves|hat$|kimono|robe|"
    r"earrings|choker|glasses|sleeves|bodysuit|serafuku|bow$|mask|scarf|uniform$")


def normalize(text):
    text = str(text).strip().lower()
    if not re.fullmatch(r"score_\d+", text):
        text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text)


def _key(text):
    """Spacing- and punctuation-free form used to compare names ("Re:Zero" == "re zero")."""
    return re.sub(r"[^a-z0-9]", "", normalize(text))


def base_name(tag):
    """'fern (sousou no frieren)' -> ('fern', 'sousou no frieren')."""
    match = re.fullmatch(r"(.+?)\s*\((.+)\)", tag)
    return (match[1].strip(), match[2].strip()) if match else (tag, "")


class Index:
    """Name lookups over the vocabulary, built once per execution (about 30k entries)."""

    def __init__(self, vocab, order=None):
        self.vocab = vocab
        self.order = order or {}
        self.names = {"character": {}, "copyright": {}, "style": {}}
        # Words of common picture tags ("chain link", "rain"): a lowercase word among them is not a name.
        self.common_words = {w for tag, category in vocab.items() if category == "general"
                             and self.order.get(tag, 0) < 6000 for w in tag.split()}
        characters = sorted((t for t, c in vocab.items() if c == "character"), key=self.rank)
        self.popular = set(characters[:1500])
        # Either half of a well-known two-word name ("Reimu", "Marisa"), unless the half is a picture word.
        self.given = {}
        for tag in characters[:1500]:
            words = base_name(tag)[0].split()
            if len(words) == 2:
                for word in words:
                    if len(word) >= 3 and word not in self.common_words and word not in COMMON:
                        self.given.setdefault(_key(word), []).append(tag)
        for tag, category in vocab.items():
            if category not in self.names:
                continue
            base, qualifier = base_name(tag)
            keys = {_key(tag), _key(base)}
            words = base.split()
            if category == "character" and len(words) == 2:
                keys.add(_key(words[1] + " " + words[0]))  # "miku hatsune"
            for key in keys:
                if len(key) >= 3:
                    self.names[category].setdefault(key, []).append(tag)

    def rank(self, tag):
        return self.order.get(tag, 10**6)

    def lookup(self, category, phrase):
        found = self.names[category].get(_key(phrase), [])
        if not found and category == "character" and " " not in phrase.strip():
            found = self.given.get(_key(phrase), [])
        return sorted(found, key=self.rank)


def _phrases(text, longest=4):
    """(start, end, phrase, original) word spans of the idea, longest first per start."""
    tokens = list(re.finditer(r"[A-Za-z0-9][A-Za-z0-9'’:/.!\-]*", text))
    for start in range(len(tokens)):
        for size in range(min(longest, len(tokens) - start), 0, -1):
            span = tokens[start:start + size]
            original = text[span[0].start():span[-1].end()].strip(".!:")
            yield start, start + size, normalize(original.replace("’", "'")), original


def _proper(original):
    return any(word[:1].isupper() for word in original.split())


def recognize(idea, vocab, order=None, index=None):
    """Characters, series, artists, styles, camera tags and spatial clauses named in the idea."""
    index = index or Index(vocab, order)
    text = str(idea or "")
    found = {"characters": [], "series": [], "artists": [], "styles": [], "camera": [], "positions": [],
             "phrases": []}
    used = set()

    def free(start, end):
        return not any(i in used for i in range(start, end))

    # Characters and series: multi-word names anywhere, single words only as capitalised proper nouns
    # that are not ordinary picture words ("Saber", not "witch" or "fox").
    lowered_text = text.lower()

    def qualifier_named(tag, phrase):
        """The series in a tag's qualifier is named elsewhere in the idea ("Fern from Frieren")."""
        words = [w for w in re.findall(r"[a-z0-9]{3,}", base_name(tag)[1]) if w not in phrase.split()]
        rest = lowered_text.replace(phrase, " ")
        return bool(words) and any(re.search(r"\b" + re.escape(w) + r"\b", rest) for w in words)

    for start, end, phrase, original in _phrases(text):
        if not free(start, end):
            continue
        single = end - start == 1
        for category, bucket in (("character", "characters"), ("copyright", "series")):
            matches = index.lookup(category, phrase)
            if not matches:
                continue
            qualified = [t for t in matches if qualifier_named(t, phrase)] if category == "character" else []
            # Single words must be capitalised names that are not ordinary picture words, unless the idea
            # also names the series that qualifies them ("Fern from Frieren", where fern is also a plant).
            if single and not qualified and _proper(original) and (phrase in COMMON or vocab.get(phrase) == "general"):
                continue
            # A lowercase single word only counts as the exact tag of a well-known character ("frieren").
            top = matches[0]
            casual = (single and not _proper(original) and category == "character" and top in index.popular
                      and phrase in (top, base_name(top)[0], *base_name(top)[0].split())
                      and phrase not in index.common_words
                      and phrase not in COMMON and phrase not in NAME_WORDS and vocab.get(phrase) != "general")
            if single and not _proper(original) and not casual and not qualified:
                continue
            # The exact tag wins, then a same-named character from a series named nearby, then the most common.
            exact = [t for t in matches if t == phrase]
            tag = (exact or qualified or matches)[0]
            if tag not in [x["tag"] for x in found[bucket]]:
                found[bucket].append({"tag": tag, "phrase": original})
            used.update(range(start, end))
            break

    # "X from Y": Y is the series of X, not a second character.
    for item in list(found["characters"]):
        if re.search(r"\bfrom\s+" + re.escape(item["phrase"]), text, re.IGNORECASE) and any(
                other is not item and any(w in base_name(other["tag"])[1] for w in item["phrase"].lower().split())
                for other in found["characters"]):
            found["characters"].remove(item)

    # Artists: only names the tag database knows, after an explicit cue ("by", "style of", "... style").
    for match in ARTIST_CUES.finditer(text):
        name = next(g for g in match.groups() if g).strip()
        name = re.sub(r"^(?:the|an?)\s+", "", name, flags=re.IGNORECASE)
        candidates = (name, " ".join(name.split()[-2:]), name.split()[-1] if name.split() else "")
        # Bare "by X" and "X style" are everyday phrases ("by the sea", "anime style"): a lowercase name
        # only counts when it is exactly an artist the database knows and not an ordinary word.
        if not match[1] and not _proper(name.split()[-1] if name.split() else ""):
            bare = normalize(name)
            exact = any(_key(base_name(t)[0]) == _key(bare) for t in index.lookup("style", bare))
            if not exact or bare in COMMON or bare in index.common_words or vocab.get(bare) == "general":
                continue
            candidates = (name,)
        for candidate in candidates:
            artists = [t for t in index.lookup("style", candidate)] if len(_key(candidate)) >= 3 else []
            if artists:
                tag = "@" + base_name(artists[0])[0] if artists[0].endswith("(style)") else "@" + artists[0]
                if tag not in found["artists"]:
                    found["artists"].append(tag)
                found["phrases"].append(normalize(match[0]))
                break

    lowered = normalize(text)
    placed = {}
    for table, bucket in ((CAMERA, "camera"), (STYLES, "styles")):
        for phrase in sorted(table, key=len, reverse=True):
            pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
            hit = re.search(pattern, lowered)
            if hit:
                target = table[phrase]
                for candidate in (target, phrase + " (medium)", phrase + " (style)", phrase + " artstyle"):
                    if candidate in vocab and candidate not in found[bucket]:
                        found[bucket].append(candidate)
                        placed[candidate] = hit.start()
                        break
                found["phrases"].append(phrase)
                # Blank the phrase with same-length padding so later positions stay comparable.
                lowered = re.sub(pattern, lambda m: " " * len(m[0]), lowered)
        # Keep the order the user wrote them in; the first framing named wins below.
        found[bucket].sort(key=lambda t: placed.get(t, 0))
    for group in CAMERA_GROUPS:
        named = [t for t in found["camera"] if t in group]
        found["camera"] = [t for t in found["camera"] if t not in group or t == (named[0] if named else t)]

    # Spatial clauses are kept word for word; the description must still say them.
    for clause in re.split(r"[,.;\n]+|\s+(?:and|while|but)\s+", text):
        clause = clause.strip()
        # Viewpoints ("from above") are camera tags, not placements.
        spatial = re.sub(r"\bfrom (?:above|below|behind|the side)\b|\blooking (?:up|down)\b", " ", clause, flags=re.IGNORECASE)
        if clause and POSITION_WORDS.search(spatial) and clause not in found["positions"]:
            found["positions"].append(clause)
    return found


class TagSpace:
    """The tagger's classifier head as tag embeddings: centred, unit length, hub-corrected."""

    def __init__(self, tags, categories, weights):
        import torch

        self.torch = torch
        self.tags = [normalize(t) for t in tags]
        self.position = {t: i for i, t in enumerate(self.tags)}
        weights = weights.float()
        weights = weights - weights.mean(0, keepdim=True)
        self.vectors = weights / weights.norm(dim=1, keepdim=True).clamp_min(1e-6)
        self.categories = list(categories)
        mask = lambda kind: torch.tensor([c == kind for c in self.categories])
        self.characters = mask("character")
        self.copyrights = mask("copyright")
        # Common appearance tags only: the list is ordered by frequency, and rare named uniforms are noise.
        self.appearance = torch.tensor([
            c == "general" and i < 3000 and bool(APPEARANCE.search(t)) and "school uniform" not in t
            and "(" not in t and not re.search(r"\b(?:tracen|sailor senshi|team)\b", t)
            for i, (t, c) in enumerate(zip(self.tags, self.categories))])
        names = self.vectors[self.appearance]
        # How close each appearance tag is to characters in general; subtracting it removes hub tags.
        self._hub = (names @ self.vectors[self.characters].T).topk(min(50, int(self.characters.sum())), dim=1).values.mean(1)
        self._appearance_names = [t for t, keep in zip(self.tags, self.appearance.tolist()) if keep]

    def series_for(self, character):
        """(series tag, score) the head ties to this character, or ('', 0) when unsure."""
        if character not in self.position:
            return "", 0.0
        qualifier = base_name(character)[1]
        scores = (self.vectors @ self.vectors[self.position[character]]).masked_fill(~self.copyrights, -9)
        values, ids = scores.topk(3)
        best, second = float(values[0]), float(values[1])
        top = [self.tags[int(i)] for i in ids]
        words = re.findall(r"[a-z0-9]{3,}", qualifier)
        # A qualifier such as "(genshin impact)" names the series; trust it among the head's top three.
        for tag, value in zip(top, values.tolist()):
            if words and all(w in tag for w in words):
                return tag, float(value)
        # Otherwise the clear leader: measured margins were >= 0.07 for every correct series checked.
        if best >= 0.22 and best - second >= 0.07:
            return top[0], best
        return "", best

    def look_of(self, character, k=8):
        """The usual look of a character as appearance tags, best first (hints, not facts)."""
        if character not in self.position:
            return []
        similarity = self.vectors[self.appearance] @ self.vectors[self.position[character]]
        values, ids = (2 * similarity - self._hub).topk(min(k, len(self._appearance_names)))
        # Measured: real looks lead with >= 0.3 (Miku 0.48, Marisa 0.62); noise peaks lower (Pikachu 0.24).
        if float(values[0]) < 0.3:
            return []
        look, seen = [], set()
        for i, v in zip(ids, values):
            tag = self._appearance_names[int(i)]
            colour = re.fullmatch(r"(\w+) (hair|eyes)", tag)
            # One hair colour and one eye colour: the strongest stays ("aqua hair", not also green and blue).
            if colour and colour[1] not in ("long", "short", "medium", "messy", "wavy", "curly", "straight"):
                if colour[2] in seen:
                    continue
                seen.add(colour[2])
            if float(v) >= 0.15:
                look.append(tag)
        return look


def facts(found, looks, series):
    """Plain lines for the language model, so it writes with what the tag database knows."""
    lines = []
    for item in found["characters"]:
        tag = item["tag"]
        name = " ".join(w.capitalize() for w in base_name(tag)[0].split())
        line = f"Character: {name}"
        if series.get(tag):
            line += f", from {series[tag].title()}"
        if looks.get(tag):
            line += ". Usually drawn with: " + ", ".join(looks[tag][:6])
        if not any(re.search(r"\b(?:hair|eyes)$", t) and not re.match(r"(?:very )?(?:long|short|medium)", t)
                   for t in looks.get(tag, [])):
            line += f". Do not state {name}'s hair or eye colour; the character tag already sets them"
        lines.append(line + ".")
    for item in found["series"]:
        lines.append(f"Series: {item['tag'].title()}.")
    if found["artists"]:
        lines.append("Artist style: " + ", ".join(found["artists"]) + ".")
    if found["styles"]:
        lines.append("Art medium and style: " + ", ".join(found["styles"]) + ".")
    if found["camera"]:
        lines.append("Camera and framing: " + ", ".join(found["camera"]) + ".")
    for clause in found["positions"]:
        lines.append(f"Keep this placement exactly: {clause}.")
    if found["characters"]:
        lines.append("Call each character by name in the description, not \"a girl\".")
    if len(found["characters"]) > 1 and not found["positions"]:
        lines.append("Place each character clearly (left / right or front / back) and describe each one in "
                     "its own sentence.")
    return lines
