"""Evidence filtering and the official CircleStone Anima prompt format."""

import re

THRESHOLDS = {"general": .17, "character": .27, "style": .15,
              "copyright": .24, "meta": .17, "rating": .41}
NEGATIVE = "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration"
RATINGS = {"rating:g": "safe", "rating:s": "sensitive", "rating:q": "nsfw", "rating:e": "explicit"}
COUNTS = re.compile(r"^(?:\d+|multiple)(?:girls|boys|others|girl|boy|other)$")
GROUPS = [
    {"solo", "multiple girls", "multiple boys", "multiple others"},
    {"monochrome", "greyscale", "full color"},
    {"day", "night"}, {"indoors", "outdoors"},
    {"eyes closed", "eyes open"}, {"looking at viewer", "looking away"},
]
LIMITS = {"concise": 25, "balanced": 50, "detailed": 85}
CLOTHING = {"overalls", "apron", "dress", "shirt", "raincoat", "coat", "jacket", "hoodie", "sweater",
            "skirt", "pants", "jeans", "shorts", "kimono", "uniform", "suit", "armor", "swimsuit"}


def normalize(tag):
    tag = tag.strip().lower()
    if not re.fullmatch(r"score_\d+", tag):
        tag = tag.replace("_", " ")
    return re.sub(r"\s+", " ", tag)


def select_tags(raw, thresholds, fidelity):
    selected = []
    rejected = []
    for category in THRESHOLDS:
        threshold = thresholds.get(category, THRESHOLDS[category])
        if category in ("character", "copyright", "style") and fidelity == "strict":
            threshold = max(threshold, .65)
        for tag, score in raw.get(category, {}).items():
            if score >= threshold:
                selected.append({"tag": normalize(tag), "original": tag, "score": float(score), "category": category})
    selected.sort(key=lambda x: (-x["score"], x["tag"]))
    unique = {}
    for item in selected:
        unique.setdefault(item["tag"], item)
    selected = list(unique.values())
    for group in GROUPS:
        matches = [x for x in selected if x["tag"] in group]
        if matches:
            for item in matches[1:]:
                selected.remove(item)
                rejected.append({**item, "reason": "mutually exclusive tag; stronger PixAI evidence retained"})
    # One count per subject class; mixed girl/boy groups remain valid.
    for kind in ("girl", "boy", "other"):
        matches = [x for x in selected if COUNTS.fullmatch(x["tag"]) and x["tag"].rstrip("s").endswith(kind)]
        for item in matches[1:]:
            selected.remove(item)
            rejected.append({**item, "reason": "conflicting subject count"})
    if any(COUNTS.fullmatch(x["tag"]) and not x["tag"].startswith("1") for x in selected):
        selected = [x for x in selected if x["tag"] != "solo"]
    present = {x["tag"] for x in selected}
    redundant = set()
    if "raincoat" in present:
        redundant.add("coat")
    if "yellow raincoat" in present:
        redundant.update(("yellow coat", "raincoat"))
    for item in selected:
        tag = item["tag"]
        if tag == "holding" and any(t.startswith("holding ") for t in present):
            redundant.add(tag)
        if tag in ("umbrella", "shirt", "pants", "dress", "hat", "hair") and any(t.endswith(" " + tag) and not t.startswith("holding ") for t in present):
            redundant.add(tag)
    selected = [x for x in selected if x["tag"] not in redundant]
    return selected, rejected


def fuse(raw, analysis, variant="Base", detail="balanced", fidelity="strict", thresholds=None,
         include_visual_details=True, artist="", safety="auto"):
    if variant not in ("Base", "Aesthetic", "Turbo") or detail not in LIMITS or fidelity not in ("strict", "balanced"):
        raise ValueError("Unknown Anima profile, detail or fidelity.")
    selected, rejected = select_tags(raw, thresholds or {}, fidelity)
    by_category = {key: [x for x in selected if x["category"] == key] for key in THRESHOLDS}
    rating = by_category["rating"][:1]
    safety_tag = RATINGS.get(rating[0]["original"], "") if safety == "auto" and rating else ""
    if safety != "auto":
        if safety not in ("omit", "safe", "sensitive", "nsfw", "explicit"):
            raise ValueError("Unknown safety tag.")
        safety_tag = "" if safety == "omit" else safety
    prefix = ["masterpiece", "best quality"]
    if variant != "Aesthetic":
        prefix.append("score_7")
    if safety_tag:
        prefix.append(safety_tag)
    # Do not copy inferred publication year, quality judgments or technical artifacts.
    meta = [x["tag"] for x in by_category["meta"] if x["tag"] in {"anime screenshot", "traditional media", "digital media", "sketch"}]
    general = by_category["general"]
    counts = [x["tag"] for x in general if COUNTS.fullmatch(x["tag"])]
    identities = [x["tag"] for cat in ("character", "copyright") for x in by_category[cat]]
    # PixAI style labels are not reliable artist attributions. Only explicit user artists get @.
    artists = ["@" + normalize(a.strip().lstrip("@")) for a in artist.split(",") if a.strip().lstrip("@").strip()]
    styles = [x["tag"] for x in by_category["style"] if not x["tag"].startswith("artist:")]
    # Style IDs often denote artists rather than media; expose these in raw analysis only.
    styles = [s for s in styles if s in {"watercolor", "oil painting", "pixel art", "sketch", "lineart", "anime coloring", "flat color", "cel shading"}]
    tags = prefix + meta + counts + identities + artists + [x["tag"] for x in general if x["tag"] not in counts] + styles
    tags = list(dict.fromkeys(tags))[:LIMITS[detail]]
    tag_set = {x["tag"] for x in selected}
    analysis = dict(analysis)
    caption_facts = []
    aliases = {"1girl": "woman", "1boy": "man", "girl": "woman", "boy": "man", "city": "urban", "building": "storefronts"}
    for sentence in re.split(r"(?<=[.!?])\s+", analysis.get("caption", "")):
        sentence = sentence.strip().rstrip(".!?")
        if not sentence:
            continue
        lowered = normalize(sentence)
        if fidelity == "strict":
            for garment in CLOTHING:
                if not any(item["category"] == "general" and (item["tag"] == garment or item["tag"].endswith(" " + garment)) for item in selected):
                    sentence = re.sub(r"\b(?:in|wearing|dressed in)\s+(?:(?:a|an|the|red|blue|green|black|white|yellow|pink|purple|orange|brown|gray|grey)\s+)*" + garment + r"\b", "", sentence, flags=re.IGNORECASE)
            sentence = re.sub(r"\s+", " ", sentence).strip()
            # A still image does not establish motion.
            sentence = re.sub(r"\bmoves along\b", "is on", sentence, flags=re.IGNORECASE)
            lowered = normalize(sentence)
        support = [item for item in selected if item["category"] == "general" and
                   (re.search(r"\b" + re.escape(item["tag"]) + r"\b", lowered) or
                    item["tag"] in aliases and re.search(r"\b" + aliases[item["tag"]] + r"\b", lowered))]
        visual_property = re.search(r"\b(light(?:ing)?|shadow|illuminat\w*|palette|color\w*|paint\w*|artwork|illustrat\w*)\b", lowered)
        if include_visual_details and (support or visual_property or fidelity == "balanced"):
            caption_facts.append({"field": "visual_description", "text": sentence,
                                  "evidence": [item["tag"] for item in support],
                                  "confidence": None,
                                  "confidence_basis": "Unscored MiniCPM observation; matching PixAI tags corroborate nouns only"})
    analysis["facts"] = analysis.get("facts", []) + caption_facts
    facts = []
    for item in analysis.get("facts", []):
        if item["confidence"] is not None and item["confidence"] < (.85 if fidelity == "strict" else .65):
            continue
        text = item["text"]
        lowered = normalize(text)
        conflict = False
        for group in GROUPS:
            supported = group & tag_set
            if supported and any(re.search(r"\b" + re.escape(opposite) + r"\b", lowered) for opposite in group - supported):
                conflict = True
        if conflict:
            rejected.append({**item, "reason": "visual prose conflicts with stronger tag evidence"})
            continue
        if lowered in tag_set or lowered in {normalize(f["text"]) for f in facts}:
            continue
        facts.append(item)
    fact_limit = {"concise": 2, "balanced": 5, "detailed": 9}[detail]
    prose = " ".join(x["text"] + "." for x in facts[:fact_limit])
    positive = ", ".join(tags) + ("\n\n" + prose if prose else "")
    negative = NEGATIVE
    if variant == "Aesthetic":
        negative = ", ".join(t for t in NEGATIVE.split(", ") if not t.startswith("score_"))
    if variant == "Turbo":
        negative = ""
    return positive, negative, {"variant": variant, "selected_tags": selected, "retained_facts": facts[:fact_limit],
                                "rejected": rejected, "safety_tag": safety_tag,
                                "negative_active": variant != "Turbo", "analysis": analysis,
                                "limits": "Model confidence is uncalibrated. Evidence text is model-generated, not independent verification."}
