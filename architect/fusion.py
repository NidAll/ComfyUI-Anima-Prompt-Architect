"""Evidence filtering and the official CircleStone Anima prompt format."""

import re

from . import direction as direction_plan

THRESHOLDS = {"general": .17, "character": .27, "style": .15,
              "copyright": .24, "meta": .17, "rating": .41}
NEGATIVE = "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration"
RATINGS = {"rating:g": "safe", "rating:s": "sensitive", "rating:q": "nsfw", "rating:e": "explicit"}
COUNTS = re.compile(r"^(?:\d+|multiple)(?:girls|boys|others|girl|boy|other)$")
GROUPS = [
    {"solo", "multiple girls", "multiple boys", "multiple others"},
    {"monochrome", "greyscale", "full color"},
    {"day", "night"}, {"indoors", "outdoors"},
    {"closed eyes", "open eyes"}, {"looking at viewer", "looking away"},
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


DIRECTION_MODES = ("lead", "follow", "replace description")


def parse_direction(text, vocab=None):
    """Direction split into tag strings, prose sentences and removals."""
    tags, prose, removed = direction_plan.parse(text, vocab)
    return [t["tag"] for t in tags], prose, removed


def split_list(value):
    return [normalize(item.lstrip("-")) for item in str(value or "").split(",") if item.strip().lstrip("-").strip()]


def plan_direction(raw, direction="", exclude_tags="", thresholds=None, fidelity="strict"):
    """What the direction does to the image tags; shared by fusion and the caption rewrite."""
    vocab = direction_plan.vocabulary(raw)
    user, prose, user_removed = direction_plan.parse(direction, vocab)
    # A rating word in the direction sets the safety tag rather than becoming a tag.
    ratings = [t["tag"] for t in user if t["tag"] in direction_plan.RATINGS]
    excluded_names = set(split_list(exclude_tags)) | set(user_removed)
    user = [t for t in user if t["tag"] not in excluded_names and t["tag"] not in direction_plan.RATINGS]
    selected, rejected = select_tags(raw, thresholds or {}, fidelity)
    image_names = [x["tag"] for x in selected]
    excluded = direction_plan.removals(excluded_names, image_names) | (excluded_names & set(image_names))
    replacements = direction_plan.conflicts([t["tag"] for t in user], image_names)
    for item in [x for x in selected if x["tag"] in excluded or x["tag"] in replacements]:
        selected.remove(item)
        reason = ("removed by user exclusion" if item["tag"] in excluded
                  else "replaced by user direction: " + replacements[item["tag"]])
        rejected.append({**item, "reason": reason})
    return {"user": user, "prose": prose, "rating": ratings[0] if ratings else "", "selected": selected,
            "rejected": rejected, "excluded_names": excluded_names, "excluded": excluded, "replacements": replacements}


def direction_brief(raw, direction="", exclude_tags="", thresholds=None, fidelity="strict"):
    """An explicit edit list for the caption rewrite, built from the resolved tag changes."""
    plan = plan_direction(raw, direction, exclude_tags, thresholds, fidelity)
    return direction_plan.edit_list(plan["user"], plan["prose"], plan["replacements"],
                                    sorted(plan["excluded"] | plan["excluded_names"]))


def fuse(raw, analysis, variant="Base", detail="balanced", fidelity="strict", thresholds=None,
         include_visual_details=True, artist="", safety="auto", direction="", direction_mode="lead",
         exclude_tags=""):
    if variant not in ("Base", "Aesthetic", "Turbo") or detail not in LIMITS or fidelity not in ("strict", "balanced"):
        raise ValueError("Unknown Anima profile, detail or fidelity.")
    if direction_mode not in DIRECTION_MODES:
        raise ValueError("Unknown direction mode.")
    plan = plan_direction(raw, direction, exclude_tags, thresholds, fidelity)
    user, user_prose, selected, rejected = plan["user"], plan["prose"], plan["selected"], plan["rejected"]
    excluded_names, excluded, replacements = plan["excluded_names"], plan["excluded"], plan["replacements"]
    if plan["rating"] and safety == "auto":
        safety = plan["rating"]
    user_names = [t["tag"] for t in user]
    replaced = set(replacements) | excluded
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
    lead = direction_mode != "follow"
    user_by = lambda *cats: [t["tag"] for t in user if t["category"] in cats and t["tag"] not in prefix]
    user_counts = [t for t in user_names if COUNTS.fullmatch(t)]
    # Do not copy inferred publication year, quality judgments or technical artifacts.
    meta = [x["tag"] for x in by_category["meta"] if x["tag"] in {"anime screenshot", "traditional media", "digital media", "sketch"}]
    general = by_category["general"]
    counts = user_counts + [x["tag"] for x in general if COUNTS.fullmatch(x["tag"])]
    characters = user_by("character") + [x["tag"] for x in by_category["character"]]
    series = user_by("copyright") + [x["tag"] for x in by_category["copyright"]]
    # PixAI style labels are not reliable artist attributions. Only explicit user artists get @.
    artists = ["@" + normalize(a.strip().lstrip("@")) for a in artist.split(",") if a.strip().lstrip("@").strip()]
    artists += ["@" + t for t in user_by("style") if "@" + t not in artists]
    styles = [x["tag"] for x in by_category["style"] if not x["tag"].startswith("artist:")]
    # Style IDs often denote artists rather than media; expose these in raw analysis only.
    styles = [s for s in styles if s in {"watercolor", "oil painting", "pixel art", "sketch", "lineart", "anime coloring", "flat color", "cel shading"}]
    user_general = [t for t in user_by("general", "meta") if t not in user_counts]
    image_general = [x["tag"] for x in general if not COUNTS.fullmatch(x["tag"])]
    # Official Anima order: quality/meta/safety, count, character, series, artist, general.
    # Your tags are never cut by the budget; image tags fill what the detail level leaves.
    fixed = list(dict.fromkeys(prefix + meta + counts + characters + series + artists + user_general))
    room = max(0, LIMITS[detail] - len(fixed))
    image_fill = [t for t in image_general + styles if t not in fixed][:room]
    general_block = (user_general + image_fill) if lead else (image_fill + user_general)
    tags = list(dict.fromkeys(prefix + meta + counts + characters + series + artists + general_block))
    tag_set = {x["tag"] for x in selected} | set(user_names)
    analysis = dict(analysis)
    caption_facts = []
    aliases = {"1girl": "woman", "1boy": "man", "girl": "woman", "boy": "man", "city": "urban", "building": "storefronts"}
    for sentence in re.split(r"(?<=[.!?])\s+", analysis.get("caption", "")):
        sentence = sentence.strip().rstrip(".!?")
        if not sentence:
            continue
        # Recoloured attributes are edited in place instead of dropping the sentence.
        for old, new in replacements.items():
            sentence = direction_plan.recolor(sentence, old, new)
        lowered = normalize(sentence)
        if fidelity == "strict":
            for garment in CLOTHING:
                if not any(tag == garment or tag.endswith(" " + garment) for tag in tag_set):
                    sentence = re.sub(r"\b(?:in|wearing|dressed in)\s+(?:(?:a|an|the|red|blue|green|black|white|yellow|pink|purple|orange|brown|gray|grey)\s+)*" + garment + r"\b", "", sentence, flags=re.IGNORECASE)
            sentence = re.sub(r"\s+", " ", sentence).strip()
            # A still image does not establish motion.
            sentence = re.sub(r"\bmoves along\b", "is on", sentence, flags=re.IGNORECASE)
            lowered = normalize(sentence)
        support = [tag for tag in tag_set if not COUNTS.fullmatch(tag) and
                   (re.search(r"\b" + re.escape(tag) + r"\b", lowered) or
                    tag in aliases and re.search(r"\b" + aliases[tag] + r"\b", lowered))]
        support += [t for t in counts if t in aliases and re.search(r"\b" + aliases[t] + r"\b", lowered)]
        visual_property = re.search(r"\b(light(?:ing)?|shadow|illuminat\w*|palette|color\w*|paint\w*|artwork|illustrat\w*)\b", lowered)
        if include_visual_details and (support or visual_property or fidelity == "balanced" or analysis.get("directed")):
            caption_facts.append({"field": "visual_description", "text": sentence,
                                  "evidence": support,
                                  "confidence": None,
                                  "confidence_basis": "Unscored " + analysis.get("source_model", "MiniCPM") + " observation; matching PixAI tags corroborate nouns only"})
    analysis["facts"] = analysis.get("facts", []) + caption_facts
    # Slots the user changed are checked too, so the caption cannot contradict the edit.
    changed_slots = [slot for slot in direction_plan.SLOTS.values() if slot & set(user_names)]
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
        # Only the clauses that contradict the direction go; the rest of the sentence stays.
        stale = lambda clause: (direction_plan.contradicts(clause, user_names)
                                or any(direction_plan.mentions(clause, tag) for tag in replaced)
                                or any(re.search(r"\b" + re.escape(o) + r"\b", clause) for slot in changed_slots
                                       for o in slot - set(user_names)))
        if stale(lowered):
            kept = direction_plan.prune(text, lambda clause: stale(normalize(clause)))
            if not kept:
                rejected.append({**item, "reason": "visual prose describes something the user direction replaced or removed"})
                continue
            item = {**item, "text": kept}
            text, lowered = kept, normalize(kept)
        if lowered in tag_set or lowered in {normalize(f["text"]) for f in facts}:
            continue
        facts.append(item)
    fact_limit = {"concise": 2, "balanced": 5, "detailed": 9}[detail]
    image_prose = [x["text"] + "." for x in facts[:fact_limit]]
    if direction_mode == "replace description":
        image_prose = []
    # A caption written with the direction already says what your sentences asked for.
    caption_text = " ".join(image_prose)
    pending = [p for p in user_prose if not (analysis.get("directed") and direction_plan.covered(p, caption_text))]
    sentences = (pending + image_prose) if lead else (image_prose + pending)
    if direction_mode == "replace description":
        sentences = user_prose
    # Requested tags the description still never mentions get one plain sentence, so text and tags agree.
    if lead and (include_visual_details or user_prose):
        extra = direction_plan.missing_sentence(user, " ".join(sentences), counts)
        if extra:
            sentences = sentences + [extra]
    prose = " ".join(sentences)
    positive = ", ".join(tags) + ("\n\n" + prose if prose else "")
    negative = NEGATIVE
    if variant == "Aesthetic":
        negative = ", ".join(t for t in NEGATIVE.split(", ") if not t.startswith("score_"))
    if variant == "Turbo":
        negative = ""
    return positive, negative, {"variant": variant, "selected_tags": selected, "retained_facts": facts[:fact_limit],
                                "rejected": rejected, "safety_tag": safety_tag,
                                "negative_active": variant != "Turbo", "analysis": analysis,
                                "direction": {"mode": direction_mode, "tags": user, "prose": user_prose,
                                              "integrated_prose": [p for p in user_prose if p not in pending],
                                              "excluded": sorted(excluded | excluded_names),
                                              "replaced": dict(sorted(replacements.items())),
                                              "caption_directed": bool(analysis.get("directed"))},
                                "limits": "Model confidence is uncalibrated. Evidence text is model-generated, not independent verification."}
