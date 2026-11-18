from pathlib import Path
import sys
import importlib.util
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architect import grounding, upsample

GENERAL = ["1girl", "2girls", "solo", "long hair", "aqua hair", "green hair", "black hair", "twintails", "fern",
           "witch hat", "smile", "night", "rain", "full body", "upper body", "from above", "from below", "from side",
           "umbrella", "stage", "singing", "chain", "fence", "aqua eyes", "blue eyes", "close-up"]
VOCAB = {**{t: "general" for t in GENERAL},
         "hatsune miku": "character", "fern (sousou no frieren)": "character", "frieren": "character",
         "saber (fate)": "character", "saber alter": "character", "link": "character",
         "vocaloid": "copyright", "sousou no frieren": "copyright", "fate/stay night": "copyright",
         "bkub": "style", "sea": "style",
         "watercolor (medium)": "meta", "1990s (style)": "general"}
ORDER = {t: i for i, t in enumerate(VOCAB)}


def recognise(text):
    return grounding.recognize(text, VOCAB, ORDER)


class RecognitionTests(unittest.TestCase):
    def test_characters_by_full_name_qualifier_and_lowercase_rules(self):
        self.assertEqual([c["tag"] for c in recognise("Hatsune Miku singing")["characters"]], ["hatsune miku"])
        self.assertEqual([c["tag"] for c in recognise("hatsune miku singing")["characters"]], ["hatsune miku"])
        # "fern" is also a plant tag; the named series makes it the character, and Frieren stays the series.
        self.assertEqual([c["tag"] for c in recognise("Fern from Frieren in the rain")["characters"]],
                         ["fern (sousou no frieren)"])
        self.assertEqual(recognise("a girl among fern leaves")["characters"], [])
        # Everyday words stay words in lowercase; capitalised names are names.
        self.assertEqual(recognise("a chain link fence")["characters"], [])
        self.assertEqual([c["tag"] for c in recognise("Link with a sword")["characters"]], ["link"])

    def test_same_named_character_follows_series_named_nearby(self):
        found = recognise("Saber from Fate stay night")
        self.assertEqual(found["characters"][0]["tag"], "saber (fate)")

    def test_camera_style_artist_and_positions(self):
        found = recognise("Miku on the left, low angle, full body, close-up, watercolor, 90s anime style, by Bkub")
        self.assertEqual(found["camera"], ["from below", "full body"], "written order; the first framing named wins over close-up")
        self.assertEqual(found["styles"], ["watercolor (medium)", "1990s (style)"])
        self.assertEqual(found["artists"], ["@bkub"])
        self.assertEqual(found["positions"], ["Miku on the left"])

    def test_everyday_phrases_are_not_artists_or_placements(self):
        found = recognise("a girl walking by the sea, seen from above")
        self.assertEqual(found["artists"], [])
        self.assertEqual(found["positions"], [])
        self.assertEqual(found["camera"], ["from above"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "needs torch")
class TagSpaceTests(unittest.TestCase):
    def setUp(self):
        import torch

        tags = ["aqua_hair", "green_hair", "twintails", "witch_hat", "smile",
                "hatsune_miku", "fern_(sousou_no_frieren)", "nobody",
                "vocaloid", "sousou_no_frieren", "other_show"]
        categories = ["general"] * 5 + ["character"] * 3 + ["copyright"] * 3
        torch.manual_seed(0)
        weights = torch.randn(len(tags), 16) * 0.05
        axis = torch.eye(16)
        weights[0] += axis[0] * 3; weights[2] += axis[0] * 2; weights[1] += axis[0] * 1 + axis[5] * 2
        weights[5] += axis[0] * 3                        # miku ~ aqua hair, twintails
        weights[8] += axis[0] * 3                        # vocaloid ~ miku
        weights[6] += axis[3] * 3; weights[9] += axis[3] * 1; weights[10] += axis[3] * 1.05
        self.space = grounding.TagSpace(tags, categories, weights)

    def test_series_by_clear_leader_or_qualifier(self):
        self.assertEqual(self.space.series_for("hatsune miku")[0], "vocaloid")
        # Fern's two candidates are nearly tied; the qualifier decides.
        self.assertEqual(self.space.series_for("fern (sousou no frieren)")[0], "sousou no frieren")
        self.assertEqual(self.space.series_for("unknown")[0], "")

    def test_look_keeps_one_hair_colour(self):
        look = self.space.look_of("hatsune miku")
        self.assertIn("aqua hair", look)
        self.assertNotIn("green hair", look)


class GroundedComposeTests(unittest.TestCase):
    def test_recognised_names_fill_their_slots_and_look_hints_need_agreement(self):
        idea = "Hatsune Miku singing on stage, low angle, watercolor"
        found = recognise(idea)
        known = {"found": found, "series": {"hatsune miku": "vocaloid"},
                 "looks": {"hatsune miku": ["aqua hair", "twintails", "witch hat"]}}
        answer = {"rating": "safe", "tags": ["1girl", "solo", "smile", "twintails", "aqua hair", "green hair",
                                             "from above", "stage"],
                  "description": "Miku sings on a bright stage. Her aqua twintails swing as she moves."}
        positive, _, info = upsample.compose(idea, answer, VOCAB, grounding=known)
        tags = positive.split("\n\n")[0].split(", ")
        self.assertEqual(tags[4:8], ["1girl", "solo", "hatsune miku", "vocaloid"])
        self.assertIn("from below", tags)
        self.assertNotIn("from above", tags, "your low angle wins over the model's viewpoint")
        self.assertIn("watercolor (medium)", tags)
        self.assertNotIn("witch hat", tags, "a hint the model did not agree with is not added")
        self.assertNotIn("green hair", tags, "one subject keeps one hair colour")
        added = info["tags"]["added"]
        self.assertLess(added.index("aqua hair"), added.index("smile"), "agreed look tags lead")
        self.assertEqual(info["recognised"]["series"], ["vocaloid"])

    def test_placement_survives_when_the_description_drops_it(self):
        idea = "2girls having tea, Fern on the left"
        found = recognise(idea)
        answer = {"rating": "safe", "tags": ["2girls"], "description": "Two girls drink tea. Steam rises from the cups."}
        positive, _, _ = upsample.compose(idea, answer, VOCAB, grounding={"found": found})
        self.assertIn("Fern on the left.", positive)

    def test_facts_name_character_series_look_and_placement(self):
        found = recognise("Hatsune Miku, Fern from Frieren, Fern on the right")
        lines = grounding.facts(found, {"hatsune miku": ["aqua hair"]}, {"hatsune miku": "vocaloid"})
        self.assertIn("Character: Hatsune Miku, from Vocaloid. Usually drawn with: aqua hair.", lines)
        self.assertTrue(any(line.startswith("Keep this placement exactly: Fern on the right") for line in lines))

    def test_live_findings_named_characters(self):
        # Lowercase given names, lowercase known artists; the sea is still the sea.
        found = recognise("miku watching the sunset, by bkub")
        self.assertEqual([c["tag"] for c in found["characters"]], ["hatsune miku"])
        self.assertEqual(found["artists"], ["@bkub"])
        idea = "Fern from Frieren casting a spell in the rain, watercolor, low angle"
        found = recognise(idea)
        answer = {"rating": "safe", "tags": ["1girl", "fern", "frieren", "black hair", "blue eyes", "umbrella"],
                  "description": "Fern from Frieren, with long flowing hair and piercing blue eyes, casts a spell "
                                 "in the rain. She stands with silver hair and a black robe, captured from below."}
        positive, _, info = upsample.compose(idea, answer, VOCAB, grounding={"found": found})
        head, prose = positive.split("\n\n")
        tags = head.split(", ")
        self.assertIn("fern (sousou no frieren)", tags)
        for gone in ("frieren", "fern", "low angle", "watercolor", "black hair", "blue eyes"):
            self.assertNotIn(gone, tags)
        self.assertIn("watercolor (medium)", tags)
        self.assertIn("from below", tags)
        self.assertNotIn("blue eyes", prose)
        self.assertNotIn("silver hair", prose)
        self.assertIn("with a black robe", prose)
        self.assertNotIn("has from below", prose)

    def test_comma_fragments_are_not_copied_as_sentences(self):
        idea = "Hatsune Miku and a friend having tea, full body"
        answer = {"rating": "safe", "tags": ["2girls", "tea"],
                  "description": "Two girls share tea at a small table. Steam curls from their cups."}
        positive, _, _ = upsample.compose(idea, answer, VOCAB, grounding={"found": recognise(idea)})
        prose = positive.split("\n\n")[1]
        self.assertNotIn("having tea", prose)
        self.assertNotIn("full body", prose)

    def test_colour_scrub_keeps_sentences_whole(self):
        from architect.direction import COLOR_WORDS
        banned = {"hair": set(COLOR_WORDS), "eyes": set(COLOR_WORDS)}
        cases = {
            "Saber, with short brown hair and sharp blue eyes, sits on the right.": "Saber sits on the right.",
            "Saber with silver hair and a red cloak stands.": "Saber with a red cloak stands.",
            "A girl with silver hair and blue eyes stands in the rain.": "A girl stands in the rain.",
            "She brushes her silver hair.": "She brushes her hair.",
            "A silver-haired knight waits.": "A knight waits.",
        }
        for before, after in cases.items():
            self.assertEqual(upsample._scrub_colours(before, banned), after)
        self.assertEqual(upsample._scrub_colours("Miku with aqua hair sings.", {"hair": {"red"}}),
                         "Miku with aqua hair sings.")

    def test_scenery_relations_and_anthro_details_are_not_forced(self):
        idea = "a fox girl sleeping under a tree, from above"
        answer = {"rating": "safe", "tags": ["1girl", "fur", "umbrella"],
                  "description": "A fox girl naps beneath a tree. Petals fall on her fur."}
        vocab = {**VOCAB, "fur": "general", "fox girl": "general", "tree": "general"}
        positive, _, _ = upsample.compose(idea, answer, vocab, grounding={"found": grounding.recognize(idea, vocab)})
        head, prose = positive.split("\n\n")
        self.assertNotIn("fur", head.split(", "))
        self.assertFalse(prose.startswith("A fox girl sleeping under"))
        self.assertNotIn("fur", prose)

    def test_request_carries_facts(self):
        text = upsample.request("Miku", "balanced", ["Character: Hatsune Miku."])
        self.assertIn("Known from the tag database", text)
        self.assertLess(text.index("Character: Hatsune Miku."), text.index("Idea: Miku"))


if __name__ == "__main__":
    unittest.main()
