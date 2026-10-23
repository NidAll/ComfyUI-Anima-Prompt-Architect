import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architect.fusion import fuse, normalize


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.tags = {"general": {"1girl": .99, "solo": .95, "black_hair": .9, "outdoors": .9,
                                  "indoors": .4, "holding_umbrella": .92},
                     "character": {"invented_character": .4}, "style": {"guessed_artist": .9},
                     "rating": {"rating:g": .95}}

    def test_official_variant_contract(self):
        base, negative, _ = fuse(self.tags, {"facts": []})
        self.assertTrue(base.startswith("masterpiece, best quality, score_7, safe, 1girl"))
        self.assertIn("score_1", negative)
        aesthetic, negative, _ = fuse(self.tags, {"facts": []}, variant="Aesthetic")
        self.assertNotIn("score_", aesthetic + negative)
        _, negative, data = fuse(self.tags, {"facts": []}, variant="Turbo")
        self.assertEqual(negative, "")
        self.assertFalse(data["negative_active"])

    def test_identity_and_artist_are_not_invented(self):
        text, _, _ = fuse(self.tags, {"facts": []}, artist="@Known_Artist, @Second_Artist")
        self.assertIn("@known artist", text)
        self.assertIn("@second artist", text)
        self.assertNotIn("@@", text)
        self.assertNotIn("invented character", text)
        self.assertNotIn("guessed artist", text)

    def test_contradictions_and_evidence(self):
        analysis = {"caption": "A woman stands indoors. Soft side lighting illuminates the scene."}
        text, _, data = fuse(self.tags, analysis)
        self.assertNotIn("indoors", text.lower())
        self.assertIn("Soft side lighting", text)
        self.assertTrue(data["rejected"])

    def test_uncorroborated_content_is_discarded(self):
        text, _, _ = fuse(self.tags, {"caption": "A dragon guards a treasure."})
        self.assertNotIn("dragon", text)

    def test_tag_normalization(self):
        self.assertEqual(normalize("Blue_Hair"), "blue hair")
        self.assertEqual(normalize("score_7"), "score_7")

    def test_subject_counts(self):
        self.tags["general"].update({"2girls": .8, "multiple_girls": .7})
        text, _, _ = fuse(self.tags, {"facts": []})
        self.assertIn("1girl", text)
        self.assertNotIn("2girls", text)

    def test_control_validation(self):
        with self.assertRaises(ValueError):
            fuse({}, {}, variant="unknown")

    def test_caption_keeps_lighting_without_fabricated_confidence(self):
        text, _, data = fuse(self.tags, {"caption": "A woman stands outdoors. Soft lighting creates gentle shadows."})
        self.assertIn("Soft lighting", text)
        self.assertIsNone(data["retained_facts"][-1]["confidence"])

    def test_visual_details_can_be_disabled(self):
        text, _, _ = fuse(self.tags, {"caption": "A woman stands outdoors."}, include_visual_details=False)
        self.assertNotIn("\n\n", text)

    def test_redundant_clothing_tags_removed(self):
        tags = {"general": {"coat": .9, "raincoat": .9, "yellow_raincoat": .9, "yellow_coat": .9}}
        text, _, _ = fuse(tags, {})
        self.assertTrue(text.endswith("yellow raincoat"))

    def test_caption_does_not_add_uncorroborated_clothing(self):
        tags = {"general": {"1girl": .9, "apron": .85}}
        text, _, _ = fuse(tags, {"caption": "A woman in overalls stands beside the counter."})
        self.assertNotIn("overalls", text)
        self.assertIn("A woman stands beside", text)


if __name__ == "__main__":
    unittest.main()
