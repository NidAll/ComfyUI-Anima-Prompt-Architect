from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architect import direction
from architect.fusion import fuse, parse_direction


# The tagger returns a score for every tag it knows; low scores still define the vocabulary.
VOCAB = ["red hair", "long hair", "short hair", "night", "day", "starry sky", "blue sky", "sunlight", "smile",
         "frown", "rain", "holding umbrella", "umbrella", "school uniform", "white dress", "black dress",
         "sword", "holding sword", "cat ears", "looking up", "hat", "witch hat", "2girls", "1girl", "solo",
         "outdoors", "indoors", "blonde hair", "green eyes"]


class DirectionTests(unittest.TestCase):
    def setUp(self):
        general = {tag.replace(" ", "_"): .01 for tag in VOCAB}
        general.update({"1girl": .99, "solo": .95, "blonde_hair": .9, "long_hair": .9, "day": .8,
                        "outdoors": .9, "smile": .7, "white_dress": .85, "blue_sky": .8, "witch_hat": .7})
        self.tags = {"general": general, "rating": {"rating:g": .95}, "character": {"hatsune_miku": .01},
                     "copyright": {"vocaloid": .01}}
        self.analysis = {"caption": "A woman with long blonde hair and a white dress smiles in a meadow. "
                                    "Warm daylight lighting falls across the grass under a blue sky."}

    def head(self, text):
        return text.split("\n\n")[0].split(", ")

    def test_parse_splits_tags_prose_and_removals(self):
        tags, prose, removed = parse_direction("red hair, night, -smile\nShe looks up at a starry sky.")
        self.assertEqual(tags[:2], ["red hair", "night"])
        self.assertEqual(prose, ["She looks up at a starry sky."])
        self.assertEqual(removed, ["smile"])

    def test_phrases_resolve_to_vocabulary_tags(self):
        vocab = direction.vocabulary(self.tags)
        self.assertEqual(direction.resolve("redhead", vocab)[:2], ("red hair", "general"))
        self.assertEqual(direction.resolve("smiling", vocab)[0], "smile")
        self.assertEqual(direction.resolve("two girls", vocab)[0], "2girls")
        self.assertEqual(direction.resolve("hatsune_miku", vocab)[1], "character")
        self.assertFalse(direction.resolve("glowing runes", vocab)[2])

    def test_sentences_become_tags_and_negations_remove(self):
        found, removed = direction.sentence_tags("She holds a sword in the rain without a hat.", direction.vocabulary(self.tags))
        self.assertIn("sword", found)
        self.assertIn("rain", found)
        self.assertEqual(removed, ["hat"])

    def test_empty_direction_keeps_previous_output(self):
        self.assertEqual(fuse(self.tags, self.analysis, detail="detailed")[0],
                         fuse(self.tags, self.analysis, detail="detailed", direction="")[0])

    def test_changes_replace_by_attribute_and_keep_anima_order(self):
        text, _, data = fuse(self.tags, self.analysis, detail="detailed", direction="red hair, night, frowning")
        head = self.head(text)
        self.assertEqual(head[:7], ["masterpiece", "best quality", "score_7", "safe", "1girl", "red hair", "night"])
        for gone in ("blonde hair", "day", "smile", "blue sky"):
            self.assertNotIn(gone, head)
        self.assertIn("frown", head)
        self.assertIn("long hair", head)  # length is another slot, so it survives
        self.assertEqual(data["direction"]["replaced"]["blonde hair"], "red hair")
        self.assertEqual(data["direction"]["replaced"]["blue sky"], "night")

    def test_caption_is_recoloured_not_dropped(self):
        analysis = {"caption": "A woman with long blonde hair stands in a field of flowers."}
        text = fuse(self.tags, analysis, detail="detailed", direction="red hair")[0]
        self.assertIn("long red hair", text)
        self.assertNotIn("blonde", text)

    def test_contradicting_caption_is_dropped(self):
        text = fuse(self.tags, self.analysis, detail="detailed", direction="night")[0]
        self.assertNotIn("daylight", text.split("\n\n")[-1])

    def test_outfit_replaces_clothes(self):
        text, _, data = fuse(self.tags, self.analysis, detail="detailed", direction="school uniform")
        self.assertIn("school uniform", self.head(text))
        self.assertNotIn("white dress", text)
        self.assertEqual(data["direction"]["replaced"]["white dress"], "school uniform")

    def test_removal_covers_specific_tags(self):
        text = fuse(self.tags, self.analysis, detail="detailed", direction="-hat")[0]
        self.assertNotIn("witch hat", text)

    def test_identity_goes_in_its_slot(self):
        head = self.head(fuse(self.tags, {"facts": []}, direction="hatsune miku, cat ears", artist="someone")[0])
        self.assertLess(head.index("1girl"), head.index("hatsune miku"))
        self.assertLess(head.index("hatsune miku"), head.index("@someone"))
        self.assertLess(head.index("@someone"), head.index("cat ears"))

    def test_rating_word_sets_safety(self):
        head = self.head(fuse(self.tags, {"facts": []}, direction="sensitive, red hair")[0])
        self.assertIn("sensitive", head[:4])
        self.assertNotIn("safe", head)

    def test_directed_caption_absorbs_user_sentences(self):
        analysis = {"caption": "A woman with long red hair looks up at a starry sky at night.", "directed": "x"}
        text, _, data = fuse(self.tags, analysis, detail="detailed",
                             direction="red hair, night\nShe looks up at a starry sky.")
        self.assertEqual(text.count("starry sky"), 2)  # once as a tag, once in the caption
        self.assertEqual(data["direction"]["integrated_prose"], ["She looks up at a starry sky."])
        self.assertIn("starry sky", self.head(text))
        self.assertIn("looking up", self.head(text))

    def test_undirected_sentences_lead_the_description(self):
        text = fuse(self.tags, self.analysis, detail="detailed", direction="She holds a sword.")[0]
        self.assertTrue(text.split("\n\n")[1].startswith("She holds a sword."))
        self.assertIn("sword", self.head(text))

    def test_edit_list_names_what_changes(self):
        from architect.fusion import direction_brief
        brief = direction_brief(self.tags, "red hair, crying, rain, -hat\nShe holds a sword.")
        self.assertIn("replace 'blonde hair' with 'red hair'", brief)
        self.assertIn("replace 'smile' with 'crying'", brief)
        self.assertIn("replace 'blue sky' with 'rain'", brief)
        self.assertIn("remove 'witch hat'", brief)
        self.assertTrue(brief.endswith("She holds a sword"))

    def test_only_stale_clauses_are_pruned(self):
        analysis = {"caption": "A woman with blonde hair wears a white dress, a witch hat and gold bracelets, standing by a fountain."}
        text = fuse(self.tags, analysis, detail="detailed", fidelity="balanced", direction="-hat")[0]
        prose = text.split("\n\n")[1]
        self.assertNotIn("hat", prose)
        self.assertIn("fountain", prose)
        self.assertIn("gold bracelets", prose)

    def test_unmentioned_changes_get_a_sentence(self):
        analysis = {"caption": "A woman with long hair stands by a wall."}
        text = fuse(self.tags, analysis, detail="detailed", fidelity="balanced",
                    direction="crying, rain, holding umbrella, short hair")[0]
        self.assertTrue(text.endswith("She is crying and holding an umbrella, she has short hair, and it is raining."))
        none = fuse(self.tags, {"caption": "A crying woman holds an umbrella in the rain; her short hair is wet."},
                    detail="detailed", fidelity="balanced", direction="crying, rain, holding umbrella, short hair")[0]
        self.assertNotIn("She is crying and", none)

    def test_follow_puts_user_after_image(self):
        head = self.head(fuse(self.tags, self.analysis, direction="cat ears", direction_mode="follow")[0])
        self.assertEqual(head[-1], "cat ears")

    def test_replace_description_uses_only_user_prose(self):
        text = fuse(self.tags, self.analysis, detail="detailed", direction="She dances in the rain.",
                    direction_mode="replace description")[0]
        self.assertTrue(text.endswith("She dances in the rain."))
        self.assertNotIn("daylight", text)

    def test_exclusions(self):
        text = fuse(self.tags, self.analysis, detail="detailed", exclude_tags="white dress, smile")[0]
        self.assertNotIn("white dress", text)
        self.assertNotIn("smile", self.head(text))

    def test_user_tags_survive_tag_limit(self):
        tags = ", ".join(f"custom tag {i}" for i in range(30))
        self.assertIn("custom tag 29", fuse(self.tags, {"facts": []}, detail="concise", direction=tags)[0])

    def test_user_count_replaces_solo(self):
        head = self.head(fuse(self.tags, {"facts": []}, direction="two girls")[0])
        self.assertEqual(head[4], "2girls")
        self.assertNotIn("1girl", head)
        self.assertNotIn("solo", head)

    def test_brief_rewrite_request_and_acceptance(self):
        brief = direction.brief("red hair, -smile\nShe looks up.")
        self.assertEqual(brief, "red hair; remove smile; She looks up")
        self.assertIn(brief, direction.rewrite_request("A girl smiles.", brief))
        original = "A woman with brown hair smiles in a sunny park under a clear blue sky."
        self.assertEqual(direction.accept_rewrite(original, "Here is the rewritten description: A woman with red hair looks up in a park."),
                         "A woman with red hair looks up in a park.")
        self.assertEqual(direction.accept_rewrite(original, "Ok."), "")
        self.assertEqual(direction.accept_rewrite(original, "Unlike the original image, her hair is red now."), "")

    def test_caption_words_that_contradict_are_dropped(self):
        analysis = {"caption": "She stands by a wall with plants. The scene is set outdoors during daytime with warm light."}
        text = fuse(self.tags, analysis, detail="detailed", fidelity="balanced", direction="night")[0]
        self.assertNotIn("daytime", text)
        self.assertIn("wall with plants", text)

    def test_removal_covers_compound_words(self):
        tags = dict(self.tags, general={**self.tags["general"], "handbag": .9, "bag": .9})
        self.assertNotIn("handbag", fuse(tags, {"facts": []}, direction="-bag")[0])

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            fuse(self.tags, {"facts": []}, direction_mode="sideways")


if __name__ == "__main__":
    unittest.main()
