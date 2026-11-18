from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architect import upsample


GENERAL = ["1girl", "2girls", "1boy", "solo", "long hair", "short hair", "red hair", "black hair", "blue eyes",
           "witch hat", "hat", "smile", "frown", "night", "day", "rain", "sunlight", "blue sky", "outdoors",
           "city", "street", "holding umbrella", "umbrella", "cowboy shot", "standing", "black dress", "cat ears",
           "no humans", "scenery", "forest"]
VOCAB = {**{t: "general" for t in GENERAL}, "hatsune miku": "character", "vocaloid": "copyright",
         "some artist": "style", "highres": "meta", "rating:g": "rating"}

ANSWER = """Rating: safe
Tags: 1girl, solo, long hair, black hair, blue eyes, witch hat, smile, day, blue sky, sunlight, outdoors, city, made up tag, masterpiece, hatsune miku, cowboy shot
Description: A witch with long black hair walks through the city under a sunny blue sky. She smiles at the viewer. Warm light falls on the street."""


class UpsampleTests(unittest.TestCase):
    def test_parse_response_reads_three_labelled_lines(self):
        parsed = upsample.parse_response("**Rating:** sensitive\nTags: 1girl, smile\nDescription: One. Two.\nThree.")
        self.assertEqual(parsed["rating"], "sensitive")
        self.assertEqual(parsed["tags"], ["1girl", "smile"])
        self.assertEqual(parsed["description"], "One. Two. Three.")

    def test_parse_response_without_labels_is_empty(self):
        self.assertEqual(upsample.parse_response("<think>hmm</think>sorry"), {"rating": "", "tags": [], "description": ""})

    def test_official_order_and_vocabulary_filter(self):
        positive, negative, info = upsample.compose("witch girl in a city", upsample.parse_response(ANSWER), VOCAB,
                                                    variant="Base")
        tags = positive.split("\n\n")[0].split(", ")
        self.assertEqual(tags[:5], ["masterpiece", "best quality", "score_7", "safe", "1girl"])
        self.assertNotIn("made up tag", tags)
        self.assertNotIn("hatsune miku", tags, "characters the idea does not name are dropped")
        self.assertEqual(tags.count("masterpiece"), 1)
        self.assertIn("witch hat", tags)
        self.assertTrue(negative.startswith("worst quality"))
        prose = positive.split("\n\n")[1]
        self.assertGreaterEqual(prose.count("."), 2)

    def test_user_words_win_over_model_details(self):
        positive, _, info = upsample.compose("red hair, night, rain, -hat", upsample.parse_response(ANSWER), VOCAB)
        tags = positive.split("\n\n")[0].split(", ")
        for kept in ("red hair", "night", "rain"):
            self.assertIn(kept, tags)
        for gone in ("black hair", "day", "blue sky", "sunlight", "witch hat"):
            self.assertNotIn(gone, tags)
        # The sunny sentence contradicts night/rain and goes; the user's tags come before added ones.
        self.assertNotIn("sunny", positive)
        self.assertLess(tags.index("red hair"), tags.index("long hair"))

    def test_named_character_series_and_artist_slots(self):
        answer = {"rating": "safe", "tags": ["1girl", "hatsune miku", "vocaloid", "long hair"], "description": ""}
        positive, _, info = upsample.compose("hatsune miku from vocaloid singing, @Some Artist", answer, VOCAB)
        tags = positive.split("\n\n")[0].split(", ")
        self.assertEqual(tags[4:8], ["1girl", "hatsune miku", "vocaloid", "@some artist"])
        self.assertFalse(info["used_model_text"])
        self.assertGreaterEqual(len(info["description"]), 1)

    def test_variants_and_safety(self):
        answer = upsample.parse_response(ANSWER.replace("Rating: safe", "Rating: nsfw"))
        positive, negative, _ = upsample.compose("a girl", answer, VOCAB, variant="Aesthetic")
        self.assertTrue(positive.startswith("masterpiece, best quality, nsfw, 1girl"))
        self.assertNotIn("score_", negative)
        positive, negative, _ = upsample.compose("a girl, safe", answer, VOCAB, variant="Turbo")
        self.assertIn("score_7, safe, 1girl", positive)
        self.assertEqual(negative, "")
        positive, _, _ = upsample.compose("a girl", answer, VOCAB, safety="omit")
        self.assertTrue(positive.startswith("masterpiece, best quality, score_7, 1girl"))

    def test_counts_and_solo(self):
        answer = {"rating": "safe", "tags": ["1girl", "solo", "smile"], "description": "Two friends laugh. They hold hands."}
        positive, _, _ = upsample.compose("2girls", answer, VOCAB)
        tags = positive.split("\n\n")[0].split(", ")
        self.assertIn("2girls", tags)
        self.assertNotIn("1girl", tags)
        self.assertNotIn("solo", tags)

    def test_live_findings_idea_not_copied_tags_mined_and_two_sentences(self):
        answer = {"rating": "safe", "tags": ["1girl", "smile"],
                  "description": "A girl with long black hair and blue eyes stands on a city street at night."}
        positive, _, info = upsample.compose("a witch girl walking through the city at night", answer, VOCAB)
        head, prose = positive.split("\n\n")
        tags = head.split(", ")
        self.assertNotIn("a witch girl walking", prose, "the raw idea is not repeated as a sentence")
        for mined in ("long hair", "black hair", "blue eyes", "street"):
            self.assertIn(mined, tags, "tags the description names are added")
        self.assertGreaterEqual(len(info["description"]), 2)
        self.assertTrue(info["description"][-1].startswith("The picture also shows"))

    def test_solo_follows_the_count(self):
        answer = {"rating": "safe", "tags": ["smile", "solo", "1girl"], "description": "A girl smiles. She waves."}
        tags = upsample.compose("a girl", answer, VOCAB)[0].split("\n\n")[0].split(", ")
        self.assertEqual(tags[4:6], ["1girl", "solo"])

    def test_capitalized_user_sentence_is_kept_when_not_covered(self):
        answer = {"rating": "safe", "tags": ["1girl"], "description": "A girl stands in a forest. Light falls through the trees."}
        positive, _, _ = upsample.compose("A cat ears girl. She holds an umbrella in the snow.", answer, VOCAB)
        self.assertIn("She holds an umbrella in the snow.", positive)

    def test_phrase_idea_and_partial_words(self):
        vocab = {**VOCAB, "cherry blossoms": "general", "cherry": "general", "tree": "general", "armor": "general",
                 "samurai": "general", "cyberpunk": "general"}
        answer = {"rating": "safe", "tags": ["1boy", "armor", "cherry blossoms"],
                  "description": "A boy in armor walks under cherry blossom trees. Petals drift around him."}
        positive, _, _ = upsample.compose("cyberpunk samurai boy", answer, vocab)
        tags = positive.split("\n\n")[0].split(", ")
        self.assertIn("cyberpunk", tags)
        self.assertIn("samurai", tags)
        self.assertNotIn("cyberpunk samurai boy", tags)
        self.assertNotIn("cyberpunk samurai boy", positive.split("\n\n")[1])
        self.assertIn("cherry blossoms", tags)
        self.assertNotIn("cherry", tags, "one word of a listed tag is not mined on its own")
        self.assertIn("tree", tags)

    def test_request_contains_idea_and_rejects_bad_settings(self):
        self.assertIn("Idea: a fox spirit", upsample.request("a fox spirit", "detailed"))
        with self.assertRaises(ValueError):
            upsample.compose("x", {}, VOCAB, variant="SDXL")


if __name__ == "__main__":
    unittest.main()
