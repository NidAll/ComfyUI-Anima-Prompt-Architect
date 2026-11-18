from comfy_api.latest import io

from .architect import loading, runtime


class AnimaPromptArchitect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        models = loading.model_choices()
        taggers = [m for m in models if m.rsplit("/", 1)[-1].startswith("pixai-")] or ["pixai-tagger-v1.0-bf16.safetensors"]
        visual = loading.visual_model_choices() or ["minicpm-v4.7-1b-heretic-int8-convrot.safetensors"]
        return io.Schema(node_id="AnimaPromptArchitect", display_name="Anima Prompt Architect",
            category="Anima/prompt", description="Sequential PixAI BF16 tags and native MiniCPM or LFM3-VL INT8 visual analysis, formatted for official Anima profiles.",
            inputs=[io.Image.Input("image"), io.Combo.Input("tagger_model", options=taggers),
                    io.Combo.Input("visual_model", options=visual),
                    io.Combo.Input("variant", options=["Base", "Aesthetic", "Turbo"], default="Base"),
                    io.Combo.Input("detail", options=["concise", "balanced", "detailed"], default="balanced"),
                    io.Combo.Input("fidelity", options=["strict", "balanced"], default="strict"),
                    io.Float.Input("general_threshold", default=.35, min=.01, max=1., step=.01),
                    io.Float.Input("identity_threshold", default=.65, min=.01, max=1., step=.01),
                    io.Float.Input("style_threshold", default=.65, min=.01, max=1., step=.01),
                    io.Boolean.Input("include_visual_details", default=True),
                    io.Combo.Input("safety", options=["auto", "omit", "safe", "sensitive", "nsfw", "explicit"]),
                    io.String.Input("artist", default="", tooltip="Optional explicit artist name(s), comma-separated. Never inferred from visual style."),
                    io.Int.Input("max_new_tokens", default=512, min=64, max=2048, step=64),
                    io.Int.Input("max_slices", default=4, min=1, max=9),
                    io.String.Input("direction", default="", multiline=True, optional=True,
                                    tooltip="Optional changes to the cloned image. Words and sentences become PixAI tags in their Anima slot and replace what they contradict (hair colour, time, expression, outfit...); '-tag' or 'no tag' removes one."),
                    io.Combo.Input("direction_mode", options=["lead", "follow", "replace description"], default="lead", optional=True,
                                   tooltip="lead (apply): the description is written with your changes applied and your tags lead. follow (add): image caption kept, your tags and sentences added after. replace description: keep the tags, use only your sentences as description."),
                    io.String.Input("exclude_tags", default="", optional=True,
                                    tooltip="Comma-separated tags to drop from the image analysis.")],
            outputs=[io.String.Output("positive"), io.String.Output("negative"),
                     io.String.Output("analysis_json"), io.String.Output("raw_tags_json")])

    @classmethod
    def execute(cls, image, tagger_model, visual_model, variant, detail, fidelity, general_threshold,
                identity_threshold, style_threshold, include_visual_details, safety, artist, max_new_tokens, max_slices,
                direction="", direction_mode="lead", exclude_tags=""):
        values = runtime.architect(image, loading.resolve_bundle(tagger_model), loading.resolve_bundle(visual_model),
            variant=variant, detail=detail, fidelity=fidelity,
            thresholds={"general": general_threshold, "character": identity_threshold,
                        "copyright": identity_threshold, "style": style_threshold},
            include_visual_details=include_visual_details, safety=safety, artist=artist,
            max_new_tokens=max_new_tokens, max_slices=max_slices,
            direction=direction or "", direction_mode=direction_mode or "lead", exclude_tags=exclude_tags or "")
        # The final prompt is shown on the node and recorded in history for front ends.
        return io.NodeOutput(*values, ui={"text": [values[0]]})


class AnimaPromptUpsampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        models = loading.model_choices()
        taggers = [m for m in models if m.rsplit("/", 1)[-1].startswith("pixai-")] or ["pixai-tagger-v1.0-bf16.safetensors"]
        language = [m for m in loading.visual_model_choices() if "lfm3" in m.lower()] or ["lfm3-vl-2b-int8-convrot.safetensors"]
        return io.Schema(node_id="AnimaPromptUpsampler", display_name="Anima Prompt Upsampler",
            category="Anima/prompt", is_output_node=True, description="Expands a short idea into a full Anima prompt in the official order (quality, safety, count, character, series, artist, tags, then prose). Text only: LFM3 writes, the PixAI vocabulary checks every tag.",
            inputs=[io.String.Input("prompt", multiline=True, default="", tooltip="Your idea: a few words, tags or sentences. Your tags always win over added ones; '-tag' or 'no tag' removes one; '@name' adds an artist."),
                    io.Combo.Input("language_model", options=language),
                    io.Combo.Input("tagger_model", options=taggers, tooltip="Only its tag vocabulary is read; the tagger is not run."),
                    io.Combo.Input("variant", options=["Base", "Aesthetic", "Turbo"], default="Turbo"),
                    io.Combo.Input("detail", options=["concise", "balanced", "detailed"], default="balanced"),
                    io.Combo.Input("safety", options=["auto", "omit", "safe", "sensitive", "nsfw", "explicit"], default="auto"),
                    io.String.Input("artist", default="", tooltip="Optional explicit artist name(s), comma-separated. Never invented."),
                    io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                    io.Float.Input("temperature", default=0.6, min=0.0, max=1.5, step=0.05, tooltip="0 always gives the same expansion; higher values vary the added details."),
                    io.Int.Input("max_new_tokens", default=384, min=128, max=1024, step=32)],
            outputs=[io.String.Output("positive"), io.String.Output("negative"), io.String.Output("analysis_json")])

    @classmethod
    def execute(cls, prompt, language_model, tagger_model, variant, detail, safety, artist, seed, temperature, max_new_tokens):
        positive, negative, analysis = runtime.upsample(prompt, loading.resolve_bundle(tagger_model),
            loading.resolve_bundle(language_model), variant=variant, detail=detail, safety=safety, artist=artist,
            max_new_tokens=max_new_tokens, seed=seed, temperature=temperature)
        return io.NodeOutput(positive, negative, analysis, ui={"text": [positive], "inspection": [analysis]})
