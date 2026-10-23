from comfy_api.latest import io

from .architect import loading, runtime


class AnimaPromptArchitect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        models = loading.model_choices()
        taggers = [m for m in models if m.rsplit("/", 1)[-1].startswith("pixai-")] or ["pixai-tagger-v1.0-bf16.safetensors"]
        visual = [m for m in models if m.rsplit("/", 1)[-1].startswith("minicpm-")] or ["minicpm-v4.7-1b-heretic-int8-convrot.safetensors"]
        return io.Schema(node_id="AnimaPromptArchitect", display_name="Anima Prompt Architect",
            category="Anima/prompt", description="Sequential PixAI BF16 tags and native MiniCPM INT8 visual analysis, formatted for official Anima profiles.",
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
                    io.Int.Input("max_slices", default=4, min=1, max=9)],
            outputs=[io.String.Output("positive"), io.String.Output("negative"),
                     io.String.Output("analysis_json"), io.String.Output("raw_tags_json")])

    @classmethod
    def execute(cls, image, tagger_model, visual_model, variant, detail, fidelity, general_threshold,
                identity_threshold, style_threshold, include_visual_details, safety, artist, max_new_tokens, max_slices):
        values = runtime.architect(image, loading.resolve_bundle(tagger_model), loading.resolve_bundle(visual_model),
            variant=variant, detail=detail, fidelity=fidelity,
            thresholds={"general": general_threshold, "character": identity_threshold,
                        "copyright": identity_threshold, "style": style_threshold},
            include_visual_details=include_visual_details, safety=safety, artist=artist,
            max_new_tokens=max_new_tokens, max_slices=max_slices)
        return io.NodeOutput(*values)
