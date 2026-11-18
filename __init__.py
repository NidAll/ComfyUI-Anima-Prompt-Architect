from comfy_api.latest import ComfyExtension

from .nodes import AnimaPromptArchitect, AnimaPromptUpsampler


class AnimaPromptArchitectExtension(ComfyExtension):
    async def get_node_list(self):
        return [AnimaPromptArchitect, AnimaPromptUpsampler]


async def comfy_entrypoint():
    return AnimaPromptArchitectExtension()
