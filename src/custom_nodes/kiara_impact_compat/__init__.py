class ImpactExecutionOrderController:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": ("CONDITIONING",),
                "value": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "IMAGE")
    RETURN_NAMES = ("signal", "value")
    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Util"

    def doit(self, signal, value):
        return (signal, value)


NODE_CLASS_MAPPINGS = {
    "ImpactExecutionOrderController": ImpactExecutionOrderController,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImpactExecutionOrderController": "Execution Order Controller",
}
