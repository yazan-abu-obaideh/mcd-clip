import abc

from transformers import CLIPProcessor, CLIPTokenizerFast, CLIPModel

_DEVICE = "cuda"
_MODEL_ID = "openai/clip-vit-base-patch32"
_CLIP_PROCESSOR = CLIPProcessor.from_pretrained(_MODEL_ID)
_TOKENIZER = CLIPTokenizerFast.from_pretrained(_MODEL_ID)
_MODEL = CLIPModel.from_pretrained(_MODEL_ID)  # .to(device)


class ClipEmbeddingCalculator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def from_text(self, text: str):
        pass

    @abc.abstractmethod
    def from_image_bytes(self, image_bytes: bytes):
        pass

    @abc.abstractmethod
    def from_image_path(self, image_path: str):
        pass


class ClipEmbeddingCalculatorImpl(ClipEmbeddingCalculator):
    def from_text(self, text: str):
        return _MODEL.get_text_features(**_TOKENIZER(text, return_tensors="pt")).detach().numpy()

    def from_image_bytes(self, image_bytes: bytes):
        pass

    def from_image_path(self, image_path: str):
        pass
