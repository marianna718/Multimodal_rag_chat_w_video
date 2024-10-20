from typing import List
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import (
    BaseModel,
)
from utils import encode_image    #, bt_embedding_from_prediction_guard
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


class BridgeTowerEmbeddings(BaseModel, Embeddings):
    """BridgeTower embedding model"""

    # Initialize tokenizer and model for local embedding
    tokenizer = AutoTokenizer.from_pretrained("your-local-model")
    model = AutoModel.from_pretrained("your-local-model")

    @staticmethod
    def local_embedding(text: str, image: str = "") -> List[float]:
        """Create embeddings for text (and optionally images) without relying on external services.

        Args:
            text: The text to embed.
            image: The path to the image (optional).

        Returns:
            List of float values representing the embedding for the text (or text-image pair).
        """
        # For this example, we are only using text embeddings; images can be integrated later.
        inputs = BridgeTowerEmbeddings.tokenizer(text, return_tensors="pt")
        outputs = BridgeTowerEmbeddings.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the local embedding function.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = []
        for text in texts:
            embedding = self.local_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using the local embedding function.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]

    def embed_image_text_pairs(self, texts: List[str], images: List[str], batch_size=2) -> List[List[float]]:
        """Embed a list of image-text pairs using the local embedding function.

        Args:
            texts: The list of texts to embed.
            images: The list of paths to images to embed.
            batch_size: The batch size to process, defaults to 2.

        Returns:
            List of embeddings, one for each image-text pair.
        """
        # Ensure the number of texts matches the number of images
        assert len(texts) == len(images), "The length of captions should be equal to the length of images."

        embeddings = []
        for path_to_img, text in tqdm(zip(images, texts), total=len(texts)):
            # Encode the image and text using the local embedding method (or modify as needed)
            image_embedding = encode_image(path_to_img)  # Assuming this function processes the image
            text_embedding = self.local_embedding(text)

            # Combine or merge the text and image embeddings (this depends on your use case)
            # Here, we append only text embedding for simplicity
            embeddings.append(text_embedding)  # Modify this if you wish to combine text and image embeddings
        return embeddings


# class BridgeTowerEmbeddings(BaseModel, Embeddings):
#     """ BridgeTower embedding model """
        
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         """Embed a list of documents using BridgeTower.

#         Args:
#             texts: The list of texts to embed.

#         Returns:
#             List of embeddings, one for each text.
#         """
#         embeddings = []
#         for text in texts:
#             embedding = bt_embedding_from_prediction_guard(text, "")
#             embeddings.append(embedding)
#         return embeddings
    

#     tokenizer = AutoTokenizer.from_pretrained("your-local-model")
#     model = AutoModel.from_pretrained("your-local-model")

#     def local_embedding(text: str, image: str) -> List[float]:
#         # Create embeddings without relying on external services
#         inputs = tokenizer(text, return_tensors="pt")
#         outputs = model(**inputs)
#         return outputs.last_hidden_state.mean(dim=1).tolist()

#     def embed_query(self, text: str) -> List[float]:
#         """Embed a query using BridgeTower.

#         Args:
#             text: The text to embed.

#         Returns:
#             Embeddings for the text.
#         """
#         return self.embed_documents([text])[0]

#     def embed_image_text_pairs(self, texts: List[str], images: List[str], batch_size=2) -> List[List[float]]:
#         """Embed a list of image-text pairs using BridgeTower.

#         Args:
#             texts: The list of texts to embed.
#             images: The list of path-to-images to embed
#             batch_size: the batch size to process, default to 2
#         Returns:
#             List of embeddings, one for each image-text pairs.
#         """

#         # the length of texts must be equal to the length of images
#         assert len(texts)==len(images), "the len of captions should be equal to the len of images"

#         embeddings = []
#         for path_to_img, text in tqdm(zip(images, texts), total=len(texts)):
#             embedding = bt_embedding_from_prediction_guard(text, encode_image(path_to_img))
#             embeddings.append(embedding)
#         return embeddings