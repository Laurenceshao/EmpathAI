import openai
import numpy as np
from typing import List, Dict, Any

class VectorDatabase:
    """
    A vector database for storing and retrieving embeddings, including functionality for synthetic data generation.
    """
    
    def __init__(self, embedding_dim: int, api_key: str, model: str = "text-embedding-ada-002"):
        """
        Initialize the VectorDatabase with OpenAI API settings and embedding dimension.

        Args:
            embedding_dim (int): Dimension of the embeddings.
            api_key (str): OpenAI API key.
            model (str): Embedding model to use.
        """
        self.embedding_dim = embedding_dim
        openai.api_key = api_key
        self.model = model
        self.embeddings = []
        self.metadata = []

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for the given text using OpenAI's API.

        Args:
            text (str): Text to embed.

        Returns:
            np.ndarray: The generated embedding as a NumPy array.
        """
        response = openai.embeddings.create(input=text, model=self.model)
        embedding = response.data[0].embedding
        return np.array(embedding)

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Add embeddings and their associated metadata to the database.

        Args:
            embeddings (np.ndarray): Numpy array of embeddings.
            metadata (List[Dict[str, Any]]): List of metadata dictionaries.
        """
        self.embeddings.extend(embeddings)
        self.metadata.extend(metadata)

    def retrieve(self, query: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the top K similar entries based on cosine similarity.

        Args:
            query_embedding (np.ndarray): The embedding of the query.
            top_k (int): Number of top similar entries to retrieve.

        Returns:
            List[Dict[str, Any]]: Retrieved metadata of the top matching entries.
        """
        response = openai.embeddings.create(input=query, model=self.model)
        query_embedding = response.data[0].embedding
        # Normalize embeddings
        normalized_db = np.vstack(self.embeddings) / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute cosine similarity
        similarities = np.dot(normalized_db, normalized_query)
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [self.metadata[i] for i in top_indices]

    def load_synthetic_conversations(self, categories: List[str], rows_per_category: int = 10):
        """
        Generate and add synthetic conversations for the specified categories to the vector database.

        Args:
            categories (List[str]): List of categories to generate conversations for.
            rows_per_category (int): Number of conversations per category.
        """
        synthetic_conversations = []
        for category in categories:
            for _ in range(rows_per_category):
                person_statement = f"A simulated statement reflecting '{category.lower()}' context."
                therapist_response = f"A supportive response tailored for '{category.lower()}' situations."
                synthetic_conversations.append({
                    "Category": category,
                    "Person": person_statement,
                    "Therapist": therapist_response
                })

        # Generate and add embeddings for synthetic conversations
        for conversation in synthetic_conversations:
            text = conversation['Person'] + " " + conversation['Therapist']
            embedding = self.generate_embedding(text)
            self.add_embeddings(np.array([embedding]), [conversation])





