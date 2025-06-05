import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    logger.warning("torch not found, custom baseline rerankers will not work.")


class AbstractReranker(ABC):
    @abstractmethod
    def get_scores(self, query: str, documents: List[str]) -> List[float]:
        pass


class SentenceTransformerEncoder:
    def __init__(self, model_name_or_path: str):
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name_or_path, revision=None, device=device)

    def encode(self, sentences: List[str]):
        return self.model.encode(sentences, show_progress_bar=True, convert_to_tensor=True)

    def get_tokenizer(self):
        return self.model.tokenizer


# GIST embeddings model supported by Sentence transformer
# https://huggingface.co/avsolatorio/GIST-large-Embedding-v0
class BiEncoderScores(AbstractReranker):
    def __init__(self, model_name_or_path: str):
        self.model = SentenceTransformerEncoder(model_name_or_path)

    def get_scores(self, query: str, passages: List[str]) -> List[float]:
        query_embedding = self.model.encode([query])[0]
        passage_embeddings = self.model.encode(passages)
        scores = F.cosine_similarity(query_embedding.unsqueeze(0), passage_embeddings).cpu().numpy()
        return [float(s) for s in scores]


# Sentence Transformer supports Jina AI (https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)
# and Mix Bread re-rankers (https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1)
class CrossEncoderScores(AbstractReranker):
    def __init__(self, model_name_or_path: str):
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)
        self.model = CrossEncoder(
            model_name_or_path,
            automodel_args={"torch_dtype": "float16"},
            trust_remote_code=True,
            device=device,
        )

    def get_tokenizer(self):
        return self.model.tokenizer

    def get_scores(self, query: str, passages: List[str]) -> List[float]:
        sentence_pairs = [[query, passage] for passage in passages]
        scores = self.model.predict(sentence_pairs, convert_to_tensor=True, show_progress_bar=True, batch_size=128).tolist()
        return [float(s) for s in scores]


# Supports the BAAI/bge... models https://huggingface.co/BAAI/bge-reranker-v2-m3
class FlagEmbeddingScores:
    def __init__(self, model_name_or_path: str):
        from FlagEmbedding import FlagReranker

        self.model = FlagReranker(model_name_or_path, use_fp16=True)

    def get_tokenizer(self):
        return self.model.tokenizer

    def get_scores(self, query: str, passages: List[str], separator: str) -> List[float]:
        sentence_pairs = [(query, passage.replace(separator, self.get_tokenizer().eos_token)) for passage in passages]
        scores = self.model.compute_score(sentence_pairs, normalize=True, batch_size=32)
        return [float(s) for s in scores]


class QWENReranker(AbstractReranker):
    """QWEN3-Reranker-8B implementation for reranking documents."""

    def __init__(
        self, model_name: str = "Qwen/Qwen3-Reranker-8B", use_flash_attention: bool = True, device: Optional[str] = None
    ):
        """Initialize the QWEN reranker.

        Args:
            model_name: Name or path of the QWEN model
            use_flash_attention: Whether to use flash attention for better performance
            device: Device to run the model on (cuda/cpu). If None, will use cuda if available
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                raise ValueError("No cuda device found. Not patient enough to run on cpu")

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_flash_attention else None,
            attn_implementation="flash_attention_2" if use_flash_attention else None,
            device_map=device,
        ).eval()

        # Setup token IDs and max length
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192

        # Setup prompt templates
        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        # Default instruction
        self.instruction = "Given the following query for the scientific literature search, retrieve relevant passages that support the information requested."

    def format_instruction(self, query: str, doc: str) -> str:
        """Format the input for the model."""
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"

    def process_inputs(self, pairs: List[str]) -> Dict[str, torch.Tensor]:
        """Process and tokenize input pairs."""
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )

        # Add prefix and suffix tokens
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens

        # Pad and convert to tensors
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)

        # Move to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        return inputs

    def compute_scores(self, inputs: Dict[str, torch.Tensor]) -> List[float]:
        """Compute relevance scores for the input pairs."""
        with torch.no_grad():
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
        return scores

    def get_scores(self, query: str, documents: List[str]) -> List[float]:
        """Get relevance scores for a query against a list of documents."""
        # Format input pairs
        pairs = [self.format_instruction(query, doc) for doc in documents]

        # Process inputs
        inputs = self.process_inputs(pairs)

        # Compute scores
        scores = self.compute_scores(inputs)

        return [float(s) for s in scores]


RERANKER_MAPPING = {
    "crossencoder": CrossEncoderScores,
    "biencoder": BiEncoderScores,
    "flag_embedding": FlagEmbeddingScores,
    "qwen": QWENReranker,
}
