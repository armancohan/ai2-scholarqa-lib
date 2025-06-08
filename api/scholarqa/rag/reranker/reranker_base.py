import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import json
import re

from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import math
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt
from vllm.distributed.parallel_state import destroy_model_parallel

from scholarqa.llms.litellm_helper import llm_completion, batch_llm_completion
from scholarqa.llms.constants import GPT_41_MINI, CompletionResult

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


class QWENRerankerOpenAI(AbstractReranker):
    """QWEN3-Reranker implementation using OpenAI-like API for scoring."""

    def __init__(
        self,
        api_url: str = "http://localhost:8000/score",
        model_name: str = "Qwen/Qwen3-Reranker-4B",
    ):
        """Initialize the QWEN reranker using OpenAI-like API.

        Args:
            api_url: URL of the scoring API endpoint
            model_name: Name of the model to use for scoring
        """
        self.api_url = api_url
        self.model_name = model_name
        self.instruction = "Given the following query for the scientific literature search, retrieve relevant passages that support the information requested. Assign low ranking the survey papers, they are usually not relevant."
        self.headers = {
            "User-Agent": "ScholarQA-Reranker",
            "Content-Type": "application/json",
            "accept": "application/json"
        }

    def _prepare_input(self, query: str, documents: List[str]) -> dict:
        input_1 = f"Instruct: {self.instruction}\nQuery: {query}"
        prompt = {
            "model": self.model_name,
            "query": input_1,
            "documents": documents,
        }
        return prompt

    def _post_request(self, query: str, documents: List[str]) -> List[float]:
        """Make a POST request to the scoring API with batch inference.
        
        Args:
            query: The query text
            documents: List of documents to score against the query
            
        Returns:
            List of relevance scores for each document
        """
        prompt = self._prepare_input(query, documents)
        response = requests.post(self.api_url, headers=self.headers, json=prompt)
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()["scores"]

    def get_scores(self, query: str, documents: List[str]) -> List[float]:
        """Get relevance scores for a query against a list of documents using the scoring API.

        Args:
            query: The search query
            documents: List of documents to score against the query

        Returns:
            List of relevance scores for each document
        """
        # Send all documents in a single batch request
        raise NotImplementedError("QWENRerankerOpenAI is not supported as vLLM serve does not yet support reranking with Qwen3-Reranker. Please use QWENRerankerVLLM instead for direct model inference.")
        scores = self._post_request(query, documents)
        return [float(s) for s in scores]


class QWENRerankerVLLM(AbstractReranker):
    """QWEN3-Reranker implementation using vLLM for direct model inference."""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(QWENRerankerVLLM, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-4B",
        max_model_len: int = 10000,
        gpu_memory_utilization: float = 0.8,
    ):
        """Initialize the QWEN reranker using vLLM.

        Args:
            model_name: Name of the model to use for scoring
            max_model_len: Maximum sequence length for the model
            gpu_memory_utilization: GPU memory utilization ratio
        """
        if self._initialized:
            return

        self.model_name = model_name
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        
        # Get number of available GPUs
        number_of_gpu = torch.cuda.device_count()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model with correct parameters
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=number_of_gpu,
            max_model_len=max_model_len,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization
        )
        
        # Setup suffix and tokens
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        
        # Setup sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.true_token, self.false_token],
        )
        
        # Default instruction
        self.instruction = "Given the following query for the scientific literature search, retrieve relevant passages that support the information requested. Assign low score of relevance to survey papers, they are usually not relevant."
        
        self._initialized = True

    def format_instruction(self, query: str, doc: str) -> List[Dict[str, str]]:
        """Format the input for the model."""
        return [
            {
                "role": "system",
                "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            },
            {
                "role": "user",
                "content": f"<Instruct>: {self.instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"
            }
        ]

    def process_inputs(self, pairs: List[tuple]) -> List[TokensPrompt]:
        """Process and tokenize input pairs."""
        messages = [self.format_instruction(query, doc) for query, doc in pairs]
        messages = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        messages = [ele[:self.max_model_len - len(self.suffix_tokens)] + self.suffix_tokens for ele in messages]
        return [TokensPrompt(prompt_token_ids=ele) for ele in messages]

    def compute_scores(self, outputs) -> List[float]:
        """Compute relevance scores from model outputs."""
        scores = []
        for output in outputs:
            final_logits = output.outputs[0].logprobs[-1]
            
            # Get logits for true/false tokens
            token_count = len(output.outputs[0].token_ids)
            if self.true_token not in final_logits:
                true_logit = -10
            else:
                true_logit = final_logits[self.true_token].logprob
            if self.false_token not in final_logits:
                false_logit = -10
            else:
                false_logit = final_logits[self.false_token].logprob
            
            # Compute probability scores
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            scores.append(score)
        return scores

    def get_scores(self, query: str, documents: List[str]) -> List[float]:
        """Get relevance scores for a query against a list of documents.

        Args:
            query: The search query
            documents: List of documents to score against the query

        Returns:
            List of relevance scores for each document
        """
        # Create query-document pairs
        pairs = [(query, doc) for doc in documents]
        
        # Process inputs
        inputs = self.process_inputs(pairs)
        
        # Generate scores
        outputs = self.model.generate(inputs, self.sampling_params, use_tqdm=False)
        scores = self.compute_scores(outputs)
        
        return [float(s) for s in scores]

    def __del__(self):
        """Cleanup when the instance is deleted."""
        if hasattr(self, 'model'):
            destroy_model_parallel()


class LLMReranker(AbstractReranker):
    """LLM-based reranker that uses litellm calls to judge document relevance."""

    def __init__(
        self,
        model: str = GPT_41_MINI,
        fallback_model: str = None,
        use_batch: bool = True,
        max_tokens: int = 10,
        temperature: float = 0.0,
        **llm_kwargs
    ):
        """Initialize the LLM reranker.

        Args:
            model: The LLM model to use for relevance scoring
            fallback_model: Fallback model if the primary model fails
            use_batch: Whether to use batch processing for multiple documents
            max_tokens: Maximum tokens for the LLM response
            temperature: Temperature for LLM generation (0.0 for deterministic)
            **llm_kwargs: Additional arguments passed to the LLM
        """
        self.model = model
        self.fallback_model = fallback_model
        self.use_batch = use_batch
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llm_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            **llm_kwargs
        }

        # System prompt for relevance scoring
        self.system_prompt = """You are an reserach scientist and expert at judging the relevance of scientific papers to research queries. Your task is to evaluate whether a scientific paper (abstract or paper snippets) addresses the given research query and provide a relevance score.

Instructions:
1. Read the research query and scientific paper content carefully
2. Judge how well the paper addresses the specific research query
3. Consider the following scoring criteria:
   - SURVEY PAPERS: Score very low (1-2) as they are typically not relevant for specific research questions
   - REVIEW PAPERS: Score low (2-3) unless they directly address the specific query)
   - EMPIRICAL STUDIES: Score higher if they present novel findings, methods, results, findings, or related to the query
   - METHODOLOGICAL PAPERS: Score high if they introduce techniques, algorithms, or approaches relevant to the query

4. Respond with ONLY a single integer between 0 and 10, where:
   - 0-2 = irrelevant papers that don't address the specific query
   - 3-4 = low relevance (mentions related concepts but doesn't directly address the query)
   - 5-6 = moderate relevance (partially addresses the query with some relevant findings)
   - 7-8 = high relevance (directly addresses the query with relevant methods/results)
   - 9-10 = perfect relevance (specifically focused on the query topic with novel contributions)

Respond with ONLY the numerical score (e.g., "7"). Do not include any explanation or additional text."""

    def format_prompt(self, query: str, document: str) -> str:
        """Format the prompt for relevance scoring."""
        return f"""Research Query: {query}

Scientific Paper Content: {document}

Relevance Score:"""

    def parse_score(self, response: str) -> float:
        """Parse the LLM response to extract the relevance score (0-10) and normalize to 0.0-1.0."""
        try:
            # Remove any whitespace and try to extract a number
            response = response.strip()
            
            # Look for an integer or decimal number
            score_match = re.search(r'(\d*\.?\d+)', response)
            if score_match:
                raw_score = float(score_match.group(1))
                # Clamp score between 0 and 10
                raw_score = max(0.0, min(10.0, raw_score))
                # Normalize to 0.0-1.0 scale for compatibility with existing interface
                normalized_score = raw_score / 10.0
                return normalized_score
            else:
                logger.warning(f"Could not parse score from response: '{response}', defaulting to 0.0")
                return 0.0
        except (ValueError, AttributeError) as e:
            logger.warning(f"Error parsing score from response: '{response}', error: {e}, defaulting to 0.0")
            return 0.0

    def score_single_document(self, query: str, document: str) -> float:
        """Score a single document against the query."""
        user_prompt = self.format_prompt(query, document)
        
        try:
            result = llm_completion(
                user_prompt=user_prompt,
                system_prompt=self.system_prompt,
                model=self.model,
                fallback=self.fallback_model,
                **self.llm_kwargs
            )
            score = self.parse_score(result.content)
            # Extract raw score for logging
            raw_score_match = re.search(r'(\d*\.?\d+)', result.content.strip())
            raw_score = float(raw_score_match.group(1)) if raw_score_match else 0
            logger.debug(f"Scored document (raw: {raw_score}/10, normalized: {score:.3f}): {document[:100]}...")
            return score
        except Exception as e:
            logger.error(f"Error scoring document: {e}")
            return 0.0

    def score_batch_documents(self, query: str, documents: List[str]) -> List[float]:
        """Score multiple documents against the query using batch processing."""
        user_prompts = [self.format_prompt(query, doc) for doc in documents]
        
        try:
            results = batch_llm_completion(
                messages=user_prompts,
                system_prompt=self.system_prompt,
                model=self.model,
                fallback=self.fallback_model,
                **self.llm_kwargs
            )
            scores = [self.parse_score(result.content) for result in results]
            logger.debug(f"Batch scored {len(documents)} documents")
            return scores
        except Exception as e:
            logger.error(f"Error in batch scoring: {e}, falling back to individual scoring")
            # Fallback to individual scoring
            return [self.score_single_document(query, doc) for doc in documents]

    def get_scores(self, query: str, documents: List[str]) -> List[float]:
        """Get relevance scores for a query against a list of documents.

        Args:
            query: The search query
            documents: List of documents to score against the query

        Returns:
            List of relevance scores (0.0 to 1.0) for each document
        """
        if not documents:
            return []

        logger.info(f"Scoring {len(documents)} documents against query using {self.model}")
        
        if self.use_batch and len(documents) > 1:
            return self.score_batch_documents(query, documents)
        else:
            return [self.score_single_document(query, doc) for doc in documents]


RERANKER_MAPPING = {
    "crossencoder": CrossEncoderScores,
    "biencoder": BiEncoderScores,
    "flag_embedding": FlagEmbeddingScores,
    "qwen": QWENReranker,
    "qwen_openai": QWENRerankerOpenAI,
    "qwen_vllm": QWENRerankerVLLM,
    "llm": LLMReranker,
}

def test_qwen_reranker_openai():
    """Test function to demonstrate QWENRerankerOpenAI usage."""
    # Initialize the reranker
    reranker = QWENRerankerOpenAI(
        api_url="http://localhost:8000/rerank",
        model_name="Qwen/Qwen3-Reranker-4B"
    )

    # Test query and documents
    query = "What are the recent advances in transformer-based language models for scientific text understanding?"
    
    documents = [
        # Relevant document about transformers in scientific text
        "Recent advances in transformer-based models have significantly improved scientific text understanding. Models like SciBERT and BioBERT have shown remarkable performance in biomedical text mining tasks, achieving state-of-the-art results in named entity recognition and relation extraction.",
        
        # Survey paper (should get lower score due to instruction)
        "This survey paper provides a comprehensive overview of various approaches to scientific text understanding, including traditional machine learning methods, deep learning techniques, and recent transformer-based models. We discuss the evolution of these methods and their applications in different scientific domains.",
        
        # Relevant document about specific application
        "We present a novel transformer-based architecture specifically designed for scientific literature analysis. Our model achieves 92% accuracy in identifying key findings from research papers and 88% in extracting causal relationships between scientific concepts.",
        
        # Less relevant document
        "The history of natural language processing dates back to the 1950s, with early systems focusing on rule-based approaches and statistical methods. This paper discusses the evolution of NLP techniques over the decades.",
        
        # Highly relevant document with specific technical details
        "Our transformer-based model, SciT5, introduces a new pre-training objective specifically for scientific text. It achieves 95% accuracy in understanding complex scientific relationships and outperforms previous models by 15% in scientific question answering tasks."
    ]

    # Get scores
    try:
        scores = reranker.get_scores(query, documents)
        
        # Print results
        print("\nQuery:", query)
        print("\nResults:")
        print("-" * 80)
        for doc, score in zip(documents, scores):
            print(f"Score: {score:.4f}")
            print(f"Document: {doc[:150]}...")
            print("-" * 80)
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")


def test_qwen_reranker_vllm():
    """Test function to demonstrate QWENRerankerVLLM usage."""
    # Initialize the reranker with correct parameters
    reranker = QWENRerankerVLLM(
        model_name="Qwen/Qwen3-Reranker-4B",
        max_model_len=10000,
        gpu_memory_utilization=0.8
    )

    # Test query and documents
    query = "What are the recent advances in transformer-based language models for scientific text understanding?"
    
    documents = [
        # Relevant document about transformers in scientific text
        "Recent advances in transformer-based models have significantly improved scientific text understanding. Models like SciBERT and BioBERT have shown remarkable performance in biomedical text mining tasks, achieving state-of-the-art results in named entity recognition and relation extraction.",
        
        # Survey paper (should get lower score due to instruction)
        "This survey paper provides a comprehensive overview of various approaches to scientific text understanding, including traditional machine learning methods, deep learning techniques, and recent transformer-based models. We discuss the evolution of these methods and their applications in different scientific domains.",
        
        # Relevant document about specific application
        "We present a novel transformer-based architecture specifically designed for scientific literature analysis. Our model achieves 92% accuracy in identifying key findings from research papers and 88% in extracting causal relationships between scientific concepts.",
        
        # Less relevant document
        "The history of natural language processing dates back to the 1950s, with early systems focusing on rule-based approaches and statistical methods. This paper discusses the evolution of NLP techniques over the decades.",
        
        # Highly relevant document with specific technical details
        "Our transformer-based model, SciT5, introduces a new pre-training objective specifically for scientific text. It achieves 95% accuracy in understanding complex scientific relationships and outperforms previous models by 15% in scientific question answering tasks."
    ]

    # Get scores
    try:
        scores = reranker.get_scores(query, documents)
        
        # Print results
        print("\nQuery:", query)
        print("\nResults:")
        print("-" * 80)
        for doc, score in zip(documents, scores):
            print(f"Score: {score:.4f}")
            print(f"Document: {doc[:150]}...")
            print("-" * 80)
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")


def test_llm_reranker():
    """Test function to demonstrate LLMReranker usage."""
    # Initialize the reranker
    reranker = LLMReranker(
        model=GPT_41_MINI,
        use_batch=True,
        max_tokens=10,
        temperature=0.0
    )

    # Test query and documents
    query = "What are the recent advances in transformer-based language models for scientific text understanding?"
    
    documents = [
        # Highly relevant empirical study (should score 8-9/10 → 0.8-0.9 normalized)
        "We present SciBERT, a pre-trained language representation model for scientific text based on BERT. Our model is trained on a large corpus of scientific publications and achieves state-of-the-art performance on scientific NLP tasks including named entity recognition, relation extraction, and text classification. Experimental results show 15% improvement over BERT-base on biomedical NER and 12% on computer science paper classification.",
        
        # Survey paper (should score 1-2/10 → 0.1-0.2 normalized)
        "This survey provides a comprehensive overview of transformer-based language models in scientific text processing. We review 150+ papers published between 2018-2023, categorizing approaches into domain adaptation, pre-training strategies, and downstream applications. We identify key trends and future research directions in scientific NLP.",
        
        # Methodological paper with novel approach (should score 8-9/10 → 0.8-0.9 normalized)
        "We introduce SciT5, a novel transformer architecture specifically designed for scientific text understanding. Our method incorporates domain-specific pre-training objectives including citation prediction and section classification. SciT5 achieves 95% accuracy on scientific relation extraction and outperforms previous models by 18% on scientific question answering benchmarks.",
        
        # Historical/background paper (should score 2-3/10 → 0.2-0.3 normalized)
        "The evolution of natural language processing in scientific domains traces back to early rule-based systems in the 1970s. This paper provides historical context for current transformer-based approaches, discussing the progression from statistical methods through early neural networks to modern attention mechanisms.",
        
        # Review paper with some relevance (should score 2-3/10 → 0.2-0.3 normalized)
        "Recent reviews of transformer architectures highlight their success across multiple domains. This review paper examines 200+ transformer variants, analyzing their architectural innovations and performance across tasks. While transformers show promise in scientific text, most work focuses on general domain applications.",
        
        # Highly relevant case study (should score 7-8/10 → 0.7-0.8 normalized)
        "We evaluate transformer-based models on PubMed abstract classification, comparing BERT, SciBERT, and domain-specific variants. Our experiments on 100K biomedical abstracts show that scientific domain pre-training improves classification accuracy by 23%. We provide detailed analysis of model performance across different medical specialties."
    ]

    # Get scores
    try:
        scores = reranker.get_scores(query, documents)
        
        # Print results
        print("\nLLM Reranker Test Results")
        print("=" * 80)
        print("Query:", query)
        print("\nResults:")
        print("-" * 80)
        for i, (doc, score) in enumerate(zip(documents, scores)):
            raw_score = score * 10  # Convert back to 0-10 for display
            print(f"Document {i+1} - Score: {raw_score:.1f}/10 (normalized: {score:.3f})")
            print(f"Content: {doc[:150]}...")
            print("-" * 80)
            
        # Show sorted results
        sorted_results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        print("\nRanked Results (by relevance score):")
        print("=" * 80)
        for i, (doc, score) in enumerate(sorted_results):
            raw_score = score * 10  # Convert back to 0-10 for display
            print(f"Rank {i+1} - Score: {raw_score:.1f}/10 (normalized: {score:.3f})")
            print(f"Content: {doc[:100]}...")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")


if __name__ == "__main__":
    test_llm_reranker()
    # test_qwen_reranker_openai()
    # test_qwen_reranker_vllm()
