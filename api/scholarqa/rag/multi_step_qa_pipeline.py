import json
import logging
import re
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field
from scholarqa.llms.constants import GPT_41
from scholarqa.llms.litellm_helper import batch_llm_completion, llm_completion
from scholarqa.llms.prompts import (
    PROMPT_ASSEMBLE_NO_QUOTES_SUMMARY,
    USER_PROMPT_PAPER_LIST_FORMAT,
    USER_PROMPT_QUOTE_LIST_FORMAT,
)
from scholarqa.utils import CompletionResult

logger = logging.getLogger(__name__)


class DimFormat(str, Enum):
    SYNTHESIS = "synthesis"
    LIST = "list"


class Dimension(BaseModel):
    name: str = Field(default=None, description=("The name of the dimension"))
    format: DimFormat = Field(
        default=None, description=("The generation format of the dimension - can be either list of synthesis")
    )
    quotes: List[int] = Field(
        default=None,
        description=("A list of indices of paper quotes in the dimension, can be empty if no relevant quotes are found"),
    )


class ClusterPlan(BaseModel):
    cot: str = Field(default=None, description=("The justification for every dimension name and its format"))
    dimensions: List[Dimension] = Field(
        default=None, description=("The list of dimensions along with the associated quote indices as per the cot plan")
    )


class MultiStepQAPipeline:
    def __init__(
        self,
        llm_model: str = None,
        quote_extraction_llm: str = None,
        clustering_llm: str = None,
        summary_generation_llm: str = None,
        fallback_llm: str = GPT_41,
        batch_workers: int = 20,
        **llm_kwargs,
    ):
        # Use llm_model as default for all steps if specific models not provided
        self.llm_model = llm_model or GPT_41  # For backward compatibility
        self.quote_extraction_llm = quote_extraction_llm or self.llm_model
        self.clustering_llm = clustering_llm or self.llm_model
        self.summary_generation_llm = summary_generation_llm or self.llm_model
        self.fallback_llm = fallback_llm
        self.batch_workers = batch_workers
        self.llm_kwargs = {"max_tokens": 4096 * 4}
        if llm_kwargs:
            self.llm_kwargs.update(llm_kwargs)

    def _add_gemini_thinking_if_needed(self, model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Add thinking parameter for Gemini models"""
        if model and "gemini-pro" in model.lower():
            kwargs = kwargs.copy()
            # kwargs["thinking"] = {"type": "disabled", "budget_tokens": 0}  # TODO: uncomment this
        return kwargs

    def step_select_quotes(
        self, query: str, scored_df: pd.DataFrame, sys_prompt: str
    ) -> Tuple[Dict[str, str], List[CompletionResult]]:
        """
        Extracts relevant quotes from a list of papers for a given query.
        """
        logger.info(
            f"Querying {self.quote_extraction_llm} to extract quotes from {len(scored_df)} papers "
            f"with {self.batch_workers} parallel workers."
        )

        # Prepare prompts for each paper for batch LLM processing.
        reference_strings = scored_df["reference_string"]
        paper_contents = scored_df["relevance_judgment_input_expanded"]
        messages = [USER_PROMPT_PAPER_LIST_FORMAT.format(query, content) for content in paper_contents]

        # Add thinking parameter for Gemini models
        llm_kwargs = self._add_gemini_thinking_if_needed(self.quote_extraction_llm, self.llm_kwargs)

        # Get quote extractions from the LLM.
        completion_results = batch_llm_completion(
            self.quote_extraction_llm,
            messages=messages,
            system_prompt=sys_prompt,
            max_workers=self.batch_workers,
            fallback=self.fallback_llm,
            **llm_kwargs,
        )

        # Clean up "None" responses from the LLM.
        def _clean_quote(content: Optional[str]) -> str:
            """Returns an empty string if the LLM provided a 'None' response or content is None."""
            if not content:
                return ""
            if content == "None" or content.startswith("None\n") or content.startswith("None "):
                return ""
            return content

        quotes = [_clean_quote(cr.content) for cr in completion_results]

        # Create a dictionary of {reference_string: quote} for valid quotes.
        # A valid quote must be longer than 10 characters.
        per_paper_summaries = {ref: quote for ref, quote in zip(reference_strings, quotes) if len(quote) > 10}

        # Sort the summaries by reference string for consistent output order.
        sorted_summaries = dict(sorted(per_paper_summaries.items()))

        return sorted_summaries, completion_results

    def step_clustering(
        self, query: str, per_paper_summaries: Dict[str, str], sys_prompt: str
    ) -> Tuple[Dict[str, Any], CompletionResult]:
        def make_prompt(query: str, paper_paper_quotes_dict: Dict[str, str]) -> str:
            # paper_paper_quotes_dict is a dictionary with keys being the paper titles and values being the quotes
            # need to make a single string with all of the quotes
            quotes = ""
            for idx, (paper, quotes_str) in enumerate(paper_paper_quotes_dict.items()):
                # there are multiple quotes per paper
                quotes_str = quotes_str.replace("\n", "")
                quotes += f"[{idx}]\t{quotes_str}" + "\n"
            prompt = USER_PROMPT_QUOTE_LIST_FORMAT.format(query, quotes)
            return prompt

        user_prompt = make_prompt(query, per_paper_summaries)
        try:
            # Add thinking parameter for Gemini models
            llm_kwargs = self._add_gemini_thinking_if_needed(self.clustering_llm, self.llm_kwargs)

            # params for reasoning mode: max_completion_tokens=4096, max_tokens=4096+1024, reasoning_effort="low"
            response = llm_completion(
                user_prompt=user_prompt,
                system_prompt=sys_prompt,
                fallback=self.fallback_llm,
                model=self.clustering_llm,
                response_format=ClusterPlan,
                **llm_kwargs,
            )
            return json.loads(response.content), response
        except Exception as e:
            logger.warning(f"Error while clustering with {self.clustering_llm}: {e}")
            raise e

    def generate_iterative_summary(
        self, query: str, per_paper_summaries_extd: Dict[str, Dict[str, Any]], plan: Dict[str, Any], sys_prompt: str
    ) -> Generator[CompletionResult, None, None]:
        # first, we need to make a map from the index to the quotes because the llm is using index only

        # now fill in the prompt
        per_paper_summaries_tuples = [(ref_string, response) for ref_string, response in per_paper_summaries_extd.items()]
        # only use the section headings from the plan, discard the quote indices
        plan_str = "\n".join([k for k in plan])
        existing_sections = []
        i = 0
        for section_name, inds in plan.items():
            # inds are a string like this: "[1, 2, 3]"
            # get the quotes for each index
            quotes = ""
            for ind in inds:
                if ind < len(per_paper_summaries_tuples):
                    quotes += per_paper_summaries_tuples[ind][0] + ": " + str(per_paper_summaries_tuples[ind][1]) + "\n"
                else:
                    logger.warning(f"index {ind} out of bounds")
            # existing sections should have their summaries removed because they are confusing.
            # remove anything in []
            already_written = "\n\n".join(existing_sections)
            already_written = re.sub(r"\[.*?\]", "", already_written)
            fill_in_prompt_args = {
                "query": query,
                "plan": plan_str,
                "already_written": already_written,
                "section_name": section_name,
            }
            if quotes:
                fill_in_prompt_args["section_references"] = quotes
                filled_in_prompt = sys_prompt.format(**fill_in_prompt_args)
            else:
                logger.warning(f"No quotes for section {section_name}")
                filled_in_prompt = PROMPT_ASSEMBLE_NO_QUOTES_SUMMARY.format(**fill_in_prompt_args)

            # Add thinking parameter for Gemini models
            llm_kwargs = self._add_gemini_thinking_if_needed(self.summary_generation_llm, self.llm_kwargs)

            response = llm_completion(
                user_prompt=filled_in_prompt, model=self.summary_generation_llm, fallback=self.fallback_llm, **llm_kwargs
            )
            existing_sections.append(response.content)
            yield response
