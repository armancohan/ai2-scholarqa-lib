import json
import logging
import os
import re
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field
from scholarqa.llms.constants import GPT_41
from scholarqa.llms.litellm_helper import (add_gemini_thinking_if_needed,
                                           batch_llm_completion,
                                           llm_completion)
from scholarqa.llms.prompts import (PROMPT_ASSEMBLE_NO_QUOTES_SUMMARY,
                                    USER_PROMPT_PAPER_LIST_FORMAT,
                                    USER_PROMPT_QUOTE_LIST_FORMAT)

from scholarqa.utils import CompletionResult, extract_json_from_response

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
        task_id: str = None,
        **llm_kwargs,
    ):
        # Use llm_model as default for all steps if specific models not provided
        self.llm_model = llm_model or GPT_41  # For backward compatibility
        self.quote_extraction_llm = quote_extraction_llm or self.llm_model
        self.clustering_llm = clustering_llm or self.llm_model
        self.summary_generation_llm = summary_generation_llm or self.llm_model
        self.fallback_llm = fallback_llm
        self.batch_workers = batch_workers
        self.task_id = task_id  # Will be updated by the ScholarQA class
        self.llm_kwargs = {"max_tokens": 4096 * 8}
        if llm_kwargs:
            self.llm_kwargs.update(llm_kwargs)

    def update_task_id(self, task_id: str):
        self.task_id = task_id

    def _save_prompt_to_outputs(self, prompt: str, section_name: str = "section", step_number: int = 0) -> None:
        """
        Elegantly save the prompt to outputs directory with task_id-based naming.

        Args:
            prompt: The prompt text to save
            section_name: Name of the section being generated
            step_number: Sequential number for multiple sections
        """
        if not self.task_id:
            logger.warning("No task_id available, skipping prompt save")
            return

        try:
            # Create outputs directory if it doesn't exist
            outputs_dir = "outputs"
            os.makedirs(outputs_dir, exist_ok=True)

            # Clean section name for filename
            clean_section_name = re.sub(r"[^\w\-_\.]", "_", section_name.lower())

            # Create filename with task_id, section name, and step number
            filename = f"{self.task_id}_prompt_{clean_section_name}_step_{step_number:02d}.txt"
            filepath = os.path.join(outputs_dir, filename)

            # Save the prompt with metadata
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# Task ID: {self.task_id}\n")
                f.write(f"# Section: {section_name}\n")
                f.write(f"# Step: {step_number}\n")
                f.write(f"# Model: {self.summary_generation_llm}\n")
                f.write(f"# Generated at: {pd.Timestamp.now().isoformat()}\n")
                f.write("\n" + "=" * 80 + "\n")
                f.write("# PROMPT CONTENT\n")
                f.write("=" * 80 + "\n\n")
                f.write(prompt)

            logger.info(f"Prompt saved to: {filepath}")

        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.warning(f"Failed to save prompt to outputs directory: {e}")

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
        llm_kwargs = add_gemini_thinking_if_needed(self.quote_extraction_llm, self.llm_kwargs)

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
            llm_kwargs = add_gemini_thinking_if_needed(self.clustering_llm, self.llm_kwargs)

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

            # Save the prompt before calling the LLM
            self._save_prompt_to_outputs(filled_in_prompt, section_name, i)

            # Add thinking parameter for Gemini models
            llm_kwargs = add_gemini_thinking_if_needed(self.summary_generation_llm, self.llm_kwargs)

            response = llm_completion(
                user_prompt=filled_in_prompt, model=self.summary_generation_llm, fallback=self.fallback_llm, **llm_kwargs
            )
            existing_sections.append(response.content)
            yield response

    def generate_future_ideas_iterative(
        self, snippets_text: str, ideation_query: str, ideation_instructions: str, num_ideas: int = 6
    ) -> Generator[CompletionResult, None, None]:
        """Generate future research ideas one at a time"""

        existing_ideas = []

        for i in range(num_ideas):
            # Create prompt for generating one idea
            already_generated = "\n\n".join([f"Idea {j+1}: {idea}" for j, idea in enumerate(existing_ideas)])
            already_generated_text = f"\n{already_generated}" if existing_ideas else "None"

            if ideation_instructions:
                additional_ideation_instructions = f"""
                Additional important instructions: {ideation_instructions}\n
                """
            else:
                additional_ideation_instructions = "\n"

            prompt = f"""You are tasked with generating one concrete future research direction related to the query:
            "{ideation_query}"

Plan:
First, I will provide you with a list of research papers and some future work snippets from these papers so you understand the latest research around this topic.
Then, I will provide you with some previously generated ideas. You should not generate ideas that are similar to the previously generated ideas.
Then, I will provide you with additional detailed instructions.

Already generated ideas:
{already_generated_text}

Recent Research Papers with Future Work Snippets from these papers:
{snippets_text}

---

Detailed instructions:

Read the context carefully and reason about possible future work directions.
Generate exactly ONE new concrete and actionable future research direction. This direction should be:

1. Specific and well-defined (not just vague suggestions)
2. Technically feasible with reasonable budget and resources. For example: we don't want large pretraining experiments, or large data annotations.
3. Innovative and forward-thinking. Ideally something that is less obvious and more interesting. You can take inspiration from the context provided and the Future Work snippets in the contex.
4. Relevant to the ideation research question: "{ideation_query}"
5. Different from any previously generated ideas
6. Include specific methodologies, approaches
7. Include specific datasets, evaluation metrics, and expected outcomes using the context provided. Look for appropriate datasets and evaluation methods.
{additional_ideation_instructions}
Format your response as follows:

## Future Research Direction {i+1}: [Descriptive Title]
[At least 2 paragraphs of detailed description including specific methodology, expected outcomes, and technical approach. Reference relevant papers when appropriate.]

Focus on being specific about methodologies, datasets, evaluation metrics, and expected outcomes rather than providing general suggestions."""

            # Add thinking parameter for Gemini models
            llm_kwargs = add_gemini_thinking_if_needed(self.summary_generation_llm, self.llm_kwargs)

            # Save the prompt before calling the LLM
            self._save_prompt_to_outputs(prompt, "future_ideas", i)

            response = llm_completion(
                user_prompt=prompt, model=self.summary_generation_llm, fallback=self.fallback_llm, **llm_kwargs
            )

            # Evaluate the generated idea
            evaluation, evaluation_json = self.evaluate_research_idea(response.content, ideation_query, snippets_text)

            # Combine idea with evaluation
            idea_with_evaluation = f"{response.content}\n\n### Idea Evaluation\n{evaluation.content}"

            # Add the revised idea to existing ideas for context
            if evaluation_json and evaluation_json.get('revised_research_idea'):
                existing_ideas.append(evaluation_json['revised_research_idea'])
            else:
                existing_ideas.append(idea_with_evaluation)

            # Create a new response object with the combined content
            combined_response = CompletionResult(
                content=idea_with_evaluation,
                model=response.model,
                cost=response.cost + evaluation.cost,
                input_tokens=response.input_tokens + evaluation.input_tokens,
                output_tokens=response.output_tokens + evaluation.output_tokens,
                total_tokens=response.total_tokens + evaluation.total_tokens,
            )

            yield combined_response

    def evaluate_research_idea(self, idea_text: str, ideation_query: str, context_snippets: str) -> Tuple[CompletionResult, Dict]:
        """Evaluate the quality and feasibility of a generated research idea"""

        evaluation_prompt = f"""You are an expert research scientist tasked with evaluating the quality of a proposed research direction. You are very critical, careful, and thorough. You do not like vague and high-level ideas and what people propose should be concretely described and not something already similar to what is out there.

You are given a topic or query on which the research idea is based on.
Then you are given a list of research papers and snippets that are relevant to the topic.
Then you are given a proposed research idea.

You need to evaluate the quality of the proposed research idea across the following dimensions and provide a comprehensive assessment.
Then you need to revise the proposed research idea based on the evaluation.
If the proposal is not savable, you need to say so.

See the sections below.

Original Research Query: "{ideation_query}"

Research Context (relevant papers and snippets):
{context_snippets}

-------------------------------------
-------------------------------------

Proposed Research Idea:
{idea_text}

Please evaluate and revise this research idea across the following dimensions and provide a comprehensive assessment:

## Evaluation Criteria:

1. **Novelty and Innovation** (1-10): How novel and innovative is this idea compared with existing research and what was described above? Does it introduce new concepts or approaches?
Note: If the proposal includes "attention mechanism", it is probably is useless idea. 

2. **Clarity, concreteness, and Specificity** (1-10): How clear and specific are the proposed methods and expected outcomes? Or it is vague and high-level?

3. **Technical Feasibility** (1-10): In terms of executing on the idea, experiments needed to run, concrete evaluation methodology, how feasible is the proposal?

## Format your evaluation as follows:

**Overall Score: X/30**

**Weaknesses:**
- [List 2-3 key weaknesses or areas for improvement]

**Final Assessment:** [1-2 sentences summarizing whether this is a strong research direction and why]

Note: Be very critical about the proposal.

**Revised Research Idea:** 
Finally, you need to revise the research idea based on the evaluation. Be very careful and concrete.
[Explain concretely what is the revised research idea in 3-4 paragraphs. Important: This should be 3-4 paragraphs not shorter. Use latex citation format. If the proposal is not savable, say so.]

Provide your final revised research idea in the following json format:
{{
    "novelty_and_innovation": int,
    "clarity_and_specificity": int,
    "technical_feasibility": int,
    "overall_score": int,
    "weaknesses": str,
    "final_assessment": str,
    "revised_research_idea_title": str,   
    "revised_research_idea_detailed_description": str 
}}

"""

        # Add thinking parameter for Gemini models
        llm_kwargs = add_gemini_thinking_if_needed(self.summary_generation_llm, self.llm_kwargs)

        evaluation_response = llm_completion(
            user_prompt=evaluation_prompt, model=self.summary_generation_llm, fallback=self.fallback_llm, **llm_kwargs
        )
        
        # Extract JSON from the response content
        json_response = extract_json_from_response(evaluation_response.content)
        
        # Format the JSON response as markdown
        if json_response and isinstance(json_response, dict):
            converted_json_to_markdown = f"""**Evaluation Results:**

**Overall Score:** {json_response.get('overall_score', 'N/A')}/30

**Individual Scores:**
- Novelty and Innovation: {json_response.get('novelty_and_innovation', 'N/A')}/10
- Clarity and Specificity: {json_response.get('clarity_and_specificity', 'N/A')}/10
- Technical Feasibility: {json_response.get('technical_feasibility', 'N/A')}/10

**Weaknesses:**
{json_response.get('weaknesses', 'No weaknesses identified')}

**Final Assessment:**
{json_response.get('final_assessment', 'No assessment provided')}

**Revised Research Idea:**
{json_response.get('revised_research_idea_title', 'No title provided')}\n\n
{json_response.get('revised_research_idea_detailed_description', 'No revision provided')}"""
        else:
            # Fallback if JSON extraction failed
            converted_json_to_markdown = evaluation_response.content
            json_response = {}
        
        # Create a new CompletionResult with formatted content
        formatted_evaluation_response = CompletionResult(
            content=converted_json_to_markdown,
            model=evaluation_response.model,
            cost=evaluation_response.cost,
            input_tokens=evaluation_response.input_tokens,
            output_tokens=evaluation_response.output_tokens,
            total_tokens=evaluation_response.total_tokens,
        )
        
        return formatted_evaluation_response, json_response
