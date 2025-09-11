from typing import List
from .generator import generate_answer
from .hallucination import detect_hallucination
from .rewritting import rewrite_query
from .reflection import regenerate_answer
from .ranking import rank_output
from .utils import filter_docs, vector_db_retrieve_context
from src.core.llm_manager import LLMManager
from dotenv import load_dotenv

load_dotenv()


class Pipeline:
    def __init__(self, hallucination: bool = True, per_span_reprompting: bool = True):
        """
        Args:
            hallucination (bool): Whether to perform hallucination detection and correction
            per_span_reprompting (bool): If True, reprocess each hallucinated span individually;
                                         if False, reprocess all at once
        """
        self.model = LLMManager().get_model()
        self.hallucination = hallucination
        self.per_span_reprompting = per_span_reprompting

    async def run(self, question: str, collection_ids: List[str]) -> dict:
        """
        Run the QA pipeline for a single question.

        Args:
            question (str): User input question

        Returns:
            dict: Output of each stage in the pipeline
        """
        # Initial retrieval
        docs = await vector_db_retrieve_context(question, collection_ids=collection_ids)

        # Generation step
        print("Generating answer...")
        generation_response = generate_answer(self.model, question, docs)

        # Early exit if hallucination pipeline is disabled
        if not self.hallucination:
            print(f"Response \n{generation_response.answer}")
            return {
                "question": question,
                "docs": docs,
                "generation_response": generation_response,
                "hallucination_response": None,
                "rewrite_response": None,
                "reflected_response": None,
                "ranked_output": None,
            }

        # Hallucination detection step
        print("Detecting hallucinations...")
        hallucination_response = detect_hallucination(
            self.model, generation_response, docs
        )

        # Early exit if no hallucinated spans
        if not any(span["prob"] > 0.5 for span in hallucination_response.soft_labels):
            print("No hallucination detected")
            return {
                "question": question,
                "docs": docs,
                "generation_response": generation_response,
                "hallucination_response": hallucination_response,
                "rewrite_response": None,
                "reflected_response": None,
                "ranked_output": None,
            }

        rewritten_response = None  # ensure it exists for return

        if self.per_span_reprompting:
            # Handle hallucinated spans one at a time
            for i, soft in enumerate(hallucination_response.soft_labels):
                prob = soft.get("prob", 1.0)
                if prob < 0.5:
                    print(
                        f"Skipping hallucinated span {i} due to low probability ({prob:.2f})"
                    )
                    continue

                print(f"Retrieving for hallucinated span {i}")
                response_copy = hallucination_response.model_copy()
                response_copy.soft_labels = [soft]

                # Rewrite prompt for individual span
                print("Rewriting the prompt...")
                rewritten_response = rewrite_query(self.model, response_copy)

                # Retrieve again using vector DB instead of rag.query
                print("Retrieving more documents via vector DB...")
                new_docs = await vector_db_retrieve_context(
                    rewritten_response.rewritten_question, collection_ids=collection_ids
                )
                docs = filter_docs(docs, new_docs)
        else:
            # Handle all hallucinated spans together in one rewrite
            print("Rewriting for all hallucinated spans together...")
            rewritten_response = rewrite_query(self.model, hallucination_response)

            # Retrieve using combined rewritten query via vector DB
            print("Retrieving more documents via vector DB...")
            new_docs = await vector_db_retrieve_context(
                rewritten_response.rewritten_question, collection_ids=collection_ids
            )
            docs = filter_docs(docs, new_docs)

        # Self-reflection / regeneration stage
        print("Self-reflection...")
        reflected_response = regenerate_answer(self.model, hallucination_response, docs)

        # Ranking between original and reflected answers
        print("Ranking...")
        ranked_response = rank_output(
            self.model,
            question,
            generation_response.answer,
            reflected_response.answer,
            docs,
        )

        if ranked_response.answer_a_score > ranked_response.answer_b_score:
            print(f"Response 1 has high score \n{generation_response.answer}")
        else:
            print(f"Response 2 has high score \n{reflected_response.answer}")

        # Final output
        return {
            "question": question,
            "docs": docs,
            "generation_response": generation_response,
            "hallucination_response": hallucination_response,
            "rewrite_response": rewritten_response,
            "reflected_response": reflected_response,
            "ranked_output": ranked_response,
        }
