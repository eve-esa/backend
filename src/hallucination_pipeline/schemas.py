from typing import List
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class generation_schema(BaseModel):
    question: str = Field(..., description="Question asked by the user.")
    answer: str = Field(..., description="Answer to the question.")


class SoftLabelDict(TypedDict):
    start: int
    end: int
    prob: float
    reason: str


class hallucination_schema(BaseModel):
    question: str = Field(..., description="Question asked by the user.")
    answer: str = Field(..., description="Answer to the question.")
    soft_labels: List[SoftLabelDict] = Field(
        ..., description="List of soft label spans."
    )


"""
class hallucination_schema(BaseModel):
    question: str = Field(..., description="Question asked by the user.")
    answer: str = Field(..., description="Answer to the question.")
    soft_labels: List[dict] = Field(..., description="Hallucination spans with {'start': 'start index', probobailty: 'between 0 to 1 if it is 0; not hallucinted','end' : 'end index','reason': reason for hallucination}")
    #hard_labels: Optional[List[List[int]]] = Field(None, description="Hard labeled spans: [[start, end], ...]")
"""


class rewrite_schema(BaseModel):
    question: str = Field(..., description="Original user question.")
    rewritten_question: str = Field(
        ..., description="Rewritten question for factual accuracy."
    )


class self_reflect_schema(BaseModel):
    question: str = Field(..., description="Question asked by the user.")
    answer: str = Field(..., description="Answer after self reflection")


class ranking_schema(BaseModel):
    answer_a_score: int = Field(..., ge=0, le=10)
    answer_b_score: int = Field(..., ge=0, le=10)
    justification_a: str
    justification_b: str
