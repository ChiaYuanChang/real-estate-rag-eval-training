from pydantic import BaseModel, Field


class RealEstateTagEvaluation(BaseModel):
    """
    Evaluation result for Real Estate Image-Text alignment.
    Assesses the quality of generated tags based on visual evidence.
    """

    confidence_score: int = Field(
        ...,
        ge=0,
        le=5,
        description=(
            "A quantitative score (0-5) representing the confidence level in the accuracy "
            "and relevance of the generated tags against the image.\n"
            "0: Fail/Hallucination\n"
            "1: Low (Many errors)\n"
            "2: Below Average\n"
            "3: Average (Basic accuracy)\n"
            "4: High (Specific & Accurate)\n"
            "5: Perfect (Flawless alignment)"
        )
    )

    reasoning: str = Field(
        ...,
        min_length=20,
        description=(
            "A concise analytical explanation justifying the confidence score. "
            "Must explicitly address: 1. Correctness 2. Specificity 3. Redundancy "
            "4. Spatial Signals 5. Hallucination. "
            "Explain exactly what matched or failed to match in the image."
        )
    )
