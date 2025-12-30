import os
from typing import Literal

import dspy
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Using OpenRouter. Switch to another LLM provider as needed
lm = dspy.LM(
    model="openrouter/google/gemini-2.0-flash-001",
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
dspy.configure(lm=lm)


class RecipeFeatureInput(BaseModel):
    id: int
    ingredients: list[str] | None = None


class FeatureExtractor(dspy.Signature):
    """
    Given a recipe's list of ingredients, extract the relevant features.
    - Treat eggs as vegetarian, and fish as non-vegetarian
    - Nuts include any kind of tree nuts, peanuts/ground nuts
    - Gluten includes wheat, barley, rye, and any foods made from these grains
    """

    id: int = dspy.InputField()
    ingredients: list[str] = dspy.InputField()
    is_vegetarian: bool = dspy.OutputField()
    has_nuts: bool = dspy.OutputField()
    has_dairy: bool = dspy.OutputField()
    has_eggs: bool = dspy.OutputField()
    category: Literal["food", "beverage"] = dspy.OutputField()


class Extract(dspy.Module):
    def __init__(self):
        self.extractor = dspy.Predict(FeatureExtractor)

    def forward(self, recipe: RecipeFeatureInput):
        return self.extractor(id=recipe.id, ingredients=recipe.ingredients)

    async def aforward(self, recipe: RecipeFeatureInput):
        return await self.extractor.acall(id=recipe.id, ingredients=recipe.ingredients)


if __name__ == "__main__":
    tests = [
        RecipeFeatureInput(
            id=1,
            ingredients=[
                "2 cups of flour",
                "1 cup of sugar",
                "1/2 cup of chopped nuts",
                "1 cup of milk",
                "2 eggs",
            ],
        ),
        RecipeFeatureInput(
            id=2,
            ingredients=[
                "3 oz. Grand Marnier",
                "1 oz. Amaro Averna",
                "Small pat salted butter (about \u00bd teaspoon)",
                "1 cup hot apple cider",
                "1\u00bd to 3 tsp. fresh lemon juice (to taste, depending on the sweetness of your cider)",
                "Garnish: freshly ground pink peppercorns",
                "plus 2 lemon wheels (optional)",
            ],
        ),
    ]
    extractor = Extract()

    for test in tests:
        result = extractor(test)
        print(result)
