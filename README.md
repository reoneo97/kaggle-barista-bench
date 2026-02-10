# Barista Bench Agent


This repository is to store code for the [Barista Bench](https://www.kaggle.com/competitions/barista-bench) kaggle competition.




## Installation

1. Install `uv`
2. Run `uv sync` in root directory


## Experiments
1. Start with a very simple default logic with some simple prompting and prompt adjustments
2. Compare against models based on parameter size 
3. Use structured output parsing for better model behavior
4. Two staged agent workflow - Take order before calculating order
   - Take Order
   - Calculate Price
5. Finetune LLM model to deal with this task?


## Scoring
- Scoring is done based on the compatibility of the JSON string compared to the result
- Total Price - Exact Match
- Drink Name and Size
- List of Modifiers


### 10th Feb
- Looked at some of the results and interestingly the one with the highest loss came from mathematical errors compared to problems with the JSON

## Interesting Links
https://www.kaggle.com/code/atharva0577/llm-barista-prefix-caching-gemma