from pathlib import Path
from ..schema import OrderList

class BaristaPrompt:

    def __init__(self, prompt_file: str, examples: list[tuple[str, str]]):
        self.menu = load_menu()
        self.system_prompt = load_prompt(prompt_file).format(
            MENU=self.menu, JSON_SCHEMA=OrderList.model_json_schema()
        )
        # print(self.system_prompt)
        self.few_shot_context = self.generate_few_shot(examples)
        # print(self.menu)


    def format_llm_input(self, customer_order: str) -> str:
        messages = [
            {
                "role": "system",
                "content":self.system_prompt
            },
            {
                "role": "user",
                "content" : f"""{self.few_shot_context} now do the same for this customer order {customer_order}
                """
            }
        ]
        return messages

    def generate_few_shot(self, examples: list[tuple[str, str]]) -> str:

        few_shot = """
Here are some examples of customer orders and the corresponding JSON output:
"""
        for customer_order, json_output in examples:
            few_shot += f"""
Customer Order:
{customer_order}

JSON Output:
{json_output}
"""
        return few_shot

    def __str__(self):
        return self.system_prompt

    


def load_menu():
    current_path = Path(__file__).parent

    md_path = Path(current_path) / 'menu.md'
    with open(md_path) as f:
        return f.read()
    


def load_prompt(filename: str):
    current_path = Path(__file__).parent
    with open(current_path / filename) as f:
        return f.read()