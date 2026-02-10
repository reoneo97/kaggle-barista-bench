from pydantic import BaseModel, Field


class OrderList(BaseModel):
    items: list[Order]
    total_price: float

class Order(BaseModel):
    name: str
    size: str | None
    quantity: int
    modifiers: list[str]


'''
#TODO:
1. Explore using Enums for better input validation
2. Separate by Food and Drink
3. Convert to Field
4. Lastly see if theres a good way of using the LLM to suggest the schema automatically
'''
# class Order(BaseModel):
#     name:str = Field(description='Name of the item from the menu')
#     quantity: int =Field(gt=0, description='Number of items')