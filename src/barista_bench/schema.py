from pydantic import BaseModel, Field



class OrderList(BaseModel):
    items: list[Order]
    total_price: float

class Order(BaseModel):
    name: str
    size: str | None
    quantity: int
    modifiers: list[str]

# TODO: Should try to use Enums to better fix the input


class Order(BaseModel):
    name:str = Field(description='Name of the item from the menu')
    quantity: int =Field(gt=0, description='Number of items')