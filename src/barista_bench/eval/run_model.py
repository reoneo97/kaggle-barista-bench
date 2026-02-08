import pandas as pd
import numpy as np
from ..models.load_model import load_pretrained_model
import requests 
from ..schema import Order, OrderList

def order_loss(y_true: Order, y_pred: Order) -> float:
    loss = 0 
    loss += (y_true.name != y_pred.name) * 5
    loss += (y_true.size != y_pred.size) 
    loss += abs(y_true.quantity - y_pred.quantity)
    loss += len(set(y_true.modifiers) ^ set(y_pred.modifiers)) * 0.25
    
    return loss

def loss_fn(y_true: OrderList, y_pred: OrderList):
    total_loss = abs(y_true.total_price - y_pred.total_price)
    
    for i, item in enumerate(y_true.items):
        loss = float('inf')
        for j, item2 in enumerate(y_pred.items):
            loss = min(loss, order_loss(item, item2))
        total_loss += loss

    return total_loss


def load_training_set(train_path: str, train_pct = 0.8):
    df = pd.read_csv(train_path)
    np.random.seed(123)


    # Train Test Split
    # train_size = int(len(df) * train_pct) 
    # idxs = np.random.permutation(df.index)
    # train_df = df.iloc[idxs[:train_size], :]
    # val_df = df.iloc[idxs[train_size:], :]


    return df


def evaluate_pretrained_model(prompt: str, train_path: str, model_id: str):
    df = load_training_set()
    model = load_pretrained_model()

def evaluate_api_model(prompt: str, train_path: str, model_id: str):
    df = load_training_set()

    pass




def generate_submission():
    pass


def eval_on_api():
    pass
