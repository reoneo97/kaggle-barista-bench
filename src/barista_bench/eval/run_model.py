import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pydantic import ValidationError
from tqdm import tqdm

from ..models.load_model import load_pretrained_model
from ..prompt.default import BaristaPrompt
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
    model = load_pretrained_model()


# def _extract_from_code_blocks(text: str) -> Optional[Any]:
#     """Extract JSON from markdown code blocks."""
#     # Look for ```json or ``` code blocks
#     patterns = [
#         r'```json\s*([\s\S]*?)\s*```',
#     ]

#     for pattern in patterns:
#         matches = re.findall(pattern, text)
#         for match in matches:
#             try:
#                 cleaned = RobustJSONParser._clean_json_string(match)
#                 return json.loads(cleaned)
#             except:
#                 continue
#     return None


def evaluate_api_model(
    model_id: str, prompt: BaristaPrompt, dataset: pd.DataFrame,
    reasoning: bool = False, provider: list[str] = None
):
    # df = load_training_set(train_path=train_path)
    dataset = dataset.copy()
    dataset['expected_order'] = dataset['expected_json'].apply(
        lambda x: OrderList.model_validate_json(x)
    )
    preds = []
    losses = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        order = row['order']
        messages = prompt.format_llm_input(order)

        payload = {
            "model": model_id,
            "messages": messages,
            'reasoning':{
                'enabled': reasoning
            },
            # 'provider':provider
        }
        if provider is not None:
            dict_ = {}
            dict_['order'] = provider
            payload['provider'] = dict_
        # print(payload)
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ['API_KEY']}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(payload),
                timeout=30,
            )
            out_response = response.json()
        except requests.exceptions.Timeout:
            losses.append(500)
            preds.append({'text':'Error with API'})
            continue
        try:
        
            out_message = out_response['choices'][0]['message']['content']
            ptn =  r'```json\s*([\s\S]*?)\s*```'
            matches = re.findall(ptn, out_message)
            
            if not matches:
                y_pred= OrderList.model_validate_json(out_message)
            else:
                y_pred= OrderList.model_validate_json(matches[0])
            
            losses.append(loss_fn(row['expected_order'], y_pred))
            preds.append(y_pred.model_dump_json())
        except Exception as e:
            print(e)
            losses.append(500)
            print("Error in response:", out_response)
            preds.append(out_response)

    dataset['pred'] = preds
    dataset['loss'] = losses
    
    return dataset


def generate_test_results(
        model_id: str, prompt: BaristaPrompt,
        dataset: pd.DataFrame, reasoning:bool = False,
        provider:list[str] = None
    ):
    dataset = dataset.copy()


def generate_api_submission(
        model_id: str, prompt: BaristaPrompt,
        dataset: pd.DataFrame,  output_path: str, reasoning:bool = False,
        provider:list[str] = None,
    ):

    num_lines = 0 
    if Path(output_path).exists():
        num_lines = len(open(output_path,'r').readlines())
        print('Number of lines', num_lines)
        if num_lines:
            dataset = dataset.iloc[num_lines-1:].copy()

    with open(output_path, 'a') as f:

        if not num_lines:
            f.write("row,order\n")

        for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
            order = row["order"]
            messages = prompt.format_llm_input(order)

            payload = {
                "model": model_id,
                "messages": messages,
                "reasoning": {"enabled": reasoning},
                # 'provider':provider
            }
            if provider is not None:
                dict_ = {}
                dict_["order"] = provider
                payload["provider"] = dict_
            # print(payload)
            try:
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.environ['API_KEY']}",
                        "Content-Type": "application/json",
                    },
                    data=json.dumps(payload),
                    timeout=30,
                )
                out_response = response.json()
            except requests.exceptions.Timeout:
                pred = '{"text": "Error with API"}'
                continue
            try:

                out_message = out_response["choices"][0]["message"]["content"]
                ptn = r"```json\s*([\s\S]*?)\s*```"
                matches = re.findall(ptn, out_message)

                if not matches:
                    y_pred = OrderList.model_validate_json(out_message)
                else:
                    y_pred = OrderList.model_validate_json(matches[0])

                pred = y_pred.model_dump_json()
            except Exception as e:
                print(e)
                print("Error in response:", out_response)
                pred = out_response

            f.write(f"{row['id']},{pred}\n")

    return dataset


def generate_api_submission2(
    model_id: str,
    prompt: BaristaPrompt,
    dataset: pd.DataFrame,
    output_path: str,
    reasoning: bool = False,
    provider: list[str] = None,
):

    num_lines = 0
    if Path(output_path).exists():
        num_lines = len(open(output_path, "r").readlines())
        print("Number of lines", num_lines)
        if num_lines:
            dataset = dataset.iloc[num_lines - 1 :].copy()

    with open(output_path, "a") as f:

        if not num_lines:
            f.write("id,predicted_json\n")

        for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
            order = row["order"]
            messages = prompt.format_llm_input(order)

            payload = {
                "model": model_id,
                "messages": messages,
                "reasoning": {"enabled": reasoning},
                # 'provider':provider
            }
            if provider is not None:
                dict_ = {}
                dict_["order"] = provider
                payload["provider"] = dict_
            # print(payload)
            try:
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.environ['API_KEY']}",
                        "Content-Type": "application/json",
                    },
                    data=json.dumps(payload),
                    timeout=30,
                )
                out_response = response.json()
            except requests.exceptions.Timeout:
                pred = '{"text": "Error with API"}'
                continue
            try:

                out_message = out_response["choices"][0]["message"]["content"]
                ptn = r"```json\s*([\s\S]*?)\s*```"
                matches = re.findall(ptn, out_message)

                if not matches:
                    y_pred = OrderList.model_validate_json(out_message)
                else:
                    y_pred = OrderList.model_validate_json(matches[0])

                pred = y_pred.model_dump_json()
            except Exception as e:
                print(e)
                print("Error in response:", out_response)
                pred = out_response

            f.write(f"{row['id']},{pred}\n")

    return dataset


def eval_on_api():
    pass
