import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from ..models.load_model import load_pretrained_model
from ..prompt.default import BaristaPrompt
from ..schema import Order, OrderList

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_JSON_BLOCK_RE = re.compile(r"```json\s*([\s\S]*?)\s*```")


def order_loss(y_true: Order, y_pred: Order) -> float:
    loss = 0
    loss += (y_true.name != y_pred.name) * 5
    loss += (y_true.size != y_pred.size)
    loss += abs(y_true.quantity - y_pred.quantity)
    loss += len(set(y_true.modifiers) ^ set(y_pred.modifiers)) * 0.25
    return loss


def loss_fn(y_true: OrderList, y_pred: OrderList) -> float:
    total_loss = abs(y_true.total_price - y_pred.total_price)
    for item in y_true.items:
        total_loss += min(order_loss(item, item2) for item2 in y_pred.items)
    return total_loss


def load_training_set(train_path: str) -> pd.DataFrame:
    return pd.read_csv(train_path)


# ---------------------------------------------------------------------------
# OpenRouter primitives
# ---------------------------------------------------------------------------

def _build_payload(
    model_id: str,
    messages: list,
    reasoning: bool,
    provider: list[str] | None,
) -> dict:
    payload = {
        "model": model_id,
        "messages": messages,
        "reasoning": {"enabled": reasoning},
    }
    if provider is not None:
        payload["provider"] = {"order": provider}
    return payload


def _call_openrouter(payload: dict, timeout: int = 30) -> dict:
    response = requests.post(
        url=_OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {os.environ['API_KEY']}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=timeout,
    )
    return response.json()


def _parse_order(response: dict) -> OrderList:
    text = response["choices"][0]["message"]["content"]
    matches = _JSON_BLOCK_RE.findall(text)
    return OrderList.model_validate_json(matches[0] if matches else text)


def _run_single(
    row_id,
    order: str,
    model_id: str,
    prompt: BaristaPrompt,
    reasoning: bool,
    provider: list[str] | None,
) -> tuple:
    """Returns (row_id, OrderList | None, error | None)."""
    messages = prompt.format_llm_input(order)
    payload = _build_payload(model_id, messages, reasoning, provider)
    try:
        response = _call_openrouter(payload)
    except requests.exceptions.Timeout:
        return row_id, None, "timeout"
    try:
        return row_id, _parse_order(response), None
    except Exception as e:
        print(f"[{row_id}] parse error: {e}\nresponse: {response}")
        return row_id, None, response


def evaluate_api_model(
    model_id: str,
    prompt: BaristaPrompt,
    dataset: pd.DataFrame,
    reasoning: bool = False,
    provider: list[str] = None,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Evaluate model output based on Training set and custom loss function

    Args:
        model_id (str): Open Router model
        prompt (BaristaPrompt): Prompt Object
        dataset (pd.DataFrame): Input Dataset
        reasoning (bool, optional): Whether to use reasoning. Defaults to False.
        provider (list[str], optional): Open Router Provider. Defaults to None.
        max_workers (int, optional): Max Concurrency. Defaults to 8.

    Returns:
        pd.DataFrame: Output DataFrame
    """
    dataset = dataset.copy()
    dataset["expected_order"] = dataset["expected_json"].apply(
        lambda x: OrderList.model_validate_json(x)
    )

    preds = [None] * len(dataset)
    losses = [500.0] * len(dataset)
    rows = list(dataset.iterrows())

    def process(args):
        idx, (_, row) = args
        row_id, order_list, err = _run_single(
            row.get("id", idx), row["order"], model_id, prompt, reasoning, provider
        )
        return idx, order_list, err, row["expected_order"]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process, item): i for i, item in enumerate(rows)}
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx, order_list, err, expected = future.result()
            if order_list is not None:
                losses[idx] = loss_fn(expected, order_list)
                preds[idx] = order_list.model_dump_json()
            else:
                preds[idx] = str(err)

    dataset["pred"] = preds
    dataset["loss"] = losses
    return dataset




def generate_api_submission(
    model_id: str,
    prompt: BaristaPrompt,
    dataset: pd.DataFrame,
    output_path: str,
    reasoning: bool = False,
    provider: list[str] = None,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Generates API Submission for Kaggle

    Args:
        model_id (str): Open Router model Name
        prompt (BaristaPrompt): Prompt
        dataset (pd.DataFrame): Dataset
        output_path (str): Output path for submission
        reasoning (bool, optional): Whether to use reasoning. Defaults to False.
        provider (list[str], optional): OpenRouter model provider. Defaults to None.
        max_workers (int, optional): Max Concurrency. Defaults to 8.

    Returns:
        pd.DataFrame: _description_
    """
    dataset = dataset.copy()

    num_lines = 0
    if Path(output_path).exists():
        num_lines = len(open(output_path).readlines())
        print("Number of lines", num_lines)
        if num_lines:
            dataset = dataset.iloc[num_lines - 1:].copy()

    rows = list(dataset.iterrows())

    def process(item):
        _, row = item
        row_id, order_list, err = _run_single(
            row["id"], row["order"], model_id, prompt, reasoning, provider
        )
        pred = order_list.model_dump_json() if order_list is not None else json.dumps({"error": str(err)})
        return row_id, pred

    with open(output_path, "a") as f:
        if not num_lines:
            f.write("id,predicted_json\n")

        # executor.map preserves submission order, safe to write sequentially
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for row_id, pred in tqdm(executor.map(process, rows), total=len(rows)):
                f.write(f"{row_id},{pred}\n")

    return dataset
