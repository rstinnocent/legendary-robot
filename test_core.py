import pandas as pd
import pytest

from core import (
    LLMBackend,
    UnsafeCodeError,
    answer_question,
    check_code_safety,
    load_files,
    run_generated_code,
    sanitize_name,
)


def test_sanitize_name():
    assert sanitize_name("Sales Q1.csv") == "sales_q1"
    assert sanitize_name("2024 Orders.xlsx") == "df_2024_orders"


def test_load_files_csv():
    files = {"data.csv": b"a,b\n1,2\n3,4\n"}
    frames = load_files(files)
    assert "data" in frames
    assert list(frames["data"].columns) == ["a", "b"]
    assert len(frames["data"]) == 2


def test_load_files_handles_name_collisions():
    files = {"orders.csv": b"a\n1\n", "Orders.csv": b"a\n2\n"}
    frames = load_files(files)
    assert len(frames) == 2  # both loaded, no overwrite


def test_check_code_safety_blocks_dangerous_code():
    with pytest.raises(UnsafeCodeError):
        check_code_safety("import os\nos.system('ls')")
    with pytest.raises(UnsafeCodeError):
        check_code_safety("open('/etc/passwd').read()")
    with pytest.raises(UnsafeCodeError):
        check_code_safety("x.__class__.__bases__")
    check_code_safety("result = df['a'].sum()")  # should not raise


def test_run_generated_code_scalar_result():
    frames = {"orders": pd.DataFrame({"amount": [10, 20, 30]})}
    res = run_generated_code("result = orders['amount'].sum()", frames)
    assert res.error is None
    assert res.result == 60


def test_run_generated_code_strips_markdown_fences():
    frames = {"orders": pd.DataFrame({"amount": [1, 2]})}
    fenced = "```python\nresult = orders['amount'].sum()\n```"
    res = run_generated_code(fenced, frames)
    assert res.error is None
    assert res.result == 3


def test_run_generated_code_cross_file_join():
    orders = pd.DataFrame({"customer_id": [1, 2, 1], "amount": [100, 200, 50]})
    customers = pd.DataFrame({"customer_id": [1, 2], "region": ["West", "East"]})
    frames = {"orders": orders, "customers": customers}
    code = (
        "merged = orders.merge(customers, on='customer_id')\n"
        "result = merged.groupby('region')['amount'].sum()"
    )
    res = run_generated_code(code, frames)
    assert res.error is None
    assert res.result["West"] == 150
    assert res.result["East"] == 200


def test_run_generated_code_produces_chart():
    frames = {"orders": pd.DataFrame({"month": ["Jan", "Feb"], "amount": [10, 20]})}
    code = "plt.bar(orders['month'], orders['amount'])\nresult = 'Monthly revenue chart'"
    res = run_generated_code(code, frames)
    assert res.error is None
    assert res.figure is not None


def test_run_generated_code_handles_runtime_error_gracefully():
    frames = {"orders": pd.DataFrame({"amount": [1, 2, 3]})}
    res = run_generated_code("result = orders['does_not_exist'].sum()", frames)
    assert res.error is not None
    assert res.result is None


class FakeBackend(LLMBackend):
    """Stand-in for a real LLM in tests — returns a canned code string."""

    def __init__(self, code: str):
        self.code = code

    def generate(self, prompt: str) -> str:
        return self.code


def test_answer_question_end_to_end():
    frames = {"orders": pd.DataFrame({"amount": [5, 15]})}
    backend = FakeBackend("result = orders['amount'].mean()")
    res = answer_question("What's the average order amount?", frames, backend)
    assert res.error is None
    assert res.result == 10.0
