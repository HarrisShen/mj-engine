import json
from typing import Any, Callable


def numeric_input(
        description: str,
        dtype: type,
        default: int | float,
        min_val: int | float = None,
        max_val: int | float = None,
        choice: list | dict | None = None,
        retry: int = 0) -> Any:
    """Ask user for a number as input, retry if invalid input is given"""
    if dtype is not int and dtype is not float:
        raise ValueError(f'Expecting type "int" or "float" for "dtype"')
    if (min_val is not None and default < min_val) or (max_val is not None and default > max_val):
        raise ValueError(f"Default value {default} is not in range ({min_val}, {max_val})")
    if choice is not None:
        if default not in choice:
            raise ValueError(f"Default value {default} is not in choice {choice}")
        if min_val is not None and min_val not in choice:
            raise ValueError(f"Cannot set range ('min_val' and 'max_val') when 'choice' is given")

    if isinstance(choice, dict):
        prompt = f"{description} - "
        prompt += ", ".join([str(k) + ". " + str(v) + ("*" if default == k else "") for k, v in choice.items()])
        prompt += ": "
    else:
        prompt = f"{description} (default: {default}): "
    for _ in range(retry + 1):
        try:
            num = dtype(input(prompt))
            if min_val is not None and num < min_val:
                raise ValueError
            if max_val is not None and num > max_val:
                raise ValueError
            if choice is not None and num not in choice:
                raise ValueError
            return choice[num] if isinstance(choice, dict) else num
        except ValueError:
            prompt = f"Invalid input, please enter a number"
            if min_val is not None:
                prompt += f" >= {min_val}"
            if max_val is not None:
                prompt += f" <= {max_val}"
            if choice is not None:
                prompt += f" in {choice}"
    print(f"Using default value {default}")
    return choice[default] if isinstance(choice, dict) else default


def unit_interval_input(
        description: str,
        default: int | float,
        retry: int = 0) -> float:
    return numeric_input(description, float, default, min_val=0, max_val=1, retry=retry)


def bool_input(description: str, default: bool = False, retry: int = 0) -> bool:
    option = "Y/n" if default else "y/N"
    prompt = f"{description} {option}: "
    for _ in range(retry + 1):
        val = input(prompt).lower()
        if val in ["y", "n"]:
            return val == "y"
    print(f"Using default value {default}")
    return default


def string_input(description: str, choices: list[str], default: str | None = None, retry: int = 0) -> str:
    # TODO: implement this as a partial substitution for choice as dict in "numeric_input"
    pass


def confirm_inputs(name: str, content: Any, callback: Callable, **kwargs) -> Any:
    print(f"{name}: \n{json.dumps(content, indent=2)}")
    while True:
        confirm = input("Confirm? (Y - yes, n - no, e - abort & exit): ")
        if confirm in ["Y", "y", ""]:
            break
        elif confirm in ["N", "n"]:
            return callback(**kwargs)
        elif confirm in ["E", "e"]:
            print("Cancelled")
            exit(0)
        else:
            print("Invalid input, please enter Y, n or e")
    return content
