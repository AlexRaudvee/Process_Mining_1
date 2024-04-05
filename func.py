import shutil
import pandas as pd
from tqdm import tqdm
import numpy as np
from ast import literal_eval
import json

custom_format = "{desc}: {percentage:.0f}%\x1b[33m|\x1b[0m\x1b[32m{bar}\x1b[0m\x1b[31m{remaining}\x1b[0m\x1b[33m|\x1b[0m {n}/{total} [{elapsed}<{remaining}]"


def saver(df: pd.DataFrame, path_name: str):
    """
    Saves a pandas dataframe to a csv file in chunks.

    Args:
        df (pd.DataFrame): The dataframe to save.
        path_name (str): The path and filename of the csv file.

    Returns:
        None

    """
    chunks = np.array_split(df.index, 100) # split into 100 chunks

    for chunck, subset in enumerate(tqdm(chunks, desc=f"Storing of data ", dynamic_ncols=True, bar_format=custom_format, ascii=' -')):
        if chunck == 0: # first row
            df.loc[subset].to_csv(path_name, mode='w', index=True)
        else:
            df.loc[subset].to_csv(path_name, header=None, mode='a', index=True)

def saver_json(df: pd.DataFrame, path_name: str):
    """
    Saves a pandas dataframe to a json file in chunks.

    Args:
        df (pd.DataFrame): The dataframe to save.
        path_name (str): The path and filename of the json file.

    Returns:
        None

    """
    chunks = np.array_split(df.index, 100) # split into 100 chunks

    for chunck, subset in enumerate(tqdm(chunks, desc=f"Storing of data ", dynamic_ncols=True, bar_format=custom_format, ascii=' -')):
        if chunck == 0:  # first chunk
            df.loc[subset].to_json(path_name, orient='records')
        else:
            df.loc[subset].to_json(path_name, orient='records', lines=True, mode='a')

def print_terminal_width_symbol(symbol):
    """
    Prints a symbol repeated for the terminal width.

    Args:
        symbol (str): The symbol to print.

    Returns:
        None

    """
    # Get the terminal width
    terminal_width, _ = shutil.get_terminal_size()

    # Print the symbol repeated for the terminal width on a single line
    print(symbol * terminal_width)

def print_centered_text(text, symbol=" "):
    """
    Prints centered text in the terminal.

    Args:
        text (str): The text to print.
        symbol (str, optional): The symbol to print before and after the text.
            Defaults to a space.

    Returns:
        None

    """
    # Get the terminal width
    terminal_width, _ = shutil.get_terminal_size()

    # Calculate the number of spaces before and after the text
    num_spaces_before = (terminal_width - len(text)) // 2
    num_spaces_after = terminal_width - len(text) - num_spaces_before

    # Print the symbol and the text with spaces
    print(symbol * num_spaces_before + text + symbol * num_spaces_after)


def extract_events(trace_list):
    try:
        trace_list_evaluated = literal_eval(
            trace_list) if isinstance(trace_list, str) else trace_list
        return [event['concept:name'] for event in trace_list_evaluated]
    except (ValueError, SyntaxError):
        # In case of any error during conversion, return an empty list or handle as needed
        print(f"Error processing trace: {trace_list}")
        return []


# Function to calculate the Jaccard Similarity for two sets
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def safely_parse_json(data):
    if isinstance(data, str):
        try:
            return json.loads(data)
        except ValueError as e:
            print(f"Error parsing JSON: {e}")
            return None  # or {}, [] depending on expected data type
    return data