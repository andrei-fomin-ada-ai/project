import json
import pandas as pd
import numpy as np
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

def load_json(file_path):
    path = Path(file_path)
    try:
        with path.open('r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f'Ошибка: Файл {file_path} не найден.')
        return None
    except json.JSONDecodeError:
        print('Ошибка: Неверный формат JSON.')
        return None


def process_data(json_data) -> pd.DataFrame:
    if not json_data:
        return None

    if isinstance(json_data, dict):
        chat_id = json_data.get('id')
        messages_list = json_data.get('messages', [])
    elif isinstance(json_data, list) and len(json_data) > 0 and 'messages' in json_data[0]:
        chat_id = json_data[0].get('id')
        messages_list = json_data[0].get('messages', [])
    else:
        return None

    if not messages_list:
        return None

    df = pd.json_normalize(messages_list)
    df['chat_id'] = chat_id
    possible_duplicates = ['message_id', 'timestamp', 'sender', 'sender_id', 'reply_to_id']
    df = df.drop(columns=[col for col in possible_duplicates if col in df.columns], errors='ignore')

    df = df.rename(columns={
        'id': 'message_id',
        'date': 'timestamp',
        'from': 'sender',
        'from_id': 'sender_id',
        'reply_to_message_id': 'reply_to_id'
    })

    df = df.loc[:, ~df.columns.duplicated()]

    if 'reply_to_id' in df.columns:
        df['is_reply'] = df['reply_to_id'].notna()
    else:
        df['is_reply'] = False

    if 'type' in df.columns:
        df = df[df['type'] != 'service'].copy()

    required_columns = [
        'message_id', 'timestamp', 'sender', 'sender_id',
        'text', 'is_reply', 'reply_to_id', 'chat_id'
    ]
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    df = df[required_columns].copy()

    def clean_text(x):
        if isinstance(x, list):
            res = ''.join([i.get('text', '') if isinstance(i, dict) else str(i) for i in x]).strip()
            return res if res else np.nan
        elif isinstance(x, str):
            res = x.strip()
            return res if res else np.nan
        return np.nan

    df['text'] = df['text'].apply(clean_text)
    df = df.dropna(subset=['text']).copy()

    df['sender'] = df['sender'].astype(object).fillna('Unknown')
    df['sender_id'] = df['sender_id'].astype(object).fillna('Unknown')
    df['chat_id'] = df['chat_id'].astype(object).fillna('Unknown')

    return df


def save_data(df):
    if df is None or df.empty:
        print('Нет данных для сохранения.')
        return
    
    data_dir = ROOT_DIR / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / 'processed.csv'

    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f'Данные сохранены в {file_path}')


if __name__ == '__main__':
    input_file = 'result.json'

    raw_data = load_json(ROOT_DIR / 'data' / 'raw' / input_file)
    if raw_data:
        df_result = process_data(raw_data)
        if df_result is not None:
            save_data(df_result)
