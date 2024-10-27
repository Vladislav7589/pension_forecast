import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder, HashingEncoder

# Функция для удаления выбросов на основе стандартного отклонения
def remove_outliers(df, column, z_thresh=3):
    mean_val = df[column].mean()
    std_dev = df[column].std()
    df = df[(np.abs(df[column] - mean_val) <= z_thresh * std_dev)]
    return df

def preprocess_data(client_path, transaction_path, target_column='erly_pnsn_flg'):
    # Загрузка данных
    trnsctns_ops_trn = pd.read_csv(transaction_path, sep=';', encoding='cp1251')
    cntbtrs_clnts_ops_trn = pd.read_csv(client_path, sep=';', encoding='cp1251', low_memory=False)

    # Копируем данные транзакций и добавляем временные признаки
    transactions = trnsctns_ops_trn.copy()
    transactions['oprtn_date'] = pd.to_datetime(transactions['oprtn_date'], errors='coerce')
    transactions['year'] = transactions['oprtn_date'].dt.year
    transactions['month'] = transactions['oprtn_date'].dt.month
    transactions['day'] = transactions['oprtn_date'].dt.day
    transactions['weekday'] = transactions['oprtn_date'].dt.weekday

    # Агрегирование данных по каждому клиенту
    agg_data = transactions.groupby('accnt_id').agg({
        'mvmnt_type': 'mean',  # Доля приходов
        'oprtn_date': 'count',  # Количество транзакций
        'year': 'nunique',      # Уникальные года
        'month': 'nunique',     # Уникальные месяцы
        'day': 'nunique'        # Уникальные дни
    })
    agg_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in agg_data.columns.values]
    agg_data.reset_index(inplace=True)

    # Заполнение пропусков в колонке 'city' значением 'Non_city'
    cntbtrs_clnts_ops_trn['city'] = cntbtrs_clnts_ops_trn['city'].fillna('Non_city')

    # Создание колонки, показывающей сколько лет осталось до пенсии
    cntbtrs_clnts_ops_trn['age_retirement'] = cntbtrs_clnts_ops_trn['pnsn_age'] - cntbtrs_clnts_ops_trn['prsnt_age']
    cntbtrs_clnts_ops_trn['retirement_status_numeric'] = cntbtrs_clnts_ops_trn['age_retirement'].apply(
        lambda x: -1 if x < 0 else (0 if x == 0 else 1)
    )
    # Удаление ненужных признаков
    cntbtrs_clnts_ops_trn.drop(columns=['brth_yr', 'dstrct', 'sttlmnt', 'prvs_npf', 'pnsn_age', 'prsnt_age'], inplace=True)

    # One-Hot Encoding для признаков с малым числом уникальных значений
    one_hot_columns = ['slctn_nmbr', 'addrss_type', 'accnt_status']
    cntbtrs_clnts_ops_trn = pd.get_dummies(cntbtrs_clnts_ops_trn, columns=one_hot_columns)

    # Label Encoding для признаков с большим числом категорий
    label_encode_columns = ['gndr', 'rgn', 'city', 'okato']
    for col in label_encode_columns:
        le = LabelEncoder()
        cntbtrs_clnts_ops_trn[col] = le.fit_transform(cntbtrs_clnts_ops_trn[col].astype(str))

    # Hash Encoding для признаков с очень большим числом категорий
    hash_encode_columns = ['pstl_code']
    he = HashingEncoder(n_components=10)  # Количество хеш-колонок можно изменить
    hash_encoded = he.fit_transform(cntbtrs_clnts_ops_trn[hash_encode_columns])
    cntbtrs_clnts_ops_trn = pd.concat([cntbtrs_clnts_ops_trn, hash_encoded], axis=1)
    cntbtrs_clnts_ops_trn.drop(columns=hash_encode_columns, inplace=True)

    # Target Encoding для категорий, связанных с целевой переменной
    target_encoder = TargetEncoder()
    cntbtrs_clnts_ops_trn['brth_plc'] = target_encoder.fit_transform(cntbtrs_clnts_ops_trn['brth_plc'], cntbtrs_clnts_ops_trn[target_column])

    # Бинарное кодирование для столбцов с 'Yes'/'No' значениями
    binary_columns = ['phn', 'email', 'lk', 'assgn_npo', 'assgn_ops']
    for col in binary_columns:
        cntbtrs_clnts_ops_trn[col] = cntbtrs_clnts_ops_trn[col].apply(lambda x: 1 if x == 'Yes' else 0)

    # Обработка даты
    date_columns = ['accnt_bgn_date']
    for col in date_columns:
        cntbtrs_clnts_ops_trn[col] = pd.to_datetime(cntbtrs_clnts_ops_trn[col], errors='coerce')
        cntbtrs_clnts_ops_trn[f'{col}_year'] = cntbtrs_clnts_ops_trn[col].dt.year
        cntbtrs_clnts_ops_trn[f'{col}_month'] = cntbtrs_clnts_ops_trn[col].dt.month
        cntbtrs_clnts_ops_trn[f'{col}_day'] = cntbtrs_clnts_ops_trn[col].dt.day
        cntbtrs_clnts_ops_trn.drop(columns=[col], inplace=True)

    # Преобразование числового признака 'cprtn_prd_d'
    cntbtrs_clnts_ops_trn['cprtn_prd_d'] = pd.to_numeric(cntbtrs_clnts_ops_trn['cprtn_prd_d'], errors='coerce')

    # Объединение данных клиентов с агрегированными данными транзакций
    merged_data = cntbtrs_clnts_ops_trn.merge(agg_data, on='accnt_id', how='left')

    # Копирование и очистка объединенных данных
    data_cleaned = merged_data.copy()
    data_cleaned = data_cleaned[(data_cleaned['mvmnt_type'] >= 0) & (data_cleaned['mvmnt_type'] <= 1)]
    data_cleaned = data_cleaned.loc[:, data_cleaned.nunique() > 1]
    data_cleaned = remove_outliers(data_cleaned, 'oprtn_date', z_thresh=3)
    data_cleaned = data_cleaned.dropna(how='all')

    return data_cleaned

# Пример использования
# data_cleaned = preprocess_data('/path/to/trnsctns_ops_trn.csv', '/path/to/cntrbtrs_clnts_ops_trn.csv')
# print(data_cleaned.head())
