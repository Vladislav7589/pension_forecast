import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from scipy.stats import mode
import joblib


def format_predictions(predictions_df):
    # Преобразование второй колонки из 0/1 в "Нет"/"Да"
    predictions_df['early_retirement'] = predictions_df['early_retirement'].apply(lambda x: 'Да' if x == 1 else 'Нет')

    # Преобразование третьей колонки в зависимости от значения
    def format_retirement_status(years):
        if years < 0:
            return f"{abs(years)} лет НА пенсии"
        elif years == 0:
            return "Вышел на пенсию"
        else:
            return f"{years} лет ДО пенсии"

    predictions_df['retirement_status'] = predictions_df['retirement_status'].apply(format_retirement_status)
    return predictions_df


def test_2(data_cleaned, target_column='erly_pnsn_flg', id_column='accnt_id'):
    # Загрузка обученных моделей
    isolation_forest = joblib.load('isolation_forest_model.pkl')
    elliptic_envelope = joblib.load('elliptic_envelope_model.pkl')

    # Подготовка данных для тестирования (исключаем ID и целевую переменную)
    data_copy = data_cleaned.copy()
    X_test = data_copy.drop(columns=['clnt_id', id_column, target_column, 'age_retirement'])

    # Предсказания каждой модели
    y_pred_if = isolation_forest.predict(X_test)
    y_pred_ee = elliptic_envelope.predict(X_test)

    # Ансамблевое предсказание (голосование)
    predictions = np.vstack((y_pred_if, y_pred_ee)).T
    y_pred = mode(predictions, axis=1)[0].flatten()
    y_pred_final = np.where(y_pred == 1, 0, 1)  # Преобразуем к значениям 0 и 1

    # Создание DataFrame с предсказаниями
    predictions_df = pd.DataFrame({
        id_column: data_cleaned[id_column],
        target_column: y_pred_final
    })

    predictions_df.to_csv('test_predictions_with_ids.csv', index=False, sep=',', encoding='utf-8')

    return predictions_df

def test_3(data_cleaned, target_column='erly_pnsn_flg', id_column='accnt_id'):
    # Загрузка обученных моделей
    isolation_forest = joblib.load('/Users/vlad/PycharmProjects/PensionForecast/pension_forecast/pension_forecast_app/sd/isolation_forest_model.pkl')
    elliptic_envelope = joblib.load('/Users/vlad/PycharmProjects/PensionForecast/pension_forecast/pension_forecast_app/sd/elliptic_envelope_model.pkl')

    # Подготовка данных для тестирования (исключаем ID и целевую переменную)
    data_copy = data_cleaned.copy()
    X_test = data_copy.drop(columns=['clnt_id', id_column, target_column, 'age_retirement'])
    y_test = data_copy[target_column].apply(lambda x: 1 if x == 0 else -1)  # 1 - класс 0, -1 - класс 1

    # Предсказания каждой модели
    y_pred_if = isolation_forest.predict(X_test)
    y_pred_ee = elliptic_envelope.predict(X_test)

    # Ансамблевое предсказание (голосование)
    predictions = np.vstack((y_pred_if, y_pred_ee)).T
    y_pred = mode(predictions, axis=1)[0].flatten()
    y_pred_final = np.where(y_pred == 1, 0, 1)  # Преобразуем к значениям 0 и 1
    y_test = np.where(y_test == 1, 0, 1)  # Преобразуем к значениям 0 и 1

    # Оценка точности и вывод отчета классификации
    accuracy = accuracy_score(y_test, y_pred_final)
    f1 = f1_score(y_test, y_pred_final, average='weighted')
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(classification_report(y_test, y_pred_final, target_names=['Anomalous', 'Normal']))

    # Создание DataFrame с предсказаниями
    predictions_df = pd.DataFrame({
        'Client_ID': data_cleaned[id_column],
        'early_retirement': y_pred_final,
        'retirement_status': data_cleaned['age_retirement']
    })

    formatted_predictions_df = format_predictions(predictions_df)
    formatted_predictions_df.to_csv('pension_forecast_app/sd/test_predictions_with_ids.csv', index=False, sep=',', encoding='utf-8')

    sampled_data = pd.concat([
        formatted_predictions_df[formatted_predictions_df['early_retirement'] == 'Нет'].sample(3, random_state=42),
        # выбираем 3 строки с флагом 0
        formatted_predictions_df[formatted_predictions_df['early_retirement'] == 'Да'].sample(2, random_state=42)
        # выбираем 2 строки с флагом 1
    ])

    # Сохранение выборки в CSV
    sampled_data.to_csv('sampled_data.csv', index=False, encoding='utf-8')
    return sampled_data, f1

# if __name__ == '__main__':
#     data_clean = preprocess_data('/home/german-rivman/VScodeProjects/GermanHack/train_data/cntrbtrs_clnts_ops_trn.csv', '/home/german-rivman/VScodeProjects/GermanHack/train_data/trnsctns_ops_trn.csv')
#     test_3(data_clean)