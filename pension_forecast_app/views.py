import json
import pandas as pd
import joblib
import logging
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def welcome_view(request):
    """
    :return: Приветственный html файл для описания endpoints
    """
    return render(request, 'home.html')


class DatasetPredictionView(APIView):
    """
    View для возвращения предиктивных данных в виде CSV
    """

    def post(self, request):
        try:
            # contributers = request.FILES.get('contributers')
            # contributers = request.FILES.get('contributers')
            logger.info(f"df_contributers: {len(request.FILES)}\n ")
            print(request)
            if not request.FILES:
                return Response({"error": "Файл обязателен"}, status=400)
            for file_name, file in request.FILES.items():
                if file_name == "contributers":
                    df_contributers = self.read_csv_file(file)
            # contributers = request.FILES.get('cntr')

            # transactions = request.FILES.get('transactions')

            # if not contributers:
            #     return Response({"error": "Both files are required"}, status=400)

            # df_contributers = self.read_csv_file(contributers)
            # df_transactions = self.read_csv_file(transactions)

            logger.info(f"df_contributers: {len(df_contributers)}\n ")

            predictions_df, f1 = self.load_model_and_predict(df_contributers)

            # Создадим CSV для ответа
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="predictions.csv"'
            predictions_df.to_csv(response, index=False)
            # df_contributers.to_csv(response, index=False)
            # Добавим F1-score в заголовки ответа
            response['X-F1-Score'] = str(f1)

            return response

        except KeyError as e:
            return JsonResponse({"error": f"KeyError: Column {str(e)} not found in data"}, status=400)

        except pd.errors.MergeError as e:
            return JsonResponse({"error": f"MergeError: {str(e)}"}, status=500)

        except Exception as e:
            return JsonResponse({"error": f"An unexpected error occurred: {str(e)}"}, status=500)

    def read_csv_file(self, file, encoding="cp1251", sep=';'):
        try:
            return pd.read_csv(file, sep=sep, encoding=encoding, index_col=0)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")

    def load_model_and_predict(self, df, model_path='data/multi_output_stacked_ensemble_model.pkl'):
        """
        Загружает модель из файла и делает предсказания для новых данных.

        :param df: Файл CSV с данными для предсказания.
        :param model_path: Путь к файлу модели (по умолчанию 'multi_output_stacked_ensemble_model.pkl').
        :return: DataFrame с предсказанными значениями для целевых колонок и идентификаторами.
        """

        # Загрузка данных для предсказания
        cntbtrs_clnts_ops_trn = df
        columns_to_drop_with_nans = ['slctn_nmbr', 'prvs_npf', 'brth_plc', 'pstl_code', 'addrss_type', 'accnt_bgn_date',
                                     'phn', 'email', 'assgn_npo', 'assgn_ops']
        full_table_new = cntbtrs_clnts_ops_trn.drop(columns=columns_to_drop_with_nans)

        le_gndr = LabelEncoder()
        le_accnt_status = LabelEncoder()
        le_rgn = LabelEncoder()

        full_table_new['gndr_encoded'] = le_gndr.fit_transform(full_table_new['gndr'])
        full_table_new['accnt_status_encoded'] = le_accnt_status.fit_transform(full_table_new['accnt_status'])
        full_table_new['rgn_encoded'] = le_rgn.fit_transform(full_table_new['rgn'])

        full_table_new['dstrct_encoded'] = LabelEncoder().fit_transform(full_table_new['dstrct'])
        full_table_new['city_encoded'] = LabelEncoder().fit_transform(full_table_new['city'])

        full_table_new['sttlmnt_encoded'] = LabelEncoder().fit_transform(full_table_new['sttlmnt'])
        full_table_new['okato_encoded'] = LabelEncoder().fit_transform(full_table_new['sttlmnt'])

        full_table_new['lk_encoded'] = LabelEncoder().fit_transform(full_table_new['lk'])

        columns_to_drop_with_nans = ['rgn', 'accnt_status', 'gndr', 'dstrct', 'city', 'sttlmnt', 'okato', 'lk']
        full_table_new = full_table_new.drop(columns=columns_to_drop_with_nans)

        input_data = full_table_new.dropna()
        # Извлечение идентификаторов и удаление их из признаков
        ids = input_data['clnt_id']
        target_column = ['erly_pnsn_flg']  # Замените на список целевых колонок
        features = full_table_new.drop(columns=target_column + ['clnt_id', 'accnt_id'])
        true_labels = full_table_new[target_column]  # Истинные метки

        # Загрузка обученной модели
        model = joblib.load(model_path)

        # Выполнение предсказания
        predictions = model.predict(features)

        # Создание DataFrame для предсказаний с идентификаторами
        predictions_df = pd.DataFrame(predictions, columns=target_column)
        predictions_df['clnt_id'] = ids  # Добавление идентификаторов к предсказаниям

        # Переупорядочиваем колонки, чтобы 'accnt_id' была первой
        columns_order = ['clnt_id'] + target_column
        predictions_df = predictions_df[columns_order]

        # Расчет F1-score
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=1)
        print(f'F1 Score: {f1:.2f}')

        return predictions_df, f1
