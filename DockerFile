# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект в рабочую директорию
COPY . .

# Запустите миграции
RUN python manage.py migrate

# Открываем порт для приложения
EXPOSE 8000

# Соберите статические файлы
RUN python manage.py collectstatic --noinput

# Команда для запуска приложения
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "pension_forecast.wsgi:application"]

