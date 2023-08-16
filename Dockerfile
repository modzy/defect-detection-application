FROM python:3.9-slim-bullseye

# set directory for app
WORKDIR /opt/app

# define argument for port
ENV PORT=8000

# copy requirements file
COPY ./flask-app/requirements.txt .
# pip install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy additional flask app files
COPY ./flask-app/static/ static/
COPY ./flask-app/templates/ templates/
COPY ./flask-app/__init__.py __init__.py
COPY ./flask-app/util.py util.py
COPY ./flask-app/app.py app.py

# entrypoint
CMD python app.py --port $PORT