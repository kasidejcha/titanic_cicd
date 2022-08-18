FROM python:3.9.4

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' ml-api-user

WORKDIR /opt/titanic_api

# Install requirements, including from Gemfury
ADD ./titanic_api /opt/titanic_api/
RUN pip install --upgrade pip
RUN pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple -r /opt/titanic_api/requirements.txt

RUN chmod +x /opt/titanic_api/run.sh
RUN chown -R ml-api-user:ml-api-user ./

USER ml-api-user

EXPOSE 8001

CMD ["bash", "./run.sh"]