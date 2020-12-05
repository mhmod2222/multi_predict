FROM python:3.8-slim

# Install mysqlclient (must be compiled).
RUN apt-get update -qq \
    && apt-get install --no-install-recommends --yes \
        build-essential \
        default-libmysqlclient-dev \
        # Necessary for mysqlclient runtime. Do not remove.
        libmariadb3 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir mysqlclient \
    && apt-get autoremove --purge --yes \
        build-essential \
        default-libmysqlclient-dev

# Install packages that do not require compilation.
#RUN python3 -m pip install --no-cache-dir \
#      numpy scipy pandas matplotlib

RUN apt-get update && \
     apt-get -y --no-install-recommends install \
     libgomp1

COPY . /app
WORKDIR /app

RUN python3 -m pip install --no-cache-dir -r requirements.txt

EXPOSE 5003 
ENTRYPOINT [ "python" ] 
CMD [ "main.py" ]