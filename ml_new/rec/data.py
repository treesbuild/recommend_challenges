import logging
import os
import re
from urllib.parse import quote_plus as urlquote

import sqlalchemy
from sqlalchemy.ext.automap import automap_base

logger = logging.getLogger()


def read_env():
    logger.info('reading .env')
    try:
        with open('.env') as f:
            content = f.read()
    except IOError:
        content = ''
    for line in content.splitlines():
        m1 = re.match(r'\A([A-Za-z_0-9]+)=(.*)\Z', line)
        if m1:
            key, val = m1.group(1), m1.group(2)
            m2 = re.match(r"\A'(.*)'\Z", val)
            if m2:
                val = m2.group(1)
            m3 = re.match(r'\A"(.*)"\Z', val)
            if m3:
                val = re.sub(r'\\(.)', r'\1', m3.group(1))

            logger.info(f'setting k:v {key}:{val}')

            os.environ.setdefault(key, val)


read_env()

db_user = os.environ["DB_USER"]
db_pass = os.environ["DB_PASS"]
db_name = os.environ["DB_NAME"]
db_port = os.environ.get("DB_PORT", '5432')
db_socket_dir = os.environ.get("DB_SOCKET_DIR", "/cloudsql")
connection_name = os.environ["UNIX_CONNECTION_NAME"]


db_config = {
    # 1 instance runs per invocation
    "pool_size": 1,
    # Temporarily exceeds the set pool_size if no connections are available.
    "max_overflow": 2,
    "pool_timeout": 30,  # 30 seconds
    "pool_recycle": 1800,  # 30 minutes
}

db_pool = sqlalchemy.create_engine(# 'postgresql+pg8000://trees_admin:1234@localhost/trees',
    sqlalchemy.engine.url.URL(
        drivername="postgresql",
        username=db_user,
        password=db_pass,
        database=db_name,
        port=db_port,
        # query={
        #    "host": "{}/{}".format(
        #        db_socket_dir,  # e.g. "/cloudsql"
        #        connection_name)  # i.e "<PROJECT-NAME>:<INSTANCE-REGION>:<INSTANCE-NAME>"
        #}
    ),
    **db_config
)
db_pool.dialect.description_encoding = None

# get and setup models we care for
# User = Base.classes.account_user
# Feedback = Base.classes.path_userpathfeedback
# Interaction = Base.classes.path_userpathinteraction
# Path = Base.classes.path_path
#
# UserPathScore = Base.classes.path_userpathscore
#
# Question = Base.classes.questionnaire_question
# Answer = Base.classes.questionnaire_useranswer
