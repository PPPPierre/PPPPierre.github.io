---
title: 【专业课】基于 Python 的后端开发
author: Stone SHI
date: 2021-10-07 14:18:00 +0200
categories: [Blogging, Study]
tags: [Backend, Python]
---

项目网站：https://www.cri.mines-paristech.fr/~coelho/kiva.html

# 步骤

```sh
msi@msi:~/Desktop/TP_Kiva/kiva/back-end$ make venv
python3 -m venv venv
. venv/bin/activate
pip install FlaskSimpleAuth anodb passlib bcrypt psycopg2-binary
pip install pytest requests
pip install mypy flake8
pip install wheel
Collecting FlaskSimpleAuth
  Downloading FlaskSimpleAuth-4.2.0-py3-none-any.whl (25 kB)
Collecting anodb
  Downloading anodb-3.0.0-py3-none-any.whl (4.6 kB)
Collecting passlib
  Downloading passlib-1.7.4-py2.py3-none-any.whl (525 kB)
     |████████████████████████████████| 525 kB 5.8 MB/s 
Collecting bcrypt
  Downloading bcrypt-3.2.0-cp36-abi3-manylinux2010_x86_64.whl (63 kB)
     |████████████████████████████████| 63 kB 5.5 MB/s 
Collecting psycopg2-binary
  Downloading psycopg2_binary-2.9.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.4 MB)
     |████████████████████████████████| 3.4 MB 11.6 MB/s 
Collecting flask
  Downloading Flask-2.0.2-py3-none-any.whl (95 kB)
     |████████████████████████████████| 95 kB 4.3 MB/s 
Collecting aiosql>=3.2.0
  Downloading aiosql-3.3.1-py3-none-any.whl (13 kB)
Collecting six>=1.4.1
  Downloading six-1.16.0-py2.py3-none-any.whl (11 kB)
Collecting cffi>=1.1
  Using cached cffi-1.14.6-cp38-cp38-manylinux1_x86_64.whl (411 kB)
Collecting Jinja2>=3.0
  Downloading Jinja2-3.0.2-py3-none-any.whl (133 kB)
     |████████████████████████████████| 133 kB 15.7 MB/s 
Collecting Werkzeug>=2.0
  Downloading Werkzeug-2.0.2-py3-none-any.whl (288 kB)
     |████████████████████████████████| 288 kB 8.1 MB/s 
Collecting itsdangerous>=2.0
  Downloading itsdangerous-2.0.1-py3-none-any.whl (18 kB)
Collecting click>=7.1.2
  Downloading click-8.0.1-py3-none-any.whl (97 kB)
     |████████████████████████████████| 97 kB 8.9 MB/s 
Collecting typing-extensions<4,>=3.7.4
  Downloading typing_extensions-3.10.0.2-py3-none-any.whl (26 kB)
Collecting contextlib2>=21.6.0
  Downloading contextlib2-21.6.0-py2.py3-none-any.whl (13 kB)
Collecting pycparser
  Using cached pycparser-2.20-py2.py3-none-any.whl (112 kB)
Collecting MarkupSafe>=2.0
  Using cached MarkupSafe-2.0.1-cp38-cp38-manylinux2010_x86_64.whl (30 kB)
Installing collected packages: MarkupSafe, Jinja2, Werkzeug, itsdangerous, click, flask, FlaskSimpleAuth, typing-extensions, contextlib2, aiosql, anodb, passlib, six, pycparser, cffi, bcrypt, psycopg2-binary
Successfully installed FlaskSimpleAuth-4.2.0 Jinja2-3.0.2 MarkupSafe-2.0.1 Werkzeug-2.0.2 aiosql-3.3.1 anodb-3.0.0 bcrypt-3.2.0 cffi-1.14.6 click-8.0.1 contextlib2-21.6.0 flask-2.0.2 itsdangerous-2.0.1 passlib-1.7.4 psycopg2-binary-2.9.1 pycparser-2.20 six-1.16.0 typing-extensions-3.10.0.2
Collecting pytest
  Downloading pytest-6.2.5-py3-none-any.whl (280 kB)
     |████████████████████████████████| 280 kB 5.4 MB/s 
Collecting requests
  Downloading requests-2.26.0-py2.py3-none-any.whl (62 kB)
     |████████████████████████████████| 62 kB 2.5 MB/s 
Collecting toml
  Downloading toml-0.10.2-py2.py3-none-any.whl (16 kB)
Collecting packaging
  Using cached packaging-21.0-py3-none-any.whl (40 kB)
Collecting attrs>=19.2.0
  Using cached attrs-21.2.0-py2.py3-none-any.whl (53 kB)
Collecting pluggy<2.0,>=0.12
  Downloading pluggy-1.0.0-py2.py3-none-any.whl (13 kB)
Collecting iniconfig
  Downloading iniconfig-1.1.1-py2.py3-none-any.whl (5.0 kB)
Collecting py>=1.8.2
  Downloading py-1.10.0-py2.py3-none-any.whl (97 kB)
     |████████████████████████████████| 97 kB 9.2 MB/s 
Collecting urllib3<1.27,>=1.21.1
  Downloading urllib3-1.26.7-py2.py3-none-any.whl (138 kB)
     |████████████████████████████████| 138 kB 15.4 MB/s 
Collecting charset-normalizer~=2.0.0; python_version >= "3"
  Downloading charset_normalizer-2.0.6-py3-none-any.whl (37 kB)
Collecting idna<4,>=2.5; python_version >= "3"
  Downloading idna-3.2-py3-none-any.whl (59 kB)
     |████████████████████████████████| 59 kB 19.1 MB/s 
Collecting certifi>=2017.4.17
  Downloading certifi-2021.5.30-py2.py3-none-any.whl (145 kB)
     |████████████████████████████████| 145 kB 18.0 MB/s 
Collecting pyparsing>=2.0.2
  Using cached pyparsing-2.4.7-py2.py3-none-any.whl (67 kB)
Installing collected packages: toml, pyparsing, packaging, attrs, pluggy, iniconfig, py, pytest, urllib3, charset-normalizer, idna, certifi, requests
Successfully installed attrs-21.2.0 certifi-2021.5.30 charset-normalizer-2.0.6 idna-3.2 iniconfig-1.1.1 packaging-21.0 pluggy-1.0.0 py-1.10.0 pyparsing-2.4.7 pytest-6.2.5 requests-2.26.0 toml-0.10.2 urllib3-1.26.7
Collecting mypy
  Downloading mypy-0.910-cp38-cp38-manylinux2010_x86_64.whl (22.8 MB)
     |████████████████████████████████| 22.8 MB 3.4 MB/s 
Collecting flake8
  Downloading flake8-3.9.2-py2.py3-none-any.whl (73 kB)
     |████████████████████████████████| 73 kB 4.9 MB/s 
Collecting mypy-extensions<0.5.0,>=0.4.3
  Downloading mypy_extensions-0.4.3-py2.py3-none-any.whl (4.5 kB)
Requirement already satisfied: toml in ./venv/lib/python3.8/site-packages (from mypy) (0.10.2)
Requirement already satisfied: typing-extensions>=3.7.4 in ./venv/lib/python3.8/site-packages (from mypy) (3.10.0.2)
Collecting pyflakes<2.4.0,>=2.3.0
  Downloading pyflakes-2.3.1-py2.py3-none-any.whl (68 kB)
     |████████████████████████████████| 68 kB 15.4 MB/s 
Collecting mccabe<0.7.0,>=0.6.0
  Downloading mccabe-0.6.1-py2.py3-none-any.whl (8.6 kB)
Collecting pycodestyle<2.8.0,>=2.7.0
  Downloading pycodestyle-2.7.0-py2.py3-none-any.whl (41 kB)
     |████████████████████████████████| 41 kB 1.9 MB/s 
Installing collected packages: mypy-extensions, mypy, pyflakes, mccabe, pycodestyle, flake8
Successfully installed flake8-3.9.2 mccabe-0.6.1 mypy-0.910 mypy-extensions-0.4.3 pycodestyle-2.7.0 pyflakes-2.3.1
Collecting wheel
  Downloading wheel-0.37.0-py2.py3-none-any.whl (35 kB)
Installing collected packages: wheel
Successfully installed wheel-0.37.0
```

```sh
msi@msi:~/Desktop/TP_Kiva/kiva/back-end$ source venv/bin/activate
(venv) msi@msi:~/Desktop/TP_Kiva/kiva/back-end$  
```

两行代码完成环境配置。

接下来设置 sql 权限，只需要设置一次即可，之后启动虚拟机不需要重新设置。

```sh
(venv) msi@msi:~/Desktop/TP_Kiva/kiva/back-end$ sudo pg_createcluster 13 main
Creating new PostgreSQL cluster 13/main ...
/usr/lib/postgresql/13/bin/initdb -D /var/lib/postgresql/13/main --auth-local peer --auth-host md5
Les fichiers de ce système de bases de données appartiendront à l'utilisateur « postgres ».
Le processus serveur doit également lui appartenir.

L'instance sera initialisée avec la locale « fr_FR.UTF-8 ».
L'encodage par défaut des bases de données a été configuré en conséquence
avec « UTF8 ».
La configuration de la recherche plein texte a été initialisée à « french ».

Les sommes de contrôle des pages de données sont désactivées.

correction des droits sur le répertoire existant /var/lib/postgresql/13/main... ok
création des sous-répertoires... ok
sélection de l'implémentation de la mémoire partagée dynamique...posix
sélection de la valeur par défaut pour max_connections... 100
sélection de la valeur par défaut pour shared_buffers... 128MB
sélection du fuseau horaire par défaut... Europe/Paris
création des fichiers de configuration... ok
lancement du script bootstrap...ok
exécution de l'initialisation après bootstrap... ok
synchronisation des données sur disque... ok

Succès. Vous pouvez maintenant lancer le serveur de bases de données en utilisant :

    pg_ctlcluster 13 main start

Ver Cluster Port Status Owner    Data directory              Log file
13  main    5432 down   postgres /var/lib/postgresql/13/main /var/log/postgresql/postgresql-13-main.log
```

```sh
(venv) msi@msi:~/Desktop/TP_Kiva/kiva/back-end$ sudo su - postgres
postgres@msi:~$
```

```sh
postgres@msi:~$ pg_lsclusters
Ver Cluster Port Status Owner    Data directory              Log file
13  main    5432 down   postgres /var/lib/postgresql/13/main /var/log/postgresql/postgresql-13-main.log
```

```sh
postgres@msi:~$ pg_ctlcluster 13 main start
Warning: the cluster will not be running as a systemd service. Consider using systemctl:
  sudo systemctl start postgresql@13-main
```

```sh
postgres@msi:~$ createuser -s msi
```

```sh
postgres@msi:~$ createdb -O msi msi
```

```sh
postgres@msi:~$ exit
déconnexion
(venv) msi@msi:~/Desktop/TP_Kiva/kiva/back-end$ psql
psql (13.4 (Ubuntu 13.4-1.pgdg20.04+1))
Saisissez « help » pour l'aide.

msi=# 
```

```sh
(venv) msi@msi:~/Desktop/TP_Kiva/kiva/back-end$ make run
branch=$(git rev-parse --abbrev-ref HEAD)
rev=$(git log -1|head -1|cut -d' ' -f2)
date=$(git log -1 --date=iso8601 | sed -n 3p | cut -c 9-27 | tr ' ' 'Z')
echo -n "$branch $rev $date" > VERSION
createdb kiva
psql -f drop.sql -f create.sql -f data.sql kiva | tee .postgres
export FLASK_APP=app.py FLASK_ENV=development
export APP_CONFIG='pg.conf'
flask run --host="0.0.0.0" >> app.log 2>&1 &
echo $! > app.pid
```

```sh
(venv) msi@msi:~/Desktop/TP_Kiva/kiva/back-end$ curl -i -X GET http://0.0.0.0:5000/version
HTTP/1.0 200 OK
Content-Type: application/json
Content-Length: 318
Server: Werkzeug/2.0.2 Python/3.8.10
Date: Thu, 07 Oct 2021 12:41:29 GMT

{
  "app": "kiva", 
  "auth": "none", 
  "branch": "master", 
  "commit": "595a4e9eee1d9ffec6fd8f182f706e9dcb6f77ba", 
  "date": "2021-10-04Z16:35:39", 
  "db": "postgres", 
  "now": "Thu, 07 Oct 2021 12:41:29 GMT", 
  "started": "2021-10-07 14:40:52.741970", 
  "user": null, 
  "variant": "anodb", 
  "version": 5
}
```




