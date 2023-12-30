
@echo OFF
rem Update ml_env conda environment

call conda env update --prune -f environment.yml

call pip-compile -v requirements/prod.in && pip-compile -v requirements/dev.in
call pip-sync requirements/prod.txt requirements/dev.txt