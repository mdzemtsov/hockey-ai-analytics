#!/bin/zsh
set -e
cd "${0:A:h}"
if [[ -x /private/tmp/hockey-analytics-env/bin/python ]]; then
  PY=/private/tmp/hockey-analytics-env/bin/python
else
  if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
  fi
  PY="$PWD/.venv/bin/python"
fi
"$PY" - <<'PY'
import urllib.request,subprocess,sys,webbrowser,time
url='http://127.0.0.1:8765'
try:
    with urllib.request.urlopen(url+'/api/bootstrap',timeout=2) as r:
        assert 'demo_available' in r.read().decode()
except Exception:
    subprocess.Popen([sys.executable,'server.py','--port','8765'],stdout=open('server.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(60):
        try:
            urllib.request.urlopen(url+'/api/bootstrap',timeout=1).close();break
        except Exception:time.sleep(.5)
    else:raise SystemExit('Не удалось запустить. Подробности в server.log')
webbrowser.open(url)
PY
