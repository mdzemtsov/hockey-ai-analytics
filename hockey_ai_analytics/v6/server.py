"""Private loopback-only selector and local analysis service."""
import argparse,json,threading,uuid,secrets,shutil,time,mimetypes,os,re,traceback,socketserver
from pathlib import Path
from urllib.parse import urlparse,parse_qs,unquote
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from engine import ROOT,prepare,analyse
from gate import validate_selections
from coach import run_coach,validate_request
import cv2
DATA=ROOT/'work';DATA.mkdir(exist_ok=True)
VIDEOS={};JOBS={};TOKEN=secrets.token_urlsafe(32);LOCK=threading.RLock()


def status_change(record,stage,fraction,message):
    with LOCK:record.update(state=stage,progress=max(0,min(1,fraction)),message=message)


def ready_video(vid):
    r=VIDEOS[vid]
    try:
        meta=prepare(r['source'],r['folder'],lambda *a:status_change(r,*a))
        with LOCK:r.update(state='ready',progress=1,message='Видео готово. Выберите кадр и игрока.',meta=meta)
    except Exception as e:
        with LOCK:r.update(state='error',message=str(e)[:400])
        (r['folder']/'error.log').write_text(traceback.format_exc())


def start_job(job):
    video=VIDEOS[job['video_id']]
    try:
        if job.get('kind')=='coach':
            report=run_coach(video,job['folder'],job['config'],lambda *a:status_change(job,*a),job['cancel'])
        else:
            report=analyse(video['source'],video['meta'],video['folder'],job['folder'],job['selections'],
                lambda *a:status_change(job,*a),job['cancel'])
        with LOCK:job.update(report,progress=1)
    except InterruptedError as e:
        with LOCK:job.update(state='cancelled',message=str(e))
    except Exception as e:
        with LOCK:job.update(state='error',message='Обработка не завершена. '+str(e)[:300])
        (job['folder']/'error.log').write_text(traceback.format_exc())


def public(record):
    return {k:v for k,v in record.items() if k not in ['folder','source','cancel']}


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def base_headers(self):
        self.send_header('X-Content-Type-Options','nosniff');self.send_header('X-Frame-Options','DENY')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; object-src 'none'")
    def json(self,status,obj):
        data=json.dumps(obj,ensure_ascii=False).encode()
        self.send_response(status);self.base_headers();self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def allowed(self,mutate=False):
        port=self.server.server_port
        if self.headers.get('Host') not in [f'127.0.0.1:{port}',f'localhost:{port}']:raise PermissionError('Недопустимый адрес запроса.')
        if mutate:
            origin=self.headers.get('Origin')
            if origin and origin not in [f'http://127.0.0.1:{port}',f'http://localhost:{port}']:raise PermissionError('Запрос с другого сайта запрещён.')
            if self.headers.get('X-Session-Token')!=TOKEN:raise PermissionError('Обновите страницу и повторите действие.')
    def body(self):
        size=int(self.headers.get('Content-Length','0'))
        if size>1000000:raise ValueError('Запрос слишком большой.')
        return json.loads(self.rfile.read(size) or b'{}')
    def file(self,path):
        if not path.is_file():self.json(404,{'error':'Файл ещё не готов.'});return
        size=path.stat().st_size;start=0;end=size-1;code=200
        if self.headers.get('Range'):
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',self.headers['Range'])
            if not match:self.json(416,{'error':'Неверный диапазон.'});return
            a,b=match.groups()
            if a:start=int(a);end=int(b) if b else size-1
            elif b:start=max(0,size-int(b))
            else:self.json(416,{'error':'Пустой диапазон.'});return
            end=min(end,size-1);code=206
            if start>end:self.json(416,{'error':'Диапазон за пределами файла.'});return
        self.send_response(code);self.base_headers();self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream');self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
        if code==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        if self.command=='HEAD':return
        try:
            with path.open('rb') as f:
                f.seek(start);remaining=end-start+1
                while remaining:
                    data=f.read(min(1024*1024,remaining))
                    if not data:break
                    self.wfile.write(data);remaining-=len(data)
        except (BrokenPipeError,ConnectionResetError):pass
    def do_HEAD(self):self.do_GET()
    def do_GET(self):
        try:
            self.allowed();url=urlparse(self.path);p=url.path
            if p=='/api/bootstrap':self.json(200,dict(token=TOKEN,demo_available=(ROOT.parent/'evidence/source.mov').exists()));return
            if p.startswith('/api/videos/'):
                vid=p.split('/')[-1]
                if vid not in VIDEOS:self.json(404,{'error':'Видео не найдено. Загрузите его снова.'});return
                self.json(200,public(VIDEOS[vid]));return
            if p.startswith('/api/jobs/'):
                jid=p.split('/')[-1]
                if jid not in JOBS:self.json(404,{'error':'Задача не найдена.'});return
                self.json(200,public(JOBS[jid]));return
            if p=='/api/frame':
                q=parse_qs(url.query);vid=q.get('video',[''])[0];i=int(q.get('index',['0'])[0]);r=VIDEOS.get(vid)
                if not r or r['state']!='ready':raise ValueError('Видео ещё готовится.')
                if not 0<=i<r['meta']['frame_count']:raise ValueError('Неверный номер кадра.')
                cap=cv2.VideoCapture(str(r['source']));cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,f=cap.read();cap.release()
                if not ok:raise ValueError('Кадр не удалось прочитать.')
                ok,data=cv2.imencode('.jpg',f,[cv2.IMWRITE_JPEG_QUALITY,96])
                self.send_response(200);self.base_headers();self.send_header('Content-Type','image/jpeg');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data.tobytes());return
            parts=p.strip('/').split('/')
            if len(parts)==3 and parts[0]=='media':
                ident,name=parts[1:]
                if ident in JOBS and JOBS[ident].get('kind')=='coach' and (name in ['coach.json','observations.json','observations.csv'] or re.fullmatch(r'frame_\d+\.jpg|voice_\d+_(en|ru)\.m4a',name)):
                    self.file(JOBS[ident]['folder']/name);return
                if name=='preview.mp4' and ident in VIDEOS:self.file(VIDEOS[ident]['folder']/name);return
                if name in ['result.mp4','target.csv','report.json','selection.json','render_validation.json','provenance.json'] and ident in JOBS:self.file(JOBS[ident]['folder']/name);return
                self.json(404,{'error':'Файл не найден.'});return
            static={'/coach':'coach.html','/coach.js':'coach.js','/coach.css':'coach.css','/':'index.html','/app.js':'app.js','/style.css':'style.css'}
            if p in static:self.file(ROOT/'web'/static[p]);return
            self.json(404,{'error':'Не найдено.'})
        except PermissionError as e:self.json(403,{'error':str(e)})
        except (ValueError,KeyError) as e:self.json(400,{'error':str(e)})
        except Exception:self.json(500,{'error':'Не удалось выполнить запрос.'})
    def do_POST(self):
        try:
            self.allowed(True);p=urlparse(self.path).path
            if p in ['/api/upload','/api/demo']:
                vid=uuid.uuid4().hex;folder=DATA/'videos'/vid;folder.mkdir(parents=True);source=folder/'source.mov'
                if p=='/api/demo':
                    original=ROOT.parent/'evidence/source.mov'
                    if not original.exists():raise ValueError('Тестовый эпизод недоступен.')
                    shutil.copyfile(original,source);name='Тестовый эпизод · 14 секунд'
                else:
                    size=int(self.headers.get('Content-Length','0'))
                    if not 0<size<=8*1024**3:raise ValueError('Выберите видео размером до 8 ГБ.')
                    name=Path(unquote(self.headers.get('X-Filename','video.mov'))).name
                    name=''.join(c for c in name if c.isprintable())[:180]
                    remaining=size
                    try:
                        with source.open('wb') as f:
                            while remaining:
                                data=self.rfile.read(min(1024*1024,remaining))
                                if not data:raise ValueError('Загрузка прервалась. Попробуйте снова.')
                                f.write(data);remaining-=len(data)
                    except Exception:shutil.rmtree(folder,ignore_errors=True);raise
                record=dict(id=vid,name=name,folder=folder,source=source,state='prepare',progress=0,message='Проверяю видео')
                VIDEOS[vid]=record;threading.Thread(target=ready_video,args=(vid,),daemon=True).start();self.json(202,{'video_id':vid});return
            if p=='/api/coach':
                body=self.body();vid=body.get('video_id');video=VIDEOS.get(vid)
                if not video or video['state']!='ready':raise ValueError('Open a video first.')
                config=validate_request(body,video['meta'])
                jid=uuid.uuid4().hex;folder=DATA/'jobs'/jid;folder.mkdir(parents=True)
                job=dict(id=jid,video_id=vid,folder=folder,kind='coach',config=config,state='queued',progress=0,message='Preparing coaching review',cancel=threading.Event())
                JOBS[jid]=job;threading.Thread(target=start_job,args=(job,),daemon=True).start();self.json(202,{'job_id':jid});return
            if p=='/api/analyze':
                body=self.body();vid=body.get('video_id');video=VIDEOS.get(vid)
                if not video or video['state']!='ready':raise ValueError('Сначала загрузите и дождитесь подготовки видео.')
                selections=validate_selections(body.get('selections'),video['meta'])
                jid=uuid.uuid4().hex;folder=DATA/'jobs'/jid;folder.mkdir(parents=True)
                job=dict(id=jid,video_id=vid,folder=folder,state='queued',progress=0,message='Ожидаю начала обработки',selections=selections,cancel=threading.Event())
                JOBS[jid]=job;threading.Thread(target=start_job,args=(job,),daemon=True).start();self.json(202,{'job_id':jid});return
            if p=='/api/cancel':
                body=self.body();job=JOBS.get(body.get('job_id'))
                if not job:raise ValueError('Задача не найдена.')
                job['cancel'].set();self.json(200,{'ok':True});return
            self.json(404,{'error':'Не найдено.'})
        except PermissionError as e:self.json(403,{'error':str(e)})
        except (ValueError,KeyError,json.JSONDecodeError) as e:self.json(400,{'error':str(e)})
        except Exception:self.json(500,{'error':'Не удалось начать обработку.'})


class LocalServer(ThreadingHTTPServer):
    def server_bind(self):
        # Avoid blocking reverse-DNS lookup for a loopback-only service.
        socketserver.TCPServer.server_bind(self)
        self.server_name='localhost';self.server_port=self.server_address[1]


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);a=parser.parse_args()
    httpd=LocalServer(('127.0.0.1',a.port),Handler)
    print(f'HOCKEY_LOCAL_URL=http://127.0.0.1:{httpd.server_port}',flush=True)
    httpd.serve_forever()
