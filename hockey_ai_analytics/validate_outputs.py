import hashlib
import json
import subprocess
import cv2
import imageio_ffmpeg
from make_review import ROOT,load_video
from tracker_v4 import run


def main():
    frames,pts=load_video(ROOT/'evidence/source.mov')
    a=run(frames,pts);b=run(frames,pts)
    assert a==b,'V4 nondeterministic'
    video,render_pts=load_video(ROOT/'results/qa_comparison.mp4')
    assert len(video)==len(frames)==424
    pts_error=max(abs(x-y) for x,y in zip(pts,render_pts))
    assert pts_error<.001
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    audio_hashes=[]
    for file in [ROOT/'evidence/source.mov',ROOT/'results/qa_comparison.mp4']:
        r=subprocess.run([ffmpeg,'-v','error','-i',str(file),'-map','0:a:0','-c','copy','-f','hash','-hash','sha256','-'],capture_output=True,text=True,check=True)
        audio_hashes.append(r.stdout.strip())
    assert audio_hashes[0]==audio_hashes[1],'Audio changed'
    crop=subprocess.run([ffmpeg,'-y','-v','error','-i',str(ROOT/'results/qa_comparison.mp4'),'-vf','crop=848:464:1696:0',
        '-map','0:v:0','-map','0:a:0?','-fps_mode','vfr','-enc_time_base','1:60000','-video_track_timescale','60000',
        '-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',str(ROOT/'results/v4_experimental.mp4')],capture_output=True,text=True,check=True)
    vv,vp=load_video(ROOT/'results/v4_experimental.mp4')
    assert len(vv)==424 and max(abs(x-y) for x,y in zip(pts,vp))<.001
    for i in [90,135,215,216,291,375,423]:
        cv2.imwrite(str(ROOT/f'evidence/render_check_{i}.jpg'),video[i])
    result=dict(v4_repeat_predictions_identical=True,decoded_comparison_frames=len(video),
                decoded_v4_frames=len(vv),max_render_pts_error_s=pts_error,audio_packet_hash_equal=True,
                audio_hash=audio_hashes[0],tests='11 unit tests; synthetic gate tests are not real hockey validation')
    (ROOT/'results/output_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
