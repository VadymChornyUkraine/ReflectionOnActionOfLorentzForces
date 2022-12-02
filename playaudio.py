from moviepy.audio.AudioClip import AudioArrayClip
import numpy as np
import audiovideo as audvid
from RALf1FiltrVID import RandomQ

wwrkdir_=r".\W10\\"
nama='explosion'

audio = audvid.audio_extract_audio(wwrkdir_ +nama+".mp4")
samples=audio.to_soundarray()

siz=len(samples)
print( audio.duration )

rate = int(siz/audio.duration)  # Sampling rate in samples per second.
duration =  int(audio.duration)  # Duration in seconds
samples=samples[0:rate*duration,:]


# loading video dsa gfg intro video 
clip = audvid.video_extract_video(wwrkdir_ +nama+".mp4")
clip = clip.subclip(0, duration) 
Liix=RandomQ(rate*duration)

samples_=samples[Liix,:].copy()
aaaaa=np.concatenate((samples ,samples ))
bbbbb=np.concatenate((samples_,samples_))
aaaaa_=np.asarray(aaaaa,complex)
bbbbb_=np.asarray(bbbbb,complex)
for i in range(len(samples_[0])):
    aaaaa_[:,i]=np.fft.fft(aaaaa[:,i])
    bbbbb_[:,i]=np.abs(np.fft.fft(bbbbb[:,i]))
    
ddddd=np.asarray(bbbbb_,float)
for j in range(1):
    for i in range(len(samples_[0])):
        ddddd[:,i]=np.fft.ifft(ddddd[:,i]*aaaaa_[:,i]).real
        samples_[:,i]=ddddd[:len(samples),i].copy()        
        samples_[:,i]=samples_[:,i]/np.std(samples_[:,i])*np.std(samples[:,i])

audio_stereo = AudioArrayClip(samples_, fps=rate)
clip.audio=audio_stereo
audvid.video_save(clip,wwrkdir_ +nama+"new.mp4") 

