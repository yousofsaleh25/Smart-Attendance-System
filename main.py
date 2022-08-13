
import pandas as pd
from datetime import datetime
import os
import arabic_reshaper
from bidi.algorithm import get_display
from absl import app, flags
from absl.flags import FLAGS
from attendence import take_attendence
flags.DEFINE_boolean('show_video', True, "open window with result video")
flags.DEFINE_string('time', 'automatic', "open window with result video")
flags.DEFINE_string('video', './test_video/test.mp4', "path input video for lecture")
flags.DEFINE_string('cource_name', 'Machine_Learning.csv', "cource name (math, physics, ..etc)")
flags.DEFINE_string('output', "attendence", "output csv files")

def main(_argv):
    if FLAGS.time == 'automatic':
        now = datetime.now()
        current_time = now.strftime("%Y/%m/%d-%H:%M:%S")
    else:
        current_time = FLAGS.time
        #print(current_time)
    att = take_attendence(FLAGS.video, show_video=FLAGS.show_video)
    os.makedirs(FLAGS.output, exist_ok=True)
    course_file = os.path.join(FLAGS.output, FLAGS.cource_name)
    att = {get_display(arabic_reshaper.reshape(k)):v for k,v in sorted(att.items(), key=lambda item: item[0])}
    #print(att)
    names = list(att.keys())
    a = list(att.values())
    
    try:
        df = pd.read_csv(course_file)
        df_2 = pd.DataFrame([[list(df["Section Number"])[-1]+1] + [current_time] + a], 
                            columns=["Section Number"] + ["Section Date"] + names )
        df = df.append(df_2)
    except:
        df = pd.DataFrame(data=[[1] + [current_time] + a ], 
                          columns=["Section Number"] + ["Section Date"] + names )
        
    df.to_csv(course_file, index=False, encoding='utf-8-sig')
    print(f"{FLAGS.cource_name.split('.')[0]} Course Attendence File Is updated Successfully")
    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
