import os
import sys
import os.path

def pic_list(dir_name):
    fp = open(dir_name + '.txt', 'w')
    cdir = os.path.abspath(dir_name)
    dirs = os.listdir(cdir)
    for d in dirs:
        dir_path = os.path.join(cdir, d)
        if d != '.' and d != '..' and os.path.isdir(dir_path):
            pics = os.listdir(dir_path)
            fp.write('%s %d\n' % (str(d), len(pics)))
            for p in pics:
                fp.write('%s\n' % str(os.path.join(dir_path, p)))                
    fp.close()
                
if __name__ == '__main__':
    pic_list('train')
    pic_list('test')
                    
