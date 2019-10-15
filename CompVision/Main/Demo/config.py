import os

raw_data = "/Data/"
BASE_DIR = 'Main'

VideoOut_file = 'C:/Storage/DDG_FaceRecog/CompVision/Main/Data/out.avi'

def get_base_dir_by_name(name):
    path = os.getcwd()
    lastchar = path.find(name) + len(name)
    return os.getcwd()[0:lastchar]

base_dir = get_base_dir_by_name(BASE_DIR).replace("\\","/")