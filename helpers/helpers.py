import time
from colorama import init, Fore, Back, Style

init()

def console_print(message):
    tid = (time.strftime("%H:%M:%S" ,time.localtime()))
    # print(f"{tid}: {message}")
    print(Fore.GREEN + tid + ": " + Fore.RESET + message)

def get_object_name(object):
    return type(object).__name__