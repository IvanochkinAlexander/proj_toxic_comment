import subprocess
import os
import telegram
import time
import pandas as pd

def send_to_telegram(text):

    """Send appropriate links to telegram channel"""

    bot = telegram.Bot(token='523412387:AAHhEckKtZiCoSG6Pd3ZGtp4-JbL06I8H2E')
    # chat_id = -1001371737931
    chat_id = 169719023
    bot.send_message(chat_id=chat_id, text=text)
    time.sleep(5)

process = subprocess.Popen("python convert_non_ascii.py",shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = process.communicate()
errcode = process.returncode

send_to_telegram('converted to ascii')
time.sleep(5)


process = subprocess.Popen("python preproc_replace.py",shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = process.communicate()
errcode = process.returncode

send_to_telegram('replaced wrong text')
time.sleep(5)

python3_command = "python2.7 spell_checker.py"  # launch your python2 script using bash
process = subprocess.Popen(python3_command.split(), stdout=subprocess.PIPE)
output, error = process.communicate()  # receive output from the python2 script

send_to_telegram('spell checking finished')
time.sleep(5)

process = subprocess.Popen("python detect_lang.py",shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = process.communicate()
errcode = process.returncode

send_to_telegram('languages detected')
time.sleep(5)

process = subprocess.Popen("python preproc_lemm.py",shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = process.communicate()
errcode = process.returncode

send_to_telegram('lemmatization with spacy finished')
time.sleep(5)

process = subprocess.Popen("python delete_dup_letters.py",shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = process.communicate()
errcode = process.returncode

send_to_telegram('deleted dup letters')
time.sleep(5)

process = subprocess.Popen("python manual_spell_check.py",shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = process.communicate()
errcode = process.returncode

send_to_telegram('Replacing words with dictionary finished')
time.sleep(5)

process = subprocess.Popen("python manual_spell_check.py",shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = process.communicate()
errcode = process.returncode

send_to_telegram('Replacing words with dictionary finished')
time.sleep(5)

process = subprocess.Popen("python search_bad_words.py",shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = process.communicate()
errcode = process.returncode

send_to_telegram('Bad words replaced')
time.sleep(5)
