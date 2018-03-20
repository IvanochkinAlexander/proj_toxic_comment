import telegram

def send_to_telegram(text):

    """Send appropriate links to telegram channel"""

    bot = telegram.Bot(token='523412387:AAHhEckKtZiCoSG6Pd3ZGtp4-JbL06I8H2E')
    chat_id =  -1001371737931
    try:
        bot.send_message(chat_id=chat_id, text=text)
    except:
        print('Telegram request failed.')
