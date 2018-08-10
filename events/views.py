from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from slackclient import SlackClient
from model.chatbot import Bot

SLACK_VERIFICATION_TOKEN = getattr(settings, 'SLACK_VERIFICATION_TOKEN', None)
SLACK_BOT_USER_TOKEN = getattr(settings,
                               'SLACK_BOT_USER_TOKEN', None)
Client = SlackClient(SLACK_BOT_USER_TOKEN)

bot_list = {}
data_dir = "/home/plays/dev/tensorb_files/train7/stored_model/"


class Events(APIView):
    def post(self, request, *args, **kwargs):

        slack_message = request.data
        print("Message received", slack_message)
        if slack_message.get('token') != SLACK_VERIFICATION_TOKEN:
            return Response(status=status.HTTP_403_FORBIDDEN)

        # verification challenge
        if slack_message.get('type') == 'url_verification':
            print('GOOOD')
            return Response(data=slack_message,
                            status=status.HTTP_200_OK)
        # greet bot
        if 'event' in slack_message:
            event_message = slack_message.get('event')

            # ignore bot's own message
            if event_message.get('subtype') == 'bot_message':
                return Response(status=status.HTTP_200_OK)

            # process user's message
            user = event_message.get('user')
            text = event_message.get('text')

            if user not in bot_list:
                bot_list[user] = Bot(data_dir, user)

            channel = event_message.get('channel')
            if 'hi' in text.lower():
                bot_text = 'Hi <@{}> :wave:'.format(user)
                Client.api_call(method='chat.postMessage',
                                channel=channel,
                                text=bot_text)
                return Response(status=status.HTTP_200_OK)
            else:
                bot_text = bot_list[user].get_answer(text, user)
                Client.api_call(method='chat.postMessage',
                                channel=channel,
                                text=bot_text)
                return Response(status=status.HTTP_200_OK)

        return Response(status=status.HTTP_200_OK)
