from google.cloud import translate

def translate_text(text_list=["YOUR_TEXT_TO_TRANSLATE"], project_id="YOUR_PROJECT_ID"):
   """Translating Text."""

   client = translate.TranslationServiceClient()

   location = "global"

   parent = f"projects/{project_id}/locations/{location}"

   # Detail on supported types can be found here:
   # https://cloud.google.com/translate/docs/supported-formats
   response = client.translate_text(
       request={
           "parent": parent,
           "contents": text_list,
           "mime_type": "text/plain",  # mime types: text/plain, text/html
           "source_language_code": "en-US",
           "target_language_code": "ta-IN",
       }
   )

   # Display the translation for each input text provided
   # for translation in response.translations:
   #     print("Translated text: {}".format(translation.translated_text))

   translations = [response.translations[i].translated_text for i in range(len(response.translations))]
   return translations


import os

os.environ[
   'GOOGLE_APPLICATION_CREDENTIALS'] = 'D:\CMU\Courses\\11737\Project\code\Translation-e1c8b86fb308.json'

# '''
# reminder/set_reminder  19:31:reminder/todo,32:41:datetime Remind me about my trip to D.C. Next week  en_XX  {"tokenizations":[{"tokens":["remind","me","about","my","trip","to","d.c",".","next","week"],"tokenSpans":[{"start":0,"length":6},{"start":7,"length":2},{"start":10,"length":5},{"start":16,"length":2},{"start":19,"length":4},{"start":24,"length":2},{"start":27,"length":3},{"start":30,"length":1},{"start":32,"length":4},{"start":37,"length":4}],"tokenizerType":2,"normalizerType":1}]}
#
# '''

s = ['set an alarm to remind me to tell my wife that I love her']
# # s = 'Add call John to Reminders'
#
#
# imp_ranges = [[4, 13], [17, 26]]
# imp_ranges = [[19, 29], [30, 39]]
#
#
# print(s)
# t = translate_text(s, 'translation-296703')
# print(t)