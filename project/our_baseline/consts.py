LABEL_PAD_INDEX = 999
PRINT_EVERY = 100

intent_set = ['weather/find', 'alarm/set_alarm', 'alarm/show_alarms', 'reminder/set_reminder', 'alarm/modify_alarm',
              'weather/checkSunrise', 'weather/checkSunset', 'alarm/snooze_alarm', 'alarm/cancel_alarm',
              'reminder/show_reminders', 'reminder/cancel_reminder', 'alarm/time_left_on_alarm']
slot_set = ['O', 'B-weather/noun', 'I-weather/noun', 'B-location', 'I-location', 'B-datetime', 'I-datetime',
            'B-weather/attribute', 'I-weather/attribute', 'B-reminder/todo', 'I-reminder/todo',
            'B-alarm/alarm_modifier', 'B-reminder/noun', 'B-reminder/recurring_period', 'I-reminder/recurring_period',
            'B-reminder/reference', 'I-reminder/noun', 'B-reminder/reminder_modifier', 'I-reminder/reference',
            'I-reminder/reminder_modifier', 'B-weather/temperatureUnit', 'I-alarm/alarm_modifier',
            'B-alarm/recurring_period', 'I-alarm/recurring_period']