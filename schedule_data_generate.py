from datetime import datetime
import time
from app.helpers import generate_data, get_minutes_to_nearest_hr

if __name__ == "__main__":
  # This data scheduler will be handled by heroku data generate.
  while True:
    minutes_to_nearest_hour = get_minutes_to_nearest_hr()

    print("Minutes to next execution: ", minutes_to_nearest_hour )
    # Socketio sleeps for as long as required

    for i in range((minutes_to_nearest_hour) * 6):
        time.sleep(10)

    start = datetime.now()
    generate_data()
    end = datetime.now()
    print("Took: ", end - start)