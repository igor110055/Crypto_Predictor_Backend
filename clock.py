from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime, timedelta
from generate_data import generate_data

sched = BlockingScheduler()


current_time = datetime.now() 
current_minute = current_time.minute
minutes_to_nearest_hour = 1 if (60 - current_minute) > 55 else 62 - current_minute
start = current_time + timedelta(minutes=minutes_to_nearest_hour )
print("Starting at: ", start)


@sched.scheduled_job("interval", start_date=start, hours=1)
def timed_job():
    print("Running: ", datetime.now())
    generate_data()


sched.start()
