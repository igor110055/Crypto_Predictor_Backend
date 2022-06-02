from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime, timedelta
from app.generate_data import generate_data

sched = BlockingScheduler()

start = datetime.now() + timedelta(minutes=1)
print("Yet to: ", datetime.now())
print(start)


@sched.scheduled_job("interval", start_date=start, hours=1)
def timed_job():
    print("Running: ", datetime.now())
    generate_data()


sched.start()
