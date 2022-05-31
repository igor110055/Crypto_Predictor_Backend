import threading
from app.main import app
from app.generate_data import setup_data_generate

if __name__ == "__main__":
  backtestThreads = []
  backtestThread_app = threading.Thread(
      target=app.run
  )
  backtestThreads.append(backtestThread_app)
  backtestThread_app.start()
  backtestThread_data = threading.Thread(
      target=setup_data_generate
  )
  backtestThreads.append(backtestThread_data)
  backtestThread_data.start()
  for backtestThread in backtestThreads:
      backtestThread.join()