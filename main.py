from waitress import serve
from api import app
import multiprocessing

if __name__ == "__main__":
    num_cpus = multiprocessing.cpu_count()

    threads_per_woker = max(1, num_cpus-1)

    print("Threads", threads_per_woker) 
    print("server started")
    serve(
        app,
        host='0.0.0.0',
        port = 8080,
        threads = threads_per_woker 
    )

   