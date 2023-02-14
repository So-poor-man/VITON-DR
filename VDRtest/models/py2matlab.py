import matlab.engine
import time

eng = matlab.engine.start_matlab()
time_begin = time.time()

a = eng.FCCP_demo2D()

time_end = time.time()
time = time_end - time_begin
print('time:', time)

