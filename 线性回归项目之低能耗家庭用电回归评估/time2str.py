import time
def time2str(current_time):
    t = time.strptime(current_time, '%Y/%m/%d  %H:%M:%S', )
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)

# a= time2str('2016/1/11  17:20:00')
# print(a)
if __name__ == '__main__':
    # t = '2016/1/11  17:20:00'
    a = time2str('2016/1/11  17:20:00')
    print(a)