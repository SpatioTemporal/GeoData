
import json
from sortedcontainers import SortedDict,SortedList

sd = SortedDict()

sd[1] = 'a'
sd[2] = 'b'
sd[3] = 'c'
sd[4] = SortedList()
sd[5] = 'd'
sd[6] = 'e'
sd[7] = 'f'

sd[4].add("a")
sd[4].add("c")
sd[4].add("b")

print('sd keys:   ',sd.keys())
print('sd keys 1: ',sd[1])

idx = 2; print('sd keys contains ',idx,sd.__contains__(idx))
idx = 8; print('sd keys contains ',idx,sd.__contains__(idx))

it = sd.irange(3,6)
for k in it:
    print(k)


it = sd.irange(3,6)
print('len: ',len(list(it)))
it = sd.irange(3,6)
print(it)
print(next(it))
print(next(it))
print(next(it))


it = sd.irange()

print('\n100')
for k in sd.keys():
    print([i for i in sd[k]])
print('\n200')
str = ""
for k in sd.keys():
  str = str+ json.dumps( {k:[i for i in sd[k]]} )

print('str: ',str)

