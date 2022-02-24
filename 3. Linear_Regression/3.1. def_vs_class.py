
'''
def는 add를 구현할 때
사용할 객체마다 함수를 만들어 줘야함
'''

cal1 = 0
cal2 = 0
def add1(num):
    global cal1
    cal1 += num
    return cal1

def add2(num):
    global cal2
    cal2 += num
    return cal2

add1(8)
add2(6)

print(cal1)
print(cal2)

'''
클래스는 함수를 구현하고
객체만 찍어주면 함수 사용 가능
'''

class cal():
    def __init__(self) -> None:
        self.result = 0
        
    def add(self,num):
        self.result += num
        return self.result
    


a1 = cal()
a2 = cal()


print(a1.add(4))
print(a2.add(7))