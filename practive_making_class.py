class FourCal:
    def setdata(self, first,second):
        self.first = first
        self.second = second

    def sum(self):
        return self.first+self.second

    def mul(self):
        return self.first * self.second

    def div(self):
        return self.first / self.second


a=FourCal()

a.setdata(4,2)


print(a.sum())

print(a.mul())

print(a.div())
