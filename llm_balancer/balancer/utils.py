

class CircularList:

    def __init__(self, max_len):
        self.list = []
        self.max_len = max_len
        self.position = -1

    def length(self):
        return len(self.list)

    def append(self, element):
        if len(self.list) == self.max_len:
            self.position = (self.position + 1) % self.max_len
            self.list[self.position] = element
            return self.position
        self.list.append(element)
        return len(self.list) - 1

    def clear(self):
        self.list.clear()
