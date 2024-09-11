def createStack():
    stack = []
    return stack

def isEmpty(stack):
   return len(stack) == 0
    
def push(stack,item):
    stack.append(item)
    print("Pushed Item : " + item)

def pop(stack):
    if isEmpty(stack):
        return "stack is Empty"
    return stack.pop()

stack = createStack()
push(stack,str(1))
push(stack,str(2))
push(stack,str(3))
push(stack,str(4))

print("Popped Item : " + pop(stack))
print("Stack after Pop : " + str(stack))