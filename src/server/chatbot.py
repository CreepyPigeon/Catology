import aiml

def do_something():
    kernel = aiml.Kernel()
    kernel.learn("chat_rules.aiml")

    while True:
        print('YOU: ')
        message = input()
        if message == 'Exit':
            break
        response = kernel.respond(message)
        print(f"Bot: {response}")


if __name__ == '__main__':
    do_something()
