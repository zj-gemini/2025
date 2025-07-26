import os

def cd(pwd: str, path: str) -> str:
    if os.path.isabs(path):
        result = os.path.normpath(path)
    else:
        result = os.path.normpath(os.path.join(pwd, path))
    return result

def test():
    print(cd("/home/user", "projects"))         # /home/user/projects
    print(cd("/home/user", "../etc"))           # /home/etc
    print(cd("/home/user", "/var/log"))         # /var/log
    print(cd("/home/user", "./docs"))           # /home/user/docs
    print(cd("/home/user", "../../"))           # /home

# Uncomment to run tests
# test()